import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import open_clip


# Project paths
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"

if not ARTIFACTS_DIR.is_dir():
    ARTIFACTS_DIR = ROOT

CHECKPOINTS_DIR = ROOT / "checkpoints"

CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
PROTOTYPES_PATH = ARTIFACTS_DIR / "experiment2_prototypes.pt"


class OneShotClassifier:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # Load model configuration
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.class_names = self.config["class_names"]

        # Load exported one-shot prototypes
        saved = torch.load(
            PROTOTYPES_PATH,
            map_location="cpu",
            weights_only=True
        )

        if list(saved["class_names"]) != self.class_names:
            raise ValueError("Prototype class order mismatch.")

        self.resnet_prototypes = saved[
            "resnet_prototypes"
        ].float().to(self.device)

        self.siglip_prototypes = saved[
            "siglip_prototypes"
        ].float().to(self.device)

        self.resnet_weight = self.config[
            "fusion_weights"
        ]["resnet"]

        self.siglip_weight = self.config[
            "fusion_weights"
        ]["siglip"]

        # Load pretrained feature extractors
        self.load_resnet()
        self.load_siglip()

        print("One-shot classifier initialized successfully.")
        print(f"Number of classes: {len(self.class_names)}")


    def load_resnet(self):

        CHECKPOINTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint_path = (
            CHECKPOINTS_DIR /
            "resnet50_places365.pth.tar"
        )

        # Download pretrained checkpoint if unavailable
        if (
            not checkpoint_path.exists()
            or checkpoint_path.stat().st_size < 1_000_000
        ):

            print("Downloading ResNet-50 Places365 weights...")

            temporary_path = checkpoint_path.with_suffix(".part")

            try:
                urlretrieve(
                    self.config["resnet_checkpoint_url"],
                    temporary_path
                )

                if temporary_path.stat().st_size < 1_000_000:
                    raise RuntimeError("Incomplete checkpoint download.")

                temporary_path.replace(checkpoint_path)

            except Exception as error:

                temporary_path.unlink(missing_ok=True)

                raise RuntimeError(
                    "Unable to download Places365 checkpoint."
                ) from error

        print("Loading ResNet-50 Places365...")

        self.resnet = models.resnet50(
            weights=None,
            num_classes=365
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        state_dict = checkpoint.get(
            "state_dict",
            checkpoint
        )

        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        self.resnet.load_state_dict(
            state_dict,
            strict=True
        )

        # Remove classification layer
        self.resnet.fc = nn.Identity()

        self.resnet = self.resnet.to(
            self.device
        ).eval()

        # Original Experiment 2 preprocessing
        self.resnet_transform = transforms.Compose([
            transforms.Resize(
                tuple(self.config["resnet_resize"])
            ),
            transforms.CenterCrop(
                self.config["resnet_center_crop"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config["resnet_mean"],
                std=self.config["resnet_std"]
            )
        ])

        print("ResNet-50 loaded successfully.")


    def load_siglip(self):

        print("Loading SigLIP 2 Base...")

        (
            self.siglip,
            _,
            self.siglip_transform
        ) = open_clip.create_model_and_transforms(
            self.config["siglip_model"],
            pretrained=self.config["siglip_pretrained"],
            device="cpu",
            cache_dir=str(CHECKPOINTS_DIR)
        )

        self.siglip = self.siglip.to(
            self.device
        ).eval()

        print("SigLIP 2 loaded successfully.")


    @staticmethod
    def normalize_scores(scores):

        return (
            scores - scores.mean(dim=1, keepdim=True)
        ) / scores.std(
            dim=1,
            keepdim=True
        ).clamp_min(1e-6)


    @torch.inference_mode()
    def predict_batch(self, images):

        # Apply model-specific preprocessing
        resnet_inputs = torch.stack([
            self.resnet_transform(image.convert("RGB"))
            for image in images
        ]).to(self.device)

        siglip_inputs = torch.stack([
            self.siglip_transform(image.convert("RGB"))
            for image in images
        ]).to(self.device)

        # Use FP16 autocast only when CUDA is available
        precision_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            )
            if self.device.type == "cuda"
            else nullcontext()
        )

        # Extract image embeddings
        with precision_context:

            resnet_features = self.resnet(
                resnet_inputs
            )

            siglip_features = self.siglip.encode_image(
                siglip_inputs
            )

        # L2 feature normalization
        resnet_features = F.normalize(
            resnet_features.float(),
            dim=1
        )

        siglip_features = F.normalize(
            siglip_features.float(),
            dim=1
        )

        # Cosine similarity with class prototypes
        resnet_scores = (
            resnet_features @ self.resnet_prototypes.T
        )

        siglip_scores = (
            siglip_features @ self.siglip_prototypes.T
        )

        # Per-image z-score normalization
        resnet_scores = self.normalize_scores(
            resnet_scores
        )

        siglip_scores = self.normalize_scores(
            siglip_scores
        )

        # Fixed equal-weight score fusion
        fused_scores = (
            self.resnet_weight * resnet_scores
            + self.siglip_weight * siglip_scores
        )

        # Predicted class indices
        predictions = fused_scores.argmax(dim=1)

        return predictions.cpu(), fused_scores.cpu()


    def predict(self, image_path):

        with Image.open(image_path) as image:

            predictions, scores = self.predict_batch(
                [image.convert("RGB")]
            )

        predicted_index = predictions[0].item()

        return {
            "predicted_class": self.class_names[
                predicted_index
            ],
            "class_index": predicted_index,
            "fusion_score": scores[
                0, predicted_index
            ].item()
        }


def main():

    parser = argparse.ArgumentParser(
        description="One-shot image classification using Experiment 2."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    classifier = OneShotClassifier()

    result = classifier.predict(image_path)

    print("\n" + "=" * 50)
    print("IMAGE CLASSIFICATION RESULT")
    print("=" * 50)

    print(f"Image: {image_path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Class index: {result['class_index']}")
    print(f"Fusion score: {result['fusion_score']:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    main()
