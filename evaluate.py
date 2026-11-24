import argparse
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from inference import OneShotClassifier


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TEST_ZIP = DATA_DIR / "test.zip"
TEST_DIR = DATA_DIR / "test"

MANIFEST_PATH = ROOT / "dataset_split_manifest.json"


def prepare_test_dataset():

    if not TEST_ZIP.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_ZIP}"
        )

    if not TEST_DIR.exists():

        print("Extracting test dataset...")

        with zipfile.ZipFile(TEST_ZIP, "r") as archive:

            # Verify extraction paths
            for member in archive.infolist():

                target = (
                    DATA_DIR / member.filename
                ).resolve()

                if not target.is_relative_to(
                    DATA_DIR.resolve()
                ):
                    raise ValueError(
                        "Unsafe ZIP extraction path."
                    )

            archive.extractall(DATA_DIR)

        print("Test dataset extracted successfully.")

    else:

        print("Using existing extracted test dataset.")

    return TEST_DIR


def load_test_samples(test_dir, class_names):

    samples = []

    valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    for class_index, class_name in enumerate(class_names):

        class_dir = test_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        image_paths = sorted([
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in valid_extensions
        ])

        for image_path in image_paths:

            samples.append(
                (image_path, class_index)
            )

    return samples


def evaluate(batch_size=8):

    start_time = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset split information
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(file)

    class_names = manifest["class_names"]
    expected_count = manifest["test_count"]

    # Prepare test dataset
    test_dir = prepare_test_dataset()

    test_samples = load_test_samples(
        test_dir,
        class_names
    )

    if len(test_samples) != expected_count:

        raise ValueError(
            f"Expected {expected_count} test images, "
            f"but found {len(test_samples)}."
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 - TEST SET EVALUATION")
    print("=" * 60)

    print(f"Number of classes: {len(class_names)}")
    print(f"Number of test images: {len(test_samples)}")
    print(f"Batch size: {batch_size}")

    # Load the exported one-shot classifier
    classifier = OneShotClassifier()

    if classifier.class_names != class_names:

        raise ValueError(
            "Classifier and dataset class orders do not match."
        )

    correct = 0
    total = 0

    results = []

    inference_start = time.perf_counter()

    # Evaluate all test images
    for start in tqdm(
        range(0, len(test_samples), batch_size),
        desc="Evaluating test images",
        unit="batch"
    ):

        batch = test_samples[
            start:start + batch_size
        ]

        images = []

        for image_path, _ in batch:

            with Image.open(image_path) as image:

                images.append(
                    image.convert("RGB")
                )

        predictions, scores = classifier.predict_batch(
            images
        )

        for index, (image_path, true_label) in enumerate(batch):

            predicted_label = predictions[index].item()

            is_correct = (
                predicted_label == true_label
            )

            correct += int(is_correct)
            total += 1

            results.append({
                "image": image_path.relative_to(
                    DATA_DIR
                ).as_posix(),

                "actual_class": class_names[
                    true_label
                ],

                "predicted_class": class_names[
                    predicted_label
                ],

                "correct": is_correct,

                "fusion_score": scores[
                    index,
                    predicted_label
                ].item()
            })

    inference_time = (
        time.perf_counter() - inference_start
    )

    # Calculate final accuracy
    accuracy = 100.0 * correct / total

    results_df = pd.DataFrame(results)

    # Save individual image predictions
    predictions_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )

    results_df.to_csv(
        predictions_path,
        index=False
    )

    # Calculate per-class accuracy
    class_results = (
        results_df
        .groupby("actual_class")
        .agg(
            total_images=("correct", "size"),
            correct_predictions=("correct", "sum")
        )
        .reset_index()
    )

    class_results["accuracy_percent"] = (
        100.0
        * class_results["correct_predictions"]
        / class_results["total_images"]
    )

    class_results.to_csv(
        OUTPUT_DIR / "per_class_accuracy.csv",
        index=False
    )

    total_runtime = (
        time.perf_counter() - start_time
    )

    # Save overall evaluation results
    summary = {
        "experiment": "Experiment 2",
        "architecture": (
            "ResNet-50 Places365 + SigLIP 2 Base"
        ),
        "device": str(classifier.device),
        "number_of_classes": len(class_names),
        "total_test_images": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "test_accuracy_percent": accuracy,
        "inference_time_seconds": inference_time,
        "total_runtime_seconds": total_runtime
    }

    with open(
        OUTPUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # Display final results
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 60)

    print("Model: ResNet-50 Places365 + SigLIP 2 Base")
    print(f"Device: {classifier.device}")
    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Total runtime: {total_runtime:.2f} seconds")

    print("\nPER-CLASS ACCURACY")
    print(
        class_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}"
        )
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate the Experiment 2 one-shot classifier."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images processed per batch."
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be positive.")

    evaluate(
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()