# Computer Vision Assignment 3: Foundation Models for One-Shot Image Classification

## 1. Project Overview

The objective of this assignment is to perform one-shot image classification across 16 image categories using exactly one labeled reference image per class.

I conducted five experiments using different pretrained models, feature extraction techniques, ensemble methods, image augmentation strategies, and reference prototype refinement techniques to investigate their effectiveness in one-shot image classification.

Among the five experiments, the combination of ResNet-50 pretrained on Places365 and SigLIP 2 Base pretrained on WebLI achieved the highest observed test accuracy of 85.25%. This configuration was selected as the final classifier.

The repository contains the complete experimental notebook, original test dataset, exported model artifacts, standalone inference and evaluation scripts, required dependencies, and classification results.

## 2. Dataset

The original training and testing datasets provided as part of Computer Vision Assignment 3 were used for this project.

The dataset consists of:

- Total number of classes: 16
- Original training images: 2,400
- Original training images per class: 150
- One-shot reference images: 16
- Reference images per class: 1
- Remaining training images used for diagnostic validation: 2,384
- Test images: 400
- Test images per class: 25

The 16 image categories are Bedroom, Coast, Flower, Forest, Highway, Industrial, InsideCity, Kitchen, LivingRoom, Mountain, Office, OpenCountry, Store, Street, Suburb, and TallBuilding.

Exactly one labeled reference image from each class was used to construct the corresponding class prototypes.

All pretrained feature extractors were kept frozen throughout the experiments, and no additional fine-tuning was performed. The remaining training images were used for diagnostic validation rather than training the model parameters or constructing additional reference prototypes.

The dataset split information and class ordering are preserved in `dataset_split_manifest.json`.

The original test dataset is included in the root directory of this repository as `test.zip`.

The original training dataset (`train.zip`) is not included in the repository due to its large file size. It can be obtained from the original Computer Vision Assignment 3 materials.

The training dataset is required for reproducing the original experiments but is not required for evaluating the exported final classifier.

## 3. Experiments

### Experiment 1: ConvNeXt and CLIP

In the first experiment, I used ConvNeXt-Base pretrained on ImageNet-22K and CLIP ViT-L/14 QuickGELU as feature extractors.

The extracted image embeddings were L2-normalized, and cosine similarity was calculated between the reference and evaluation image embeddings.

Per-image z-score normalization was applied before combining the similarity scores obtained from both models using equal fusion weights.

Results:

- Validation accuracy: 80.12%
- Test accuracy: 81.25%

### Experiment 2: ResNet-50 Places365 and SigLIP 2 Base

In the second experiment, I used ResNet-50 pretrained on Places365 and SigLIP 2 Base pretrained on WebLI.

ResNet-50 Places365 was selected because it provides scene-oriented visual representations. SigLIP 2 Base was introduced to obtain complementary visual features learned through large-scale vision-language pretraining.

The extracted embeddings were L2-normalized, and cosine similarity was calculated between the evaluation image embeddings and reference class prototypes.

The similarity scores obtained from both models were independently normalized using per-image z-score normalization and combined using equal fusion weights of 0.50.

Results:

- Validation accuracy: 84.10%
- Test accuracy: 85.25%

The individual model test accuracies were:

- ResNet-50 Places365: 83.50%
- SigLIP 2 Base: 78.75%
- Combined classifier: 85.25%

The combined classifier performed better than either individual model.

This experiment achieved the highest observed test accuracy among all five experiments and was selected as the final classifier.

### Experiment 3: ResNet-50 Places365, SigLIP 2 Base, and DINOv2

In the third experiment, I introduced DINOv2 ViT-B/14 as an additional feature extractor along with ResNet-50 Places365 and SigLIP 2 Base.

The objective was to investigate whether combining three different pretrained feature representations could improve classification performance.

The cosine similarity scores obtained from all three models were independently normalized using per-image z-score normalization and combined using equal fusion weights.

Results:

- Validation accuracy: 82.63%
- Test accuracy: 83.50%

The addition of DINOv2 did not improve classification accuracy compared to Experiment 2.

### Experiment 4: ResNet-50 Places365, SigLIP 2 Base, and MoCo v2

In the fourth experiment, I combined ResNet-50 Places365, SigLIP 2 Base, and MoCo v2 pretrained ResNet-50 using the official 200-epoch pretrained checkpoint.

Two approaches were evaluated: original-view classification and multi-view classification.

The multi-view approach used the following image transformations and weights:

- Original image: 0.40
- Horizontal flip: 0.20
- Center zoom: 0.20
- Left crop: 0.10
- Right crop: 0.10

The embeddings obtained from the individual views were L2-normalized, combined using weighted averaging, and normalized again before calculating cosine similarity.

The multi-view approach was applied to both reference and evaluation images.

The similarity scores obtained from the three pretrained models were independently normalized using per-image z-score normalization and combined using equal fusion weights.

Original-view results:

- Validation accuracy: 83.10%
- Test accuracy: 84.50%

Multi-view results:

- Validation accuracy: 82.30%
- Test accuracy: 82.75%

The results showed that multi-view augmentation did not improve classification performance for this particular configuration.

### Experiment 5: Reference Prototype Refinement

In the fifth experiment, I used the same pretrained models as Experiment 2 and investigated whether augmenting only the reference images could improve the quality of the class prototypes.

The following reference image transformations and weights were used:

- Original image: 0.70
- Horizontal flip: 0.10
- Mild center crop: 0.10
- Cutout: 0.10

The mild center crop retained approximately 90% of the original image dimensions. The Cutout transformation masked approximately 8% of the image area using the ImageNet mean color.

The embeddings obtained from the original and transformed reference images were L2-normalized, combined using weighted averaging, and normalized again to construct the refined class prototypes.

The evaluation images were processed without augmentation, and the equal-weight score fusion strategy remained unchanged.

Original prototype results:

- Validation accuracy: 84.10%
- Test accuracy: 85.25%

Refined prototype results:

- Validation accuracy: 84.19%
- Test accuracy: 85.00%

Although reference prototype refinement achieved a slightly higher diagnostic validation accuracy, its test accuracy was lower than that of Experiment 2.

## 4. Final Model Architecture

Experiment 2 was selected as the final classifier because it achieved the highest observed test accuracy of 85.25%.

The final classifier consists of two pretrained feature extractors.

### ResNet-50 Places365

ResNet-50 pretrained on Places365 provides scene-oriented visual representations that are useful for scene classification.

The original classification layer is removed, and the model is used as a frozen feature extractor.

The preprocessing procedure consists of:

- Resizing the image to 256 x 256.
- Applying a center crop of size 224 x 224.
- Converting the image to a tensor.
- Applying ImageNet normalization.

The normalization parameters are:

- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

### SigLIP 2 Base

SigLIP 2 Base provides complementary visual representations learned through large-scale vision-language pretraining.

The model is loaded through OpenCLIP using the following configuration:

- Model architecture: ViT-B-16-SigLIP2
- Pretrained weights: webli

The corresponding pretrained evaluation transformations provided by OpenCLIP are used for image preprocessing.

### Classification Procedure

1. Load the input image.
2. Apply the corresponding preprocessing transformations for each pretrained model.
3. Extract image embeddings using ResNet-50 Places365 and SigLIP 2 Base.
4. Apply L2 normalization to the extracted embeddings.
5. Calculate cosine similarity between the input image embeddings and the saved reference class prototypes.
6. Apply per-image z-score normalization to the similarity scores obtained from each model.
7. Combine the normalized similarity scores using equal fusion weights.
8. Select the class with the highest combined similarity score.

The final fusion score is calculated as:

```text
Final Score = 0.5 * Normalized ResNet Score
            + 0.5 * Normalized SigLIP Score
```

Per-image z-score normalization is performed across the 16 class similarity scores independently for each model. This helps reduce differences in the similarity score distributions before combining the scores.

The reference class prototypes and final model configuration have been exported and included in this repository.

Therefore, the final classifier can be reconstructed without repeating the original experiments or performing additional training.

## 5. Repository Files

The following files are included in this repository:

- `ComputerVision_Assignment3_Final_GoogleColab.ipynb`: Contains the complete implementation of all five experiments, including dataset preparation, one-shot reference selection, feature extraction, classification, experimental comparisons, and final model export.

- `test.zip`: Contains the original test dataset provided for Computer Vision Assignment 3.

- `dataset_split_manifest.json`: Contains the dataset split information and class ordering used in the experiments.

- `experiment2_prototypes.pt`: Contains the saved one-shot reference class prototypes obtained from ResNet-50 Places365 and SigLIP 2 Base.

- `model_config.json`: Contains the final model architecture, pretrained model information, preprocessing parameters, feature normalization method, similarity calculation method, class ordering, and fusion weights.

- `inference.py`: Reconstructs the final classifier, automatically downloads the required pretrained weights, loads the saved reference prototypes, and performs image classification.

- `evaluate.py`: Extracts the test dataset, evaluates the final classifier, calculates the overall and per-class classification accuracy, and saves the evaluation results.

- `requirements.txt`: Contains the required Python dependencies for executing the standalone implementation.

- `evaluation_summary.json`: Contains the overall classification results and execution information obtained during the local evaluation.

- `per_class_accuracy.csv`: Contains the classification results for each of the 16 image categories.

- `test_predictions.csv`: Contains the individual test image predictions and corresponding evaluation information.

- `README.md`: Contains the project description, experimental observations, and instructions for recreating and evaluating the final classifier.

The original test dataset, reference class prototypes, and model configuration are included in the repository.

The original training dataset is not included due to its large file size. However, it is not required for evaluating the exported final classifier.

The large pretrained checkpoint files are not included in the repository because they are downloaded automatically during execution.

## 6. System Requirements

The following requirements are recommended for executing the standalone implementation:

- Python 3.10
- Windows, Linux, or macOS with compatible Python dependencies
- CPU or CUDA-compatible GPU
- Internet connection for dependency installation and initial pretrained model downloads
- Sufficient disk space for Python libraries, pretrained model weights, datasets, and generated evaluation results

Python 3.10.11 was used during the successful local evaluation on Windows.

A dedicated GPU is not mandatory. The final classifier was successfully evaluated using both a Google Colab GPU environment and a local Windows CPU environment.

The pretrained model weights require approximately 1.6 GB of storage. Additional storage is required for Python dependencies, model caching, and the datasets.

The required Python libraries are:

- torch
- torchvision
- open_clip_torch
- timm
- numpy
- pandas
- Pillow
- tqdm

All these dependencies are listed in `requirements.txt`.

## 7. Steps to Recreate and Evaluate the Final Model

The following steps explain how to download the repository, configure the Python environment, prepare the dataset directory, reconstruct the final classifier, and reproduce the reported test accuracy.

The original test dataset, final model configuration, and reference class prototypes are already included in the repository.

There is no need to obtain the test dataset from an external source, repeat the original experiments, or perform additional model training to evaluate the final classifier.

The original training dataset is not required for this evaluation.

### Step 1: Download the GitHub Repository

The GitHub repository is available at:

https://github.com/adaparishitha-creator/Foundation_Model

Open the repository, click the Code button, and select Download ZIP.

Extract the downloaded repository ZIP file to a preferred location on the system.

Alternatively, if Git is installed, the repository can be cloned using the following commands:

```bash
git clone https://github.com/adaparishitha-creator/Foundation_Model.git
cd Foundation_Model
```

Open the downloaded repository folder in Visual Studio Code or another preferred development environment.

Open a terminal in the directory containing `inference.py`, `evaluate.py`, and `requirements.txt`.

Make sure the terminal is opened in the actual repository directory containing these files rather than its parent directory.

After downloading the repository, the following files should be available:

```text
Foundation_Model/

    ComputerVision_Assignment3_Final_GoogleColab.ipynb
    dataset_split_manifest.json
    evaluate.py
    evaluation_summary.json
    experiment2_prototypes.pt
    inference.py
    model_config.json
    per_class_accuracy.csv
    requirements.txt
    test_predictions.csv
    test.zip
    README.md
```

The downloaded repository folder may be named `Foundation_Model-main` when downloaded using the Download ZIP option.

### Step 2: Create and Activate a Python Virtual Environment

Make sure Python 3.10 is installed on the system.

Creating a separate Python virtual environment is recommended to avoid dependency conflicts.

For Windows PowerShell, execute:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For Linux or macOS, execute:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Verify the Python version using the following command:

```bash
python --version
```

The Python version used during my successful local evaluation was Python 3.10.11.

### Step 3: Install the Required Python Libraries

Upgrade pip using the following command:

```bash
python -m pip install --upgrade pip
```

For a CPU-only Windows environment, install PyTorch and TorchVision using the official CPU wheel index:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining required dependencies:

```bash
python -m pip install -r requirements.txt
```

This installation procedure was used during my successful local CPU evaluation.

For other supported environments, the dependencies can be installed directly using:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, a compatible CUDA-enabled PyTorch installation is required. The appropriate installation instructions are available at:

https://pytorch.org/get-started/locally/

The implementation automatically detects whether CUDA is available and selects the appropriate execution device.

### Step 4: Prepare the Dataset Directory

The original test dataset (`test.zip`) is already included in the root directory of the downloaded GitHub repository.

Create a new folder named `data` inside the downloaded repository.

The folder can be created manually or by executing the following command from the repository directory:

```bash
mkdir data
```

Move the existing `test.zip` file from the root directory into the newly created `data` folder.

This can be done manually using the file explorer or by executing the following command.

For Windows PowerShell:

```powershell
Move-Item test.zip data/
```

For Linux or macOS:

```bash
mv test.zip data/
```

If the file has already been moved manually, the corresponding command does not need to be executed.

The test dataset should now be available at the following location:

```text
data/test.zip
```

The evaluation script specifically expects the test dataset at `data/test.zip`.

There is no need to download the test dataset separately because it is already included in the repository.

The original internal directory structure of the ZIP file should remain unchanged.

There is no need to manually extract the ZIP file because the `evaluate.py` script automatically extracts the test dataset during execution.

The original training dataset (`train.zip`) is not included in the repository due to its large file size. It is not required for evaluating the exported final classifier because the reference class prototypes are already included in the repository.

After completing this step, the repository should have the following structure:

```text
Foundation_Model/

    data/
        test.zip

    ComputerVision_Assignment3_Final_GoogleColab.ipynb
    dataset_split_manifest.json
    evaluate.py
    evaluation_summary.json
    experiment2_prototypes.pt
    inference.py
    model_config.json
    per_class_accuracy.csv
    requirements.txt
    test_predictions.csv
    README.md
```

The `.venv` directory may also be present if the Python virtual environment was created inside the repository.

The existing model configuration and reference prototype files should remain in their original locations.

There is no need to manually create the `artifacts`, `checkpoints`, or `outputs` folders.

The implementation automatically creates required checkpoint and output directories during execution.

Once the `test.zip` file has been placed inside the `data` folder, the dataset preparation is complete.

### Step 5: Automatic Pretrained Checkpoint Downloads

The final classifier requires pretrained weights for the following two models:

- ResNet-50 pretrained on Places365
- SigLIP 2 Base pretrained on WebLI

The pretrained checkpoint files are not included in the GitHub repository because of their large file sizes.

The functionality required to download and load the pretrained weights is implemented directly in `inference.py`.

During the first execution, the implementation automatically creates the `checkpoints` folder and downloads the required pretrained weights if they are not already available locally.

The ResNet-50 Places365 checkpoint is downloaded from:

http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar

SigLIP 2 Base is loaded through OpenCLIP using the following configuration:

```text
Model architecture: ViT-B-16-SigLIP2
Pretrained weights: webli
```

The corresponding SigLIP 2 pretrained weights are retrieved through OpenCLIP and Hugging Face.

The ResNet-50 Places365 checkpoint is stored in the project checkpoint directory, while the SigLIP 2 pretrained weights are managed through the corresponding model cache.

The downloaded weights are reused during subsequent executions whenever available.

There is no need to:

- Manually create the checkpoints folder.
- Manually download the pretrained model weights.
- Copy checkpoint files from another system.
- Modify the pretrained model paths in the Python scripts.

The initial execution requires active internet connection if the pretrained weights are not already available.

The first execution may take longer because of the pretrained checkpoint downloads. The download time depends on the internet connection speed and availability of the pretrained model sources.

A Hugging Face authentication token is normally not required for downloading these public pretrained weights.

If a Windows symlink warning or Hugging Face authentication warning appears during downloading, the evaluation can continue as long as the required pretrained weights are successfully downloaded and loaded.

### Step 6: Execute Final Model Evaluation

After installing the required Python libraries and placing the `test.zip` file inside the `data` folder, execute the following command from the repository directory:

```bash
python evaluate.py --batch-size 8
```

The implementation automatically performs the following operations:

1. Locates the original test dataset ZIP file.
2. Extracts the test dataset if it has not already been extracted.
3. Loads the dataset split information and class ordering.
4. Detects whether CPU or CUDA execution is available.
5. Creates the checkpoints folder when required.
6. Downloads the missing pretrained model weights.
7. Reconstructs the ResNet-50 Places365 and SigLIP 2 Base feature extractors.
8. Loads the saved reference class prototypes.
9. Applies the corresponding preprocessing transformations.
10. Extracts and normalizes the test image embeddings.
11. Calculates cosine similarity between the test image embeddings and reference class prototypes.
12. Applies per-image z-score normalization.
13. Combines the normalized similarity scores using equal fusion weights.
14. Predicts the corresponding class for each of the 400 test images.
15. Calculates the overall and per-class classification accuracy.
16. Creates the outputs folder and saves the evaluation results.

The expected evaluation results, based on the verified executions, are:

```text
Model: ResNet-50 Places365 + SigLIP 2 Base

Total test images: 400
Correct predictions: 341
Incorrect predictions: 59
Test accuracy: 85.25%
```

The total execution time may vary depending on the hardware configuration, internet connection, and availability of previously downloaded pretrained checkpoints.

Minor numerical differences may occur across hardware platforms and library versions.

If the system has limited available memory, a smaller batch size can be used:

```bash
python evaluate.py --batch-size 4
```

Changing the batch size does not change the classification methodology. However, batch size 8 was used for the verified local evaluation results reported in this repository. Batch size 4 is provided as an optional alternative for systems with limited memory and was not separately evaluated.

### Step 7: Verify the Generated Evaluation Results

After successful execution, the implementation automatically creates a folder with the name `outputs` inside the repository.

The following files are generated inside this folder:

```text
outputs/

    evaluation_summary.json
    per_class_accuracy.csv
    test_predictions.csv
```

The `evaluation_summary.json` file contains the overall classification accuracy, number of correct and incorrect predictions, and execution information.

The `per_class_accuracy.csv` file contains the classification results for each of the 16 image categories.

The `test_predictions.csv` file contains individual test image predictions and corresponding evaluation information.

The evaluation result files already included in the root directory of the repository are from the completed local evaluation.

The newly generated files inside the `outputs` folder contain the results of the current execution.

There is no need to manually create outputs folder.

After successful execution, the original test dataset will also be extracted inside the `data` folder, and the required pretrained model weights will be stored locally.

### Step 8: Summary of Manual and Automatic Setup

The following operations need to be performed manually:

1. Download and extract or clone the GitHub repository.
2. Create and activate the Python virtual environment.
3. Install the required Python dependencies.
4. Create a folder named `data` inside the downloaded repository.
5. Move the existing `test.zip` file into the `data` folder.
6. Execute `python evaluate.py --batch-size 8`.

The following operations are handled automatically by the implementation:

- Test dataset extraction
- Checkpoints folder creation
- Pretrained model downloading
- Pretrained model loading
- CPU or GPU detection
- Final classifier reconstruction
- Saved reference prototype loading
- Image preprocessing
- Feature extraction
- Test image classification
- Overall accuracy calculation
- Per-class accuracy calculation
- Outputs folder creation
- Evaluation result generation

The original test dataset is already included in the repository. The original training dataset is not required for evaluating the exported final classifier.

There is no need to obtain the test dataset separately, manually extract the test dataset, create the checkpoints folder, manually download pretrained weights, create an artifacts folder, repeat the five experiments, or perform additional model training.

## 8. Single-Image Classification

The exported final classifier can also be used to classify an individual image without evaluating the complete test dataset.

Execute the following command from the repository directory:

```bash
python inference.py --image "path/to/image.jpg"
```

Replace `path/to/image.jpg` with the actual location of the image that needs to be classified.

The image can be provided using either an absolute or relative file path.

For example, after extracting the test dataset through the evaluation script, the path of an existing image from one of the test class folders can be provided as input.

The program displays the classification prediction for the provided image.

The complete test dataset is not required for single-image classification.

The saved reference class prototypes and model configuration included in the repository are used to reconstruct the final classifier.

If the required pretrained weights are not already available, they are downloaded automatically.

The classifier predicts one of the 16 image categories used in this assignment.

## 9. Reproducing All Five Experiments in Google Colab

The complete experimental implementation is available in:

`ComputerVision_Assignment3_Final_GoogleColab.ipynb`

The following steps can be followed to reproduce the original experiments.

### Step 1: Open the Notebook

Download `ComputerVision_Assignment3_Final_GoogleColab.ipynb` from the repository and open it in Google Colab.

Google Colab is available at:

https://colab.research.google.com/

### Step 2: Enable GPU Execution

In Google Colab, select Runtime and then Change runtime type.

Select a GPU hardware accelerator, such as a T4 GPU.

GPU execution is recommended for reproducing the complete experiments because multiple pretrained models and image augmentation strategies are evaluated.

### Step 3: Prepare the Original Datasets

Both the original training and testing datasets are required for reproducing all five experiments.

The required dataset files are:

- `train.zip`
- `test.zip`

The original test dataset (`test.zip`) is included in the root directory of the GitHub repository.

The original training dataset (`train.zip`) is not included in the repository due to its large file size. It can be obtained from the original Computer Vision Assignment 3 materials.

Download the test dataset from the repository and obtain the original training dataset from the assignment materials.

If the original dataset files are already available locally, the existing ZIP files can be used.

Upload both ZIP files to the Google Colab runtime or make them available through the storage location configured in the notebook.

Configure their locations according to the dataset path definitions in the notebook.

The original internal directory structure of both ZIP archives should remain unchanged.

Dataset extraction is handled by the notebook.

The training dataset is required for reproducing the original experiments but is not required for evaluating the exported final classifier using the standalone evaluation script.

### Step 4: Execute the Notebook

Execute the notebook cells sequentially, starting with the environment setup and library imports.

The notebook contains the implementation for:

- GPU verification and library imports
- Dataset preparation and extraction
- Dataset and class verification
- One-shot reference image selection
- Diagnostic validation split creation
- Pretrained model loading
- Image preprocessing and feature extraction
- Reference class prototype construction
- Cosine similarity-based classification
- Similarity score normalization and fusion
- Execution of all five experiments
- Experimental result comparisons
- Final model artifact export

The original dataset files and reference image selection procedure should be preserved when reproducing the experiments.

The provided `dataset_split_manifest.json` file can be used to verify the reference image selection and class ordering.

An active internet connection is required for installing the necessary libraries and downloading any missing pretrained model weights.

The execution time may vary depending on the GPU configuration, available memory, internet connection, and model cache availability.

### Step 5: Review the Experimental Results

After executing the notebook, review the validation and test accuracy obtained from each experiment.

Experiment 2 achieved the highest observed test accuracy of 85.25%.

The notebook also contains the procedure used to export the final Experiment 2 reference class prototypes and model configuration.

Executing the complete notebook is necessary when reproducing all five original experiments.

For evaluating the already exported final classifier, the standalone evaluation procedure described in Section 7 is sufficient.

## 10. Reproducibility Verification

After completing all five experiments in Google Colab, I exported the reference class prototypes and model configuration for Experiment 2.

I then implemented standalone inference and evaluation scripts and evaluated the final classifier locally using Visual Studio Code on Windows.

During the initial local execution, the pretrained ResNet-50 Places365 and SigLIP 2 Base weights were successfully downloaded automatically.

The original Google Colab test accuracy of 85.25% was reproduced successfully.

After uploading the required implementation and model files to GitHub, I downloaded the repository again and evaluated the implementation in a separate directory.

For this verification, I reused the existing Python environment, original test dataset, and previously downloaded pretrained checkpoints to avoid downloading the same model weights again.

The downloaded GitHub repository successfully reproduced the same test accuracy without retraining or manually rearranging the exported model files.

The results obtained from the three execution configurations are provided below.

### Google Colab GPU Execution

- Execution device: GPU
- Total test images: 400
- Correct predictions: 341
- Incorrect predictions: 59
- Test accuracy: 85.25%

### Original Local VS Code CPU Execution

- Operating system: Windows
- Python version: 3.10.11
- Execution device: CPU
- Batch size: 8
- Total test images: 400
- Correct predictions: 341
- Incorrect predictions: 59
- Test accuracy: 85.25%
- Inference time: 62.04 seconds
- Total runtime: 127.72 seconds

### Downloaded GitHub Repository CPU Execution

- Operating system: Windows
- Python version: 3.10.11
- Execution device: CPU
- Batch size: 8
- Total test images: 400
- Correct predictions: 341
- Incorrect predictions: 59
- Test accuracy: 85.25%
- Inference time: 108.71 seconds
- Total runtime: 114.84 seconds

The same overall classification accuracy was obtained in all three execution configurations.

The reported execution times are specific to the corresponding execution environments and should not be interpreted as a direct hardware performance comparison.

Automatic checkpoint downloading was verified during the original local evaluation, while compatibility with the downloaded GitHub repository structure was verified separately using the existing Python environment and cached pretrained weights.

The original test dataset was subsequently included in the repository to simplify dataset preparation for reproduction.

The original training dataset was not included due to its large file size and is not required for evaluating the exported final classifier.

## 11. Per-Class Classification Results

The final Experiment 2 classifier achieved following classification results:

- Bedroom: 20 out of 25 images, 80.00%
- Coast: 25 out of 25 images, 100.00%
- Flower: 25 out of 25 images, 100.00%
- Forest: 25 out of 25 images, 100.00%
- Highway: 25 out of 25 images, 100.00%
- Industrial: 18 out of 25 images, 72.00%
- InsideCity: 11 out of 25 images, 44.00%
- Kitchen: 25 out of 25 images, 100.00%
- LivingRoom: 24 out of 25 images, 96.00%
- Mountain: 22 out of 25 images, 88.00%
- Office: 25 out of 25 images, 100.00%
- OpenCountry: 1 out of 25 images, 4.00%
- Store: 24 out of 25 images, 96.00%
- Street: 24 out of 25 images, 96.00%
- Suburb: 25 out of 25 images, 100.00%
- TallBuilding: 22 out of 25 images, 88.00%

Overall results:

- Total test images: 400
- Correct predictions: 341
- Incorrect predictions: 59
- Overall test accuracy: 85.25%

Seven classes achieved 100% classification accuracy, and 13 out of the 16 classes achieved an accuracy of at least 80%.

OpenCountry and InsideCity were the most challenging categories for the final classifier. These two categories accounted for 38 of the 59 incorrect predictions.

The detailed classification results and individual image predictions are available in the JSON and CSV files included in the repository.

## 12. Experimental Observations and Limitations

Through these experiments, I observed that combining complementary pretrained feature representations can improve one-shot image classification performance.

ResNet-50 pretrained on Places365 provided useful scene-oriented features, while SigLIP 2 Base provided complementary visual representations.

Individually, ResNet-50 Places365 achieved a test accuracy of 83.50%, and SigLIP 2 Base achieved a test accuracy of 78.75%. Combining both models increased the test accuracy to 85.25%.

The combination of ResNet-50 Places365 and SigLIP 2 Base also performed better than the combination of ConvNeXt and CLIP evaluated in Experiment 1.

Adding DINOv2 as an additional feature extractor did not improve classification accuracy. Similarly, the multi-view augmentation approach evaluated in Experiment 4 did not improve classification performance compared to its original-view configuration.

Reference prototype refinement in Experiment 5 slightly improved diagnostic validation accuracy but did not improve test accuracy.

These observations indicate that increasing the number of pretrained models or applying additional image transformations does not necessarily improve classification performance.

One limitation of the one-shot classification approach is that each class is represented by only one labeled reference image. If the selected image is not sufficiently representative of all the visual variations within a class, classification performance may decrease.

This limitation may be particularly relevant to challenging categories such as OpenCountry and InsideCity.

Experiment 2 was selected based on the observed test accuracy across the completed experiments. Therefore, the reported test accuracy represents the best observed experimental performance rather than an independent post-selection estimate of generalization performance.

Another limitation is that the standalone implementation downloads pretrained weights from external sources. Successful first-time execution depends on the availability of these sources and a compatible Python environment.

The standalone implementation was verified using Python 3.10.11 on Windows. Differences in hardware platforms and library versions may result in minor numerical variations.

Additionally, the original training dataset is not included in the repository due to its large file size. Reproducing all five experiments requires obtaining the original training dataset from the assignment materials.

However, this does not affect the evaluation of the exported final classifier because the saved reference class prototypes and original test dataset are already included in the repository.

## 13. Conclusion

The objective of this assignment was to investigate the effectiveness of pretrained models and foundation model representations for one-shot image classification using exactly one labeled reference image per class.

Among the five experiments, the combination of ResNet-50 pretrained on Places365 and SigLIP 2 Base pretrained on WebLI achieved the highest observed test accuracy of 85.25%.

The final classifier correctly classified 341 out of 400 test images without additional fine-tuning of the pretrained feature extractors.

The experimental results showed that combining complementary pretrained visual representations can provide effective classification performance even when only one labeled reference image per class is available.

The reported test accuracy was successfully reproduced in Google Colab, the original local VS Code implementation, and the downloaded GitHub repository.

The final classifier can be reconstructed using the exported reference class prototypes, model configuration, and automatically downloaded pretrained model weights.

The original test dataset included in the repository is sufficient to evaluate the final classifier without requiring the original training dataset or performing additional model training.

## 14. References

- Places365: https://github.com/CSAILVision/places365
- SigLIP 2: https://arxiv.org/abs/2502.14786
- OpenCLIP: https://github.com/mlfoundations/open_clip
- ConvNeXt: https://github.com/facebookresearch/ConvNeXt
- CLIP: https://github.com/openai/CLIP
- DINOv2: https://github.com/facebookresearch/dinov2
- MoCo: https://github.com/facebookresearch/moco
- PyTorch: https://pytorch.org/
- TorchVision: https://pytorch.org/vision/
- timm: https://github.com/huggingface/pytorch-image-models
