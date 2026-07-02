# Medical Image Classification: AneRBC Anemia Detection

An end-to-end deep learning pipeline for binary classification of red blood cells (**Anemic** vs. **Healthy**) from macroscopic blood-smear images in the AneRBC dataset. The project covers custom CNN architectures, transfer learning with pretrained backbones, and Explainable AI (XAI) via Integrated Gradients.

The code is organized into a modular, industry-standard structure, driven entirely by a master orchestrator script (`main.py`).

---

## Repository Structure

```text
AneRBC-Anemia-Classifier/
├── src/
│   ├── data/
│   │   └── data_loader.py          # Dataset parsing, subsets, DataLoaders
│   ├── models/
│   │   ├── custom_models.py        # Custom 3, 4, and 5-Layer CNN architectures
│   │   └── pretrained_models.py    # MobileNetV2, SqueezeNet, ResNet18 transfer learning
│   ├── training/
│   │   └── trainer.py              # Training/validation loops and model checkpointing
│   └── visualization/
│       ├── evaluation.py           # Metrics, confusion matrices, and learning curves
│       └── xai.py                  # Captum Integrated Gradients for XAI
├── data/                           # Ignored by Git: Raw AneRBC dataset goes here
├── saved_models/                   # Ignored by Git: Saved .pth weights and .pkl history
├── .gitignore
├── requirements.txt
├── setup_data.py                   # Automated data downloader and extractor
├── main.py                         # Master orchestrator script (toggles for pipeline steps)
└── README.md

```

# 1. Environment Setup
Developed using PyTorch in a Python 3.10+ environment (optimized for CUDA/T4 GPUs). 

To reproduce locally:

Bash
## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

## Install all required dependencies
pip install -r requirements.txt

# 2. Dataset Preparation

Dataset: AneRBC - a benchmark dataset for computer-aided anemia diagnosis using RBC images.

Instead of manually placing files, use the automated setup utility to fetch and structure the data (targeting the Original_images modality):

Bash
python setup_data.py

**Option 1:** Provide a direct URL to download and automatically extract the dataset.

**Option 2:** Provide the path to a locally downloaded .zip file for automated extraction.

Images are resized to 224x224 and normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). The data loader performs a deterministic (seed=42) 70/15/15 stratified train/validation/test split. A balanced 4,000-image prototyping subset is used by default for rapid testing.

# 3. Running the Pipeline (Training, Evaluation & XAI)

This project utilizes a centralized orchestrator (main.py). You do not need to run individual scripts. Open main.py and toggle the boolean flags at the top of the file to True or False to execute specific parts of the pipeline:

Python
## --- PIPELINE CONTROLS inside main.py ---
RUN_DATA_PIPELINE = True

TRAIN_CUSTOM_MODEL = True

TRAIN_PRETRAINED_MODEL = True

RUN_EVALUATION = True

RUN_XAI = True

Once your flags are set, execute the pipeline:

Bash

**python main.py**

**Training:** Runs the selected models (Custom CNNs and/or Pretrained models like MobileNetV2), saving <Name>_weights.pth and <Name>_history.pkl into the saved_models/ directory.

**Evaluation:** Evaluates the saved models on the unseen test set, printing a Classification Report (Precision, Recall, F1) and plotting Confusion Matrices.

**Explainable AI (XAI):** Generates Integrated Gradients attribution maps, comparing how the Custom and Pretrained models analyze the biological morphology of the same test image.

# 4. Results Summary

| Rank | Model | Type | Test Acc. | Macro F1 | Trainable Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | MobileNetV2 | Pretrained | 86% | 0.86 | 2,562 |
| 2 | Custom 4-Layer CNN | Custom | 85% | 0.85 | ≈ 26.1 M |
| 3 | ResNet18 | Pretrained | 84% | 0.84 | 1,026 |
| 4 | Custom 5-Layer CNN | Custom | 81% | 0.81 | ≈ 14.4 M |
| 5 | SqueezeNet1.1 | Pretrained | 80% | 0.80 | 1,026 |
| 6 | Custom 3-Layer CNN | Custom | 50% | 0.33 | ≈ 51.5 M |


Full analysis, learning curves, and the detailed XAI critique can be found in the accompanying Critical Evaluation Report.

