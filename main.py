# Name: Muhammad Faheem
# SRN: [Insert Your SRN Here]
# Module: Main Orchestrator

import os
import torch

# Import our custom modules from the src/ directory
from src.data.data_loader import map_and_validate_dataset, create_stratified_subset, prepare_dataloaders
from src.models.custom_models import CNN4Layer
from src.models.pretrained_models import setup_pretrained_model
from src.training.trainer import train_and_validate, save_model_and_history
from src.visualization.evaluation import evaluate_on_test_set
from src.visualization.xai import apply_integrated_gradients

# --- PIPELINE CONTROLS ---
# Toggle these to True/False to run specific parts of the project
RUN_DATA_PIPELINE = True
TRAIN_CUSTOM_MODEL = True
TRAIN_PRETRAINED_MODEL = True
RUN_EVALUATION = True
RUN_XAI = True

def main():
    print("=== Starting AneRBC Anemia Classification Pipeline ===\n")
    
    # 1. DATA PIPELINE
    if RUN_DATA_PIPELINE:
        print("[1] Running Data Pipeline...")
        # Note: Update 'data/AneRBC_raw_data' to your actual local dataset path
        base_dir = 'data/AneRBC_raw_data' 
        
        # Check if data exists before proceeding
        if not os.path.exists(base_dir):
            print(f"ERROR: Dataset not found at {base_dir}. Please download and extract it.")
            return

        image_paths, labels = map_and_validate_dataset(base_dir)
        subset_paths, subset_labels = create_stratified_subset(image_paths, labels, total_samples=4000)
        
        # Create DataLoaders
        train_loader, val_loader, test_loader, label_encoder = prepare_dataloaders(subset_paths, subset_labels)
        class_names = label_encoder.classes_
        print("DataLoaders created successfully!\n")

    # 2. MODEL PREPARATION
    save_directory = './saved_models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize our best models
    model_custom = CNN4Layer(num_classes=2).to(device)
    model_pretrained = setup_pretrained_model('mobilenet', num_classes=2).to(device)

    # 3. TRAINING CUSTOM MODEL
    if TRAIN_CUSTOM_MODEL and RUN_DATA_PIPELINE:
        print("[2] Training Custom 4-Layer CNN...")
        history_custom = train_and_validate(model_custom, train_loader, val_loader, epochs=10, lr=0.001)
        save_model_and_history(model_custom, history_custom, "CNN_4Layer", save_directory)
        print("\n")

    # 4. TRAINING PRETRAINED MODEL
    if TRAIN_PRETRAINED_MODEL and RUN_DATA_PIPELINE:
        print("[3] Training Pretrained MobileNetV2...")
        history_pretrained = train_and_validate(model_pretrained, train_loader, val_loader, epochs=10, lr=0.001)
        save_model_and_history(model_pretrained, history_pretrained, "MobileNetV2", save_directory)
        print("\n")

    # 5. EVALUATION
    if RUN_EVALUATION and RUN_DATA_PIPELINE:
        print("[4] Evaluating Models on Test Set...")
        evaluate_on_test_set(model_custom, test_loader, class_names, "CNN_4Layer")
        evaluate_on_test_set(model_pretrained, test_loader, class_names, "MobileNetV2")
        print("\n")

    # 6. EXPLAINABLE AI (XAI)
    if RUN_XAI and RUN_DATA_PIPELINE:
        print("[5] Generating XAI Integrated Gradients...")
        # Get a single test image
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        sample_img = images[0].unsqueeze(0)
        sample_label = labels[0].item()
        
        print(f"Target Image True Class: {class_names[sample_label]}")
        apply_integrated_gradients(model_custom, sample_img, sample_label, class_names, "CNN_4Layer")
        apply_integrated_gradients(model_pretrained, sample_img, sample_label, class_names, "MobileNetV2")

if __name__ == "__main__":
    main()