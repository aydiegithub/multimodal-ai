import os
import sys
import torch
import json
from colorama import init, Fore

from models import MultimodalSentimentModel, MultimodalTrainer
from dataset import prepare_dataloaders


def train_local():
    init(autoreset=True)

    BASE_DATA_DIR = "../dataset"

    # Verify directories exist before starting
    if not os.path.exists(BASE_DATA_DIR):
        print(
            Fore.RED + f"CRITICAL ERROR: Could not find dataset at {os.path.abspath(BASE_DATA_DIR)}")
        print("Please check your folder structure.")
        return

    # Define specific paths based on your MELD dataset structure
    TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train/train_sent_emo.csv")
    TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train/train_splits")

    DEV_CSV = os.path.join(BASE_DATA_DIR, "dev/dev_sent_emo.csv")
    DEV_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "dev/dev_splits_complete")

    TEST_CSV = os.path.join(BASE_DATA_DIR, "test/test_sent_emo.csv")
    TEST_VIDEO_DIR = os.path.join(
        BASE_DATA_DIR, "test/output_repeated_splits_test")

    # Directory to save the best model
    MODEL_SAVE_DIR = "saved_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.CYAN + f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

    # Data Preparation
    print(Fore.CYAN + "\nPreparing DataLoaders...")
    try:
        train_loader, dev_loader, test_loader = prepare_dataloaders(
            train_csv=TRAIN_CSV, train_video_dir=TRAIN_VIDEO_DIR,
            dev_csv=DEV_CSV, dev_video_dir=DEV_VIDEO_DIR,
            test_csv=TEST_CSV, test_video_dir=TEST_VIDEO_DIR,
            batch_size=BATCH_SIZE
        )
        print(Fore.GREEN + "DataLoaders ready.")
    except Exception as e:
        print(Fore.RED + f"Error loading data: {e}")
        print("Double check that the CSV files and Video folders exist in your 'dataset' directory.")
        return

    # Model Initialization
    print(Fore.CYAN + "\nInitializing Model...")
    model = MultimodalSentimentModel().to(device)

    # Initialize Trainer
    trainer = MultimodalTrainer(model, train_loader, dev_loader)

    # Training Loop
    print(Fore.CYAN + "\nStarting Training Loop...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(Fore.YELLOW + f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")

        # 1. Train
        train_metrics = trainer.train_epoch()
        print(f"Train Loss: {train_metrics['total']:.4f}")

        # 2. Validate
        val_loss, val_metrics = trainer.evaluate(dev_loader, phase="val")
        print(f"Val Loss:   {val_loss['total']:.4f}")
        print(
            f"Val Acc (Emo): {val_metrics['emotion_accuracy']:.2f} | Acc (Sent): {val_metrics['sentiment_accuracy']:.2f}")

        # 3. Save Best Model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            save_path = os.path.join(
                MODEL_SAVE_DIR, "best_multimodal_model.pth")
            torch.save(model.state_dict(), save_path)
            print(Fore.GREEN + f"Saved new best model to {save_path}")

        # Monitor GPU Memory
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU Mem: {mem_used:.2f} GB")

    # Final Test Evaluation
    print(Fore.CYAN + "\nTraining Complete. Evaluating on Test Set using Best Model...")

    # Load best model weights
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_multimodal_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        print(Fore.GREEN + "Loaded best model weights.")
    else:
        print(Fore.RED + "Warning: Best model file not found. Using current model weights.")

    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")

    print(Fore.GREEN + "\nTest Set Results:")
    print(json.dumps(test_metrics, indent=4))


if __name__ == "__main__":
    train_local()
