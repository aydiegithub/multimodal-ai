import os
import torch
import json
from colorama import init, Fore
from training.models import MultimodalSentimentModel, MultimodalTrainer
from dataset import prepare_dataloaders


def train_local():
    init(autoreset=True)

    BASE_DATA_DIR = "dataset"

    TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train/train_sent_emo.csv")
    TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "train/train_splits")

    DEV_CSV = os.path.join(BASE_DATA_DIR, "dev/dev_sent_emo.csv")
    DEV_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "dev/dev_splits_complete")

    TEST_CSV = os.path.join(BASE_DATA_DIR, "test/test_sent_emo.csv")
    TEST_VIDEO_DIR = os.path.join(
        BASE_DATA_DIR, "test/output_repeated_splits_test")

    MODEL_SAVE_DIR = "saved_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 5e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Fore.CYAN + f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # --- Data Preparation ---
    print(Fore.CYAN + "\nPreparing DataLoaders...")
    try:
        train_loader, dev_loader, test_loader = prepare_dataloaders(
            train_csv=TRAIN_CSV, train_video_dir=TRAIN_VIDEO_DIR,
            dev_csv=DEV_CSV, dev_video_dir=DEV_VIDEO_DIR,
            test_csv=TEST_CSV, test_video_dir=TEST_VIDEO_DIR,
            batch_size=BATCH_SIZE
        )
    except FileNotFoundError as e:
        print(Fore.RED + f"Error loading data: {e}")
        print("Please ensure your dataset paths in 'start_training.py' are correct.")
        return

    print(Fore.CYAN + "\nInitializing Model...")
    model = MultimodalSentimentModel().to(device)

    # Initialize Trainer
    trainer = MultimodalTrainer(model, train_loader, dev_loader)

    # Training Loop
    print(Fore.CYAN + "\nStarting Training Loop...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(Fore.YELLOW + f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")

        # Train
        train_metrics = trainer.train_epoch()
        print(f"Train Loss: {train_metrics['total']:.4f}")

        # Validate
        val_loss, val_metrics = trainer.evaluate(dev_loader, phase="val")
        print(f"Val Loss:   {val_loss['total']:.4f}")
        print(
            f"Val Acc (Emo): {val_metrics['emotion_accuracy']:.2f} | Acc (Sent): {val_metrics['sentiment_accuracy']:.2f}")

        # Save Best Model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            save_path = os.path.join(
                MODEL_SAVE_DIR, "best_multimodal_model.pth")
            torch.save(model.state_dict(), save_path)
            print(Fore.GREEN + f"Saved new best model to {save_path}")

    # Final Test Evaluation
    print(Fore.CYAN + "\nTraining Complete. Evaluating on Test Set using Best Model...")

    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(
        MODEL_SAVE_DIR, "best_multimodal_model.pth")))
    model.to(device)

    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")

    print(Fore.GREEN + "\nTest Set Results:")
    print(json.dumps(test_metrics, indent=4))


if __name__ == "__main__":
    train_local()
