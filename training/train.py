import os
import argparse
import torchaudio

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def pargs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data Directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def main():
    # Install FFMPEG
    print("Available audio backends: ")
    print(str(torchaudio.list_audio_backends()))


if __name__ == "__main__":
    main()
