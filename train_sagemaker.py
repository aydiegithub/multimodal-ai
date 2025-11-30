from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from colorama import Fore


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="bucket-name/tensorboard",
        container_local_output_path="opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="my-new_role",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch_size": 32,
            "epoch": 25,
        },
        tensorboard_config=tensorboard_config
    )

    print(Fore.GREEN + "STARTED TRAINING..." + Fore.RESET)
    # Start training
    estimator.fit({
        "training": "s3://bucket-name/dataset/training",
        "validation": "s3://bucket-name/dataset/dev",
        "test": "s3://bucket-name/dataset/test"
    })


if __name__ == "__main__":
    start_training()
