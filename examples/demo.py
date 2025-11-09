"""Small demo: shows how to call the library functions programmatically.

This demo expects you to have a dataset prepared. It doesn't ship data.
"""
from detection.model import train_model, predict_file, load_model


def run_demo():
    print("This demo shows the expected calls:")
    print("train_model(dataset_dir, out_model_path)")
    print("predict_file(model, audio_path)")


if __name__ == "__main__":
    run_demo()
