import argparse
from .model import train_model, load_model, predict_file, train_compare
try:
    from .deep import train_deep, predict_deep
except Exception:
    train_deep = None
    predict_deep = None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="detection", description="Deepfake voice detection (baseline)")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Train a detector from a dataset directory")
    p_train.add_argument("--dataset", required=True, help="Dataset root with 'real' and 'fake' subfolders")
    p_train.add_argument("--out-model", required=True, help="Path to save trained model (joblib)")
    p_train.add_argument("--feature-set", choices=("base", "bio"), default="base", help="Feature set to use")

    p_pred = sub.add_parser("predict", help="Predict one audio file")
    p_pred.add_argument("--model", required=True, help="Path to trained model")
    p_pred.add_argument("--file", required=True, help="Audio file to run prediction on")

    p_tc = sub.add_parser("train-compare", help="Train baseline and bio models and compare them")
    p_tc.add_argument("--dataset", required=True, help="Dataset root with 'real' and 'fake' subfolders")
    p_tc.add_argument("--out-base", required=True, help="Path to save baseline model")
    p_tc.add_argument("--out-bio", required=True, help="Path to save bio model")

    p_deep = sub.add_parser("train-deep", help="Train a simple deep CNN (uses PyTorch, optional GPU)")
    p_deep.add_argument("--dataset", required=True, help="Dataset root with 'real' and 'fake' subfolders")
    p_deep.add_argument("--out-model", required=True, help="Path to save trained deep model (torch .pt)")
    p_deep.add_argument("--epochs", type=int, default=10)
    p_deep.add_argument("--batch-size", type=int, default=8)
    p_deep.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Device to train on if available")

    p_pred_deep = sub.add_parser("predict-deep", help="Predict with deep model")
    p_pred_deep.add_argument("--model", required=True, help="Path to trained deep model (.pt)")
    p_pred_deep.add_argument("--file", required=True, help="Audio file to run prediction on")
    p_pred_deep.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_model(args.dataset, args.out_model, feature_set=args.feature_set)
    elif args.cmd == "predict":
        label = predict_file(args.model, args.file)
        print("fake" if label == 1 else "real")
    elif args.cmd == "train-compare":
        train_compare(args.dataset, args.out_base, args.out_bio)
    elif args.cmd == "train-deep":
        if train_deep is None:
            print("PyTorch/deep training is not available in this environment. Install torch to enable it.")
        else:
            train_deep(args.dataset, args.out_model, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    elif args.cmd == "predict-deep":
        if predict_deep is None:
            print("PyTorch/deep prediction is not available in this environment. Install torch to enable it.")
        else:
            prob = predict_deep(args.model, args.file, device=args.device)
            print(f"P(fake)={prob:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
