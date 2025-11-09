from .features import extract_features
from .model import train_model, load_model, predict_file
try:
	from .deep import train_deep, load_deep, predict_deep
	__all__ = ["extract_features", "train_model", "load_model", "predict_file", "train_deep", "load_deep", "predict_deep"]
except Exception:
	__all__ = ["extract_features", "train_model", "load_model", "predict_file"]
