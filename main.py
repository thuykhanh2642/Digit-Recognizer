import argparse
import os
import sys
import math
import numpy as np
from pathlib import Path

# 3rd party
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Optional, only used if you call --plot
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageOps
except Exception as e:
    Image = None  # We'll error nicely if user asks for predict-image

BUNDLE_PATH = Path("digits_mlp.joblib")

def load_data(test_size=0.2, seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float32)   # (n, 64)
    y = digits.target.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (X_train, y_train), (X_test, y_test), digits.target_names

def build_model():
    # Small, fast MLP
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )

def train(args):
    (X_train, y_train), (X_test, y_test), classes = load_data()
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train_sc, y_train)

    # Evaluate
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    bundle = {"model": model, "scaler": scaler, "classes": classes}
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Saved model bundle -> {BUNDLE_PATH.resolve()}")

    if args.plot:
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        plt.figure()
        disp.plot(values_format="d")
        plt.title("Digits MLP - Confusion Matrix")
        plt.show()

        # Loss curve
        if hasattr(model, "loss_curve_"):
            plt.figure()
            plt.plot(model.loss_curve_)
            plt.xlabel("Epoch")
            plt.ylabel("Training loss")
            plt.title("MLP Loss Curve")
            plt.show()

def ensure_bundle():
    if not BUNDLE_PATH.exists():
        print("No trained model found. Run: python digits_nn.py train", file=sys.stderr)
        sys.exit(1)
    return joblib.load(BUNDLE_PATH)

def eval_model(args):
    bundle = ensure_bundle()
    model, scaler = bundle["model"], bundle["scaler"]
    (X_train, y_train), (X_test, y_test), _ = load_data()
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

def predict_values(args):
    bundle = ensure_bundle()
    model, scaler = bundle["model"], bundle["scaler"]
    vals = np.array(args.values, dtype=np.float32).reshape(1, -1)
    if vals.shape[1] != 64:
        print("You must pass exactly 64 numbers (flattened 8x8).", file=sys.stderr)
        sys.exit(2)
    vals_sc = scaler.transform(vals)
    pred = int(model.predict(vals_sc)[0])
    print(pred)

def to_digits_space(img_8x8_gray):
    """
    Convert a PIL grayscale 8x8 image to the sklearn digits feature vector scale (0..16).
    The sklearn digits dataset uses small 8x8 images with pixel intensities in [0,16].
    This function returns a float32 vector of length 64 in that range.
    """
    arr = np.asarray(img_8x8_gray, dtype=np.float32)  # 8x8, 0..255
    # Normalize to 0..16
    arr = (arr / 255.0) * 16.0
    return arr.flatten().astype(np.float32)

def preprocess_image(path, invert_if_needed=True):
    if Image is None:
        print("Pillow is required for image prediction. Install with: pip install pillow", file=sys.stderr)
        sys.exit(3)

    img = Image.open(path).convert("L")  # grayscale
    # Center-crop to square, then resize to 8x8
    w, h = img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    img = img.resize((8, 8), Image.BILINEAR)

    # Many user-written digits are black on white or vice-versa.
    # Heuristic: if background seems dark, invert to make background dark and digit bright.
    if invert_if_needed:
        np_img = np.array(img, dtype=np.float32)
        if np.mean(np_img) < 127:
            img = ImageOps.invert(img)

    return to_digits_space(img)

def predict_image(args):
    bundle = ensure_bundle()
    model, scaler = bundle["model"], bundle["scaler"]
    feats = preprocess_image(args.image)
    feats = feats.reshape(1, -1)
    feats_sc = scaler.transform(feats)
    pred = int(model.predict(feats_sc)[0])
    print(pred)

    if args.show:
        # Visualize what the model saw
        plt.figure()
        plt.imshow(feats.reshape(8,8), interpolation="nearest")
        plt.title(f"Predicted: {pred}")
        plt.axis("off")
        plt.show()

def main():
    p = argparse.ArgumentParser(description="Handwritten digit MLP (train/eval/predict).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model and save to digits_mlp.joblib")
    p_train.add_argument("--plot", action="store_true", help="Show confusion matrix and loss curve after training")

    p_eval = sub.add_parser("eval", help="Evaluate the saved model on the built-in test split")

    p_pred = sub.add_parser("predict", help="Predict from 64 raw values (flattened 8x8).")
    p_pred.add_argument("--values", nargs="+", type=float, required=True, help="64 numbers (0..16)")

    p_predimg = sub.add_parser("predict-image", help="Predict from a PNG/JPG you provide.")
    p_predimg.add_argument("image", type=str, help="Path to your image file")
    p_predimg.add_argument("--show", action="store_true", help="Show the 8x8 the model sees")

    args = p.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_model(args)
    elif args.cmd == "predict":
        predict_values(args)
    elif args.cmd == "predict-image":
        predict_image(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
