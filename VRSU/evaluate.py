"""
Evaluate saved MobileNetV2 model on flat TEST folder.
Run: python evaluate.py
"""

import os, sys
import numpy as np
import cv2

# ── Hardcoded paths ────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_Data"
MODEL_PATH   = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_sign_mobilenet.h5"
IMG_SIZE     = 96

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print(f"[INFO] TensorFlow {tf.__version__} detected.")
except ImportError:
    print("[ERROR] TensorFlow not installed.")
    sys.exit(1)

try:
    from sklearn.metrics import classification_report
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

# ── Load model ─────────────────────────────────────────────────────────────────
print("[INFO] Loading saved model …")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ── Build class index from DATA folder ────────────────────────────────────────
train_dir = os.path.join(DATA_DIR, "DATA")
classes   = sorted(os.listdir(train_dir))          # ['0', '1', '2', ...]
class_indices = {cls: idx for idx, cls in enumerate(classes)}
print(f"[INFO] {len(class_indices)} classes found.")

# ── Evaluate on flat TEST folder ───────────────────────────────────────────────
test_dir    = os.path.join(DATA_DIR, "TEST")
image_files = [f for f in os.listdir(test_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"[INFO] Found {len(image_files)} test images.")

y_true, y_pred = [], []

for fname in image_files:
    # Parse class from filename prefix e.g. "000_0001_j.png" → class "0"
    label_str = fname.split("_")[0].lstrip("0") or "0"

    if label_str not in class_indices:
        continue

    img = cv2.imread(os.path.join(test_dir, fname))
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    pred_idx = np.argmax(model.predict(img, verbose=0), axis=1)[0]
    true_idx = class_indices[label_str]

    y_true.append(true_idx)
    y_pred.append(pred_idx)

# ── Results ────────────────────────────────────────────────────────────────────
correct = sum(t == p for t, p in zip(y_true, y_pred))
print(f"\n  Test images evaluated : {len(y_true)}")
print(f"  Test accuracy         : {correct / len(y_true) * 100:.2f}%")

if SK_AVAILABLE and len(y_true) > 0:
    num_classes = len(class_indices)
    print("\n[INFO] Classification report (first 15 classes):\n")
    print(classification_report(y_true, y_pred,
                                labels=list(range(min(15, num_classes))),
                                zero_division=0))