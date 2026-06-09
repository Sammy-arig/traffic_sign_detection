"""
Train Indian Traffic Sign Classifier using MobileNetV2 Transfer Learning.
Two-phase training:
  Phase 1 — Train only the new classification head (frozen base)
  Phase 2 — Fine-tune the top layers of MobileNetV2 (unfrozen)
"""

import os, sys, time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

DATA_DIR     = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_Data"
OUTPUT_MODEL = "traffic_sign_mobilenet.h5"

PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 25
BATCH_SIZE    = 32
IMG_SIZE      = 96

random.seed(42)
np.random.seed(42)

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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


def build_mobilenet_model(num_classes: int, img_size: int):
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = layers.Input(shape=(img_size, img_size, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(512, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="MobileNetV2_TrafficSigns")
    return model, base


def plot_history(histories, save_path="training_history_mobilenet.png"):
    acc      = histories[0].history["accuracy"]     + histories[1].history["accuracy"]
    val_acc  = histories[0].history["val_accuracy"] + histories[1].history["val_accuracy"]
    loss     = histories[0].history["loss"]         + histories[1].history["loss"]
    val_loss = histories[0].history["val_loss"]     + histories[1].history["val_loss"]
    phase2_start = len(histories[0].history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_vals, val_vals, title in [
        (axes[0], acc,  val_acc,  "Accuracy"),
        (axes[1], loss, val_loss, "Loss"),
    ]:
        ax.plot(train_vals, label="Train")
        ax.plot(val_vals,   label="Validation")
        ax.axvline(x=phase2_start, color="red", linestyle="--", label="Fine-tune start")
        ax.set_title(title); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[INFO] Training plot saved → {save_path}")


def train():
    train_dir = os.path.join(DATA_DIR, "DATA")
    test_dir  = os.path.join(DATA_DIR, "TEST")

    for d in [train_dir, test_dir]:
        if not os.path.isdir(d):
            print(f"[ERROR] Directory not found: {d}")
            sys.exit(1)

    num_classes = len(os.listdir(train_dir))
    print(f"[INFO] Detected {num_classes} classes in {train_dir}")
    print(f"[INFO] Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"[INFO] Phase 1 epochs: {PHASE1_EPOCHS}  |  Phase 2 epochs: {PHASE2_EPOCHS}\n")

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        brightness_range=[0.7, 1.3],
        horizontal_flip=False,
        fill_mode="nearest",
        validation_split=0.20,
    )

    target = (IMG_SIZE, IMG_SIZE)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=target, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", shuffle=True)

    val_data = train_gen.flow_from_directory(
        train_dir, target_size=target, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False)

    model, base = build_mobilenet_model(num_classes, IMG_SIZE)
    model.summary()

    # ── PHASE 1 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1: Training classification head (base frozen)")
    print("="*60 + "\n")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs_phase1 = [
        callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1),
        callbacks.CSVLogger("training_log_phase1.csv"),
    ]

    t0 = time.time()
    history1 = model.fit(
        train_data, epochs=PHASE1_EPOCHS,
        validation_data=val_data, callbacks=cbs_phase1,
    )
    print(f"[INFO] Phase 1 done in {(time.time()-t0)/60:.1f} min")

    # ── PHASE 2 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("="*60 + "\n")

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"[INFO] Unfrozen {trainable_count} layers in MobileNetV2 base")

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs_phase2 = [
        callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3,
            min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1),
        callbacks.CSVLogger("training_log_phase2.csv"),
    ]

    t0 = time.time()
    history2 = model.fit(
        train_data, epochs=PHASE2_EPOCHS,
        validation_data=val_data, callbacks=cbs_phase2,
    )
    print(f"[INFO] Phase 2 done in {(time.time()-t0)/60:.1f} min")
    print(f"[INFO] Best model saved → {OUTPUT_MODEL}")

    plot_history([history1, history2])

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on test set …")

    class_indices = train_data.class_indices
    y_true, y_pred_list = [], []

    # collect all test images (flat or subfolders)
    image_files = []
    for entry in os.listdir(test_dir):
        entry_path = os.path.join(test_dir, entry)
        if os.path.isdir(entry_path):
            for fname in os.listdir(entry_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(entry_path, fname))
        elif entry.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(test_dir, entry))

    print(f"[INFO] Found {len(image_files)} test images.")

    # batch predict for speed
    batch_imgs, batch_labels = [], []
    for fpath in image_files:
        fname     = os.path.basename(fpath)
        label_str = str(int(fname.split("_")[0]))
        if label_str not in class_indices:
            continue
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img.astype("float32"))
        batch_imgs.append(img)
        batch_labels.append(class_indices[label_str])

    batch_imgs = np.array(batch_imgs)
    preds = np.argmax(model.predict(batch_imgs, batch_size=64, verbose=1), axis=1)
    y_true      = batch_labels
    y_pred_list = preds.tolist()

    correct = sum(t == p for t, p in zip(y_true, y_pred_list))
    print(f"\n  Test images evaluated : {len(y_true)}")
    print(f"  Test accuracy         : {correct / len(y_true) * 100:.2f}%")

    if SK_AVAILABLE and len(y_true) > 0:
        print("\n[INFO] Classification report (first 15 classes):\n")
        print(classification_report(y_true, y_pred_list,
                                    labels=list(range(min(15, num_classes))),
                                    zero_division=0))


if __name__ == "__main__":
    train()