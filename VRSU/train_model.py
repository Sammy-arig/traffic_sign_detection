import random
random.seed(42)
import numpy as np
np.random.seed(42)
import os, sys, time
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR     = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_Data"
OUTPUT_MODEL = "traffic_sign_model.h5"
EPOCHS       = 30
BATCH_SIZE   = 32
IMG_SIZE     = 64

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    from tensorflow.keras import layers, models, callbacks, regularizers
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    print(f"[INFO] TensorFlow {tf.__version__} detected.")
except ImportError:
    print("[ERROR] TensorFlow not installed.")
    sys.exit(1)

try:
    from sklearn.metrics import classification_report
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


def build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    inp = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="TrafficSignNet")


def plot_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
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
    print(f"[INFO] Image size: {IMG_SIZE}x{IMG_SIZE}  |  Epochs: {EPOCHS}\n")

    train_gen = ImageDataGenerator(
        rescale           = 1.0 / 255,
        rotation_range    = 15,
        zoom_range        = 0.20,
        width_shift_range = 0.15,
        height_shift_range= 0.15,
        shear_range       = 0.10,
        brightness_range  = [0.7, 1.3],
        horizontal_flip   = False,
        fill_mode         = "nearest",
        validation_split  = 0.15,
    )

    target = (IMG_SIZE, IMG_SIZE)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=target, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", shuffle=True)

    val_data = train_gen.flow_from_directory(
        train_dir, target_size=target, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False)

    model = build_model(num_classes, IMG_SIZE)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(
            OUTPUT_MODEL, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1, min_delta=0.001),
        callbacks.CSVLogger("training_log.csv"),
    ]

    print(f"\n[INFO] Starting training for up to {EPOCHS} epochs …\n")
    t0 = time.time()
    history = model.fit(
        train_data, epochs=EPOCHS,
        validation_data=val_data, callbacks=cbs,
    )
    print(f"\n[INFO] Training finished in {(time.time()-t0)/60:.1f} min")
    print(f"[INFO] Best model saved → {OUTPUT_MODEL}")

    plot_history(history)

    # ── Evaluate on TEST (flat or subfolders) ─────────────────────────────
    print("\n[INFO] Evaluating on test set …")

    class_indices = train_data.class_indices
    y_true, y_pred_list = [], []

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
        img = img.astype("float32") / 255.0
        batch_imgs.append(img)
        batch_labels.append(class_indices[label_str])

    batch_imgs = np.array(batch_imgs)
    preds      = np.argmax(model.predict(batch_imgs, batch_size=64, verbose=1), axis=1)
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

    print(f"\n[INFO] Done! Model saved as: {OUTPUT_MODEL}\n")


if __name__ == "__main__":
    train()