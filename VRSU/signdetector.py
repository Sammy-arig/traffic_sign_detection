import cv2
import numpy as np
import os
import sys
import time

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not found. Run:  pip install tensorflow")

CLASS_LABELS = {
    0:  "Speed limit (5km/h)",
    1:  "Speed limit (15km/h)",
    2:  "Speed limit (30km/h)",
    3:  "Speed limit (40km/h)",
    4:  "Speed limit (50km/h)",
    5:  "Speed limit (60km/h)",
    6:  "Speed limit (70km/h)",
    7:  "Speed limit (80km/h)",
    8:  "Dont Go straight or left",
    9:  "Dont Go straight or Right",
    10: "Dont Go straight",
    11: "Dont Go Left",
    12: "Dont Go Left or Right",
    13: "Dont Go Right",
    14: "Dont overtake from Left",
    15: "No Uturn",
    16: "No Car",
    17: "No horn",
    18: "Speed limit (40km/h)",
    19: "Speed limit (50km/h)",
    20: "Go straight or right",
    21: "Go straight",
    22: "Go Left",
    23: "Go Left or right",
    24: "Go Right",
    25: "Keep Left",
    26: "Keep Right",
    27: "Roundabout mandatory",
    28: "Watch out for cars",
    29: "Horn",
    30: "Bicycles crossing",
    31: "Uturn",
    32: "Road Divider",
    33: "Traffic signals",
    34: "Danger Ahead",
    35: "Zebra Crossing",
    36: "Bicycles crossing",
    37: "Children crossing",
    38: "Dangerous curve to the left",
    39: "Dangerous curve to the right",
    40: "Unknown1",
    41: "Unknown2",
    42: "Unknown3",
    43: "Go right or straight",
    44: "Go left or straight",
    45: "Unknown4",
    46: "ZigZag Curve",
    47: "Train Crossing",
    48: "Under Construction",
    49: "Unknown5",
    50: "Fences",
    51: "Heavy Vehicle Accidents",
    52: "Unknown6",
    53: "Give Way",
    54: "No stopping",
    55: "No entry",
    56: "Unknown7",
    57: "Unknown8",
}

MODEL_PATH = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_sign_model.h5"
IMG_SIZE   = 64
CONFIDENCE = 0.60
CAMERA_ID  = 0


def get_sign_color(class_id):
    if class_id <= 7:   return (0, 0, 220)      # Red  - Speed limits
    if class_id <= 17:  return (0, 0, 220)      # Red  - Prohibitory
    if class_id <= 31:  return (220, 130, 0)    # Blue - Mandatory
    if class_id <= 39:  return (0, 200, 255)    # Yellow - Warning
    if class_id <= 55:  return (0, 165, 255)    # Orange - Other
    return (150, 150, 150)                       # Grey  - Unknown


class TrafficSignROI:
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
        red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        red_mask    = cv2.bitwise_or(red1, red2)
        blue_mask   = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20,  100, 100), (35,  255, 255))

        combined = cv2.bitwise_or(red_mask, blue_mask)
        combined = cv2.bitwise_or(combined, yellow_mask)

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h_frame, w_frame = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if not (0.4 < aspect < 2.5):
                continue
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(w_frame - x, w + 2 * pad_x)
            h = min(h_frame - y, h + 2 * pad_y)
            boxes.append((x, y, w, h))
        return boxes


def centre_crop_boxes(frame):
    fh, fw = frame.shape[:2]
    sizes  = [min(fw, fh) // 2, min(fw, fh) // 3]
    return [((fw - s) // 2, (fh - s) // 2, s, s) for s in sizes]


def draw_prediction(frame, box, label, confidence, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label}  {confidence*100:.1f}%"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    label_y = y - 8 if y - 8 - th - 4 >= 0 else y + h + th + 8
    cv2.rectangle(frame,
                  (x, label_y - th - 4),
                  (x + tw + 4, label_y + baseline),
                  color, cv2.FILLED)
    cv2.putText(frame, text, (x + 2, label_y - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(frame, fps, total_detections):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, "Indian Traffic Sign Detector",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Detections: {total_detections}",
                (w - 270, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to quit  |  's' to save screenshot",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


def run_detector():
    if not TF_AVAILABLE:
        print("[ERROR] TensorFlow is required.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        sys.exit(1)

    print(f"[INFO] Loading model from {MODEL_PATH} …")
    model = load_model(MODEL_PATH)
    print(f"[INFO] Model loaded. Input shape: {model.input_shape}")

    roi_detector = TrafficSignROI()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Camera started. Press 'q' to quit, 's' to screenshot.\n")

    fps              = 0.0
    frame_count      = 0
    total_dets       = 0
    t_start          = time.time()
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        display = frame.copy()

        boxes = roi_detector.detect(frame)
        if not boxes:
            boxes = centre_crop_boxes(frame)

        for box in boxes:
            x, y, w, h = box
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_batch   = np.expand_dims(roi_resized.astype("float32") / 255.0, axis=0)

            preds      = model.predict(roi_batch, verbose=0)[0]
            class_id   = int(np.argmax(preds))
            confidence = float(preds[class_id])

            if confidence < CONFIDENCE:
                continue

            label = CLASS_LABELS.get(class_id, f"Class {class_id}")
            color = get_sign_color(class_id)
            draw_prediction(display, box, label, confidence, color)
            total_dets += 1

        elapsed = time.time() - t_start
        if elapsed >= 0.5:
            fps         = frame_count / elapsed
            frame_count = 0
            t_start     = time.time()

        draw_hud(display, fps, total_dets)
        cv2.imshow("Indian Traffic Sign Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(fname, display)
            screenshot_count += 1
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector stopped.")


if __name__ == "__main__":
    run_detector()