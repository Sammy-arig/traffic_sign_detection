import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} — WORKS")
        cap.release()
    else:
        print(f"Camera {i} — not found")