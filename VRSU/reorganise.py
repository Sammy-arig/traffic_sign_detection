"""
Reorganize flat TEST folder into class subfolders.
Run once: python reorganize_test.py
"""

import os, shutil

TEST_DIR = r"C:\Users\samee\PycharmProjects\PythonProject\VRSU\traffic_Data\TEST"

image_files = [f for f in os.listdir(TEST_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

moved = 0
for fname in image_files:
    class_id = str(int(fname.split("_")[0]))   # "001" → "1"
    class_folder = os.path.join(TEST_DIR, class_id)
    os.makedirs(class_folder, exist_ok=True)
    shutil.move(os.path.join(TEST_DIR, fname),
                os.path.join(class_folder, fname))
    moved += 1

print(f"[INFO] Done. {moved} images moved into class subfolders.")