import os
import cv2
import numpy as np

# =====================================================
# MAIN FOLDER
# =====================================================

MAIN_FOLDER = r"data(1)"   # Change this path

# =====================================================
# FIND THE SUBFOLDER
# =====================================================

subfolders = [
    os.path.join(MAIN_FOLDER, folder)
    for folder in os.listdir(MAIN_FOLDER)
    if os.path.isdir(os.path.join(MAIN_FOLDER, folder))
]

if len(subfolders) == 0:
    raise Exception("No subfolder found inside MainFolder")

IMAGE_FOLDER = subfolders[0]

print("Image Folder Found:")
print(IMAGE_FOLDER)

# =====================================================
# LOAD ALL JPG IMAGES
# =====================================================

image_paths = []

for file in os.listdir(IMAGE_FOLDER):

    if file.lower().endswith(".jpg"):

        image_paths.append(
            os.path.join(IMAGE_FOLDER, file)
        )

print(f"\nTotal JPG Images Found: {len(image_paths)}")

# =====================================================
# READ IMAGES
# =====================================================

IMG_SIZE = 224

images = []

for img_path in image_paths:

    img = cv2.imread(img_path)

    if img is None:
        print("Cannot read:", img_path)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0

    images.append(img)

    print("Loaded:", os.path.basename(img_path))

images = np.array(images)

print("\nDataset Shape:", images.shape)