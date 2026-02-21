import torch
import cv2
import os
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

# ---------------- CONFIG ----------------
IMAGE_PATH = "h.jpg"
OUTPUT_DIR = "detected_cells"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print("Loading Table Transformer model...")

processor = DetrImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------------- DETECT & CROP CELLS ----------------
def detect_and_crop_cells(image_path):

    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    h, w = img_cv.shape[:2]

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=0.7,
        target_sizes=[(h, w)]
    )[0]

    cell_count = 0

    print("\nDetected objects:")
    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):

        label_name = model.config.id2label[label.item()]
        box = box.cpu().numpy().astype(int)

        # 🔥 Detect only table cells
        if label_name == "table cell":
            x1, y1, x2, y2 = box

            # Crop cell
            cropped_cell = img_cv[y1:y2, x1:x2]

            # Save cell image
            cell_filename = os.path.join(OUTPUT_DIR, f"cell_{cell_count}.jpg")
            cv2.imwrite(cell_filename, cropped_cell)

            print(f"Saved: {cell_filename} | Score: {score:.2f}")
            cell_count += 1

    print(f"\nTotal cells saved: {cell_count}")


if __name__ == "__main__":
    detect_and_crop_cells(IMAGE_PATH)