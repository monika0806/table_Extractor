import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "a.jpeg"  # change if needed
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------
# LOAD TrOCR
# -----------------------------
print("🔄 Loading TrOCR (handwritten)...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.to(DEVICE)
model.eval()


# -----------------------------
# OCR FUNCTION
# -----------------------------
def ocr_cell(cell_image):
    pil_img = Image.fromarray(cell_image)

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# -----------------------------
# TABLE EXTRACTION
# -----------------------------
def extract_table(image_path):

    print("📐 Detecting grid...")

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found!")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )

    # Horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Vertical line detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    # Combine
    grid = cv2.add(horizontal, vertical)

    # Find contours (cells)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small noise
        if w > 30 and h > 30:
            boxes.append((x, y, w, h))

    if not boxes:
        print("❌ No table grid detected.")
        return

    # Sort boxes by y then x
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Group into rows
    rows = []
    current_row = []
    last_y = -1
    row_threshold = 20  # adjust if needed

    for box in boxes:
        x, y, w, h = box

        if last_y == -1:
            current_row.append(box)
            last_y = y
        elif abs(y - last_y) <= row_threshold:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            last_y = y

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    if not rows:
        print("❌ Rows formation failed.")
        return

    print(f"✅ Detected {len(rows)} rows × {max(len(r) for r in rows)} columns")

    print("\n🔍 Running OCR per cell...\n")

    # OCR each cell
    for row_idx, row in enumerate(rows):
        row_text = []
        for col_idx, (x, y, w, h) in enumerate(row):
            cell = image[y:y+h, x:x+w]

            text = ocr_cell(cell)
            row_text.append(text)

            print(f"[Row {row_idx+1}, Col {col_idx+1}] → {text}")

        print(" | ".join(row_text))
        print("-" * 50)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    extract_table(IMAGE_PATH)