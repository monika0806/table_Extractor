import cv2
import os
import torch
import numpy as np
import pandas as pd
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- CONFIG ----------------
IMAGE_PATH = "b.jpeg"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_cells")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "output.csv")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "output.xlsx")

# ---------------- LOAD OCR ----------------
print("🔄 Loading TrOCR (handwritten)...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
model.eval()


# ---------------- OCR FUNCTION ----------------
def trocr_ocr(img):
    if img is None or img.size == 0:
        return ""

    pil_img = Image.fromarray(img).convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=32)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text


# ---------------- LINE-BASED CELL DETECTION ----------------
def detect_cells(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strong binary threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine grid
    grid = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    H, W = image.shape[:2]
    cells = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if 40 < w < W * 0.9 and 30 < h < H * 0.9:
            cells.append((x, y, w, h))

    print(f"📦 Detected {len(cells)} cells")

    return cells


# ---------------- GROUP ROWS ----------------
def group_rows(cells, row_thresh=30):

    cells = sorted(cells, key=lambda b: b[1])
    rows = []

    for cell in cells:
        x, y, w, h = cell
        placed = False

        for row in rows:
            if abs(row[0][1] - y) < row_thresh:
                row.append(cell)
                placed = True
                break

        if not placed:
            rows.append([cell])

    for row in rows:
        row.sort(key=lambda b: b[0])

    return rows


# ---------------- RECONSTRUCT TABLE ----------------
def reconstruct_table(cells, img):

    rows = group_rows(cells)

    table = []

    for r_idx, row in enumerate(rows):
        row_text = []

        for c_idx, (x, y, w, h) in enumerate(row):

            crop = img[y:y+h, x:x+w]

            cv2.imwrite(
                os.path.join(DEBUG_DIR, f"r{r_idx}_c{c_idx}.jpg"),
                crop
            )

            text = trocr_ocr(crop)
            row_text.append(text)

        table.append(row_text)

    if len(table) == 0:
        return []

    max_cols = max(len(r) for r in table)
    table = [r + [""] * (max_cols - len(r)) for r in table]

    return table


# ---------------- AUTO HEADER DETECTION ----------------
def auto_detect_headers(table):

    if len(table) == 0:
        return [], []

    def looks_like_header(row):
        alpha = sum(1 for c in row if re.search(r"[A-Za-z]", c))
        return alpha >= len(row) // 2

    for i, row in enumerate(table):
        if looks_like_header(row):
            headers = [c if c else f"col_{j+1}" for j, c in enumerate(row)]
            return headers, table[i+1:]

    headers = [f"col_{i+1}" for i in range(len(table[0]))]
    return headers, table[1:]


# ---------------- MAIN EXTRACTION ----------------
def extract_table(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Image not found")

    print("📐 Detecting cells...")
    cells = detect_cells(img)

    if len(cells) == 0:
        print("❌ No cells detected. Try adjusting threshold or image contrast.")
        return

    print("✍️ OCR + reconstruct...")
    table = reconstruct_table(cells, img)

    if len(table) == 0:
        print("❌ Table reconstruction failed.")
        return

    print("🧠 Auto-detecting headers...")
    headers, data = auto_detect_headers(table)

    df = pd.DataFrame(data, columns=headers)

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("✅ CSV saved:", OUTPUT_CSV)
    print("✅ Excel saved:", OUTPUT_XLSX)
    print("🧩 Debug crops saved in:", DEBUG_DIR)


# ---------------- RUN ----------------
if __name__ == "__main__":
    extract_table(IMAGE_PATH)
