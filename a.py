import cv2
import os
import torch
import numpy as np
import pandas as pd
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- CONFIG ----------------
IMAGE_PATH = "c.jpeg"
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
model.to(DEVICE).eval()

# ---------------- OCR ----------------
def trocr_ocr(img):
    if img is None or img.size == 0:
        return ""
    pil_img = Image.fromarray(img).convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=32)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ---------------- CELL DETECTION ----------------
def detect_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,21,5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    H, W = image.shape[:2]
    cells = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 and h > 30 and w < W*0.9 and h < H*0.9:
            cells.append((x, y, w, h))

    return sorted(cells, key=lambda b: (b[1], b[0]))

# ---------------- RECONSTRUCT TABLE ----------------
def reconstruct_table(cells, img):
    entries = []

    for i, (x, y, w, h) in enumerate(cells):
        crop = img[y:y+h, x:x+w]
        text = trocr_ocr(crop)

        cv2.imwrite(os.path.join(DEBUG_DIR, f"{i}.jpg"), crop)

        cx = x + w // 2
        cy = y + h // 2
        entries.append((cx, cy, text))

    ROW_THRESH = 35
    rows = []

    for cx, cy, txt in entries:
        placed = False
        for row in rows:
            if abs(row[0][1] - cy) < ROW_THRESH:
                row.append((cx, cy, txt))
                placed = True
                break
        if not placed:
            rows.append([(cx, cy, txt)])

    rows = sorted(rows, key=lambda r: np.mean([v[1] for v in r]))

    table = []
    for r in rows:
        r_sorted = sorted(r, key=lambda v: v[0])
        table.append([v[2] for v in r_sorted])

    max_cols = max(len(r) for r in table)
    table = [r + [""]*(max_cols-len(r)) for r in table]

    return table

# ---------------- AUTO HEADER DETECTION ----------------
def auto_detect_headers(table):
    def looks_like_header(row):
        alpha = sum(1 for c in row if re.search(r"[A-Za-z]", c))
        return alpha >= len(row)//2

    for i, row in enumerate(table):
        if looks_like_header(row):
            headers = [c if c else f"col_{j+1}" for j,c in enumerate(row)]
            return headers, table[i+1:]

    headers = [f"col_{i+1}" for i in range(len(table[0]))]
    return headers, table[1:]

# ---------------- MAIN ----------------
def extract_table(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Image not found")

    print("📐 Detecting cells...")
    cells = detect_cells(img)
    print("✍️ OCR + reconstruct...")
    table = reconstruct_table(cells, img)

    print("🧠 Auto-detecting headers...")
    headers, data = auto_detect_headers(table)

    df = pd.DataFrame(data, columns=headers)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("✅ CSV saved:", OUTPUT_CSV)
    print("✅ Excel saved:", OUTPUT_XLSX)
    print("🧩 Debug crops:", DEBUG_DIR)

# ---------------- RUN ----------------
if __name__ == "__main__":
    extract_table(IMAGE_PATH)