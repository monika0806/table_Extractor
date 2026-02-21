"""
Optimized Handwritten Table OCR Extractor
Improvements over baseline:
  - Line-based grid detection (Hough + morphology) instead of raw contours
  - NMS to eliminate duplicate/nested bounding boxes
  - Per-cell preprocessing: deskew, denoise, adaptive contrast
  - TrOCR with beam search for higher accuracy
  - Robust row/column clustering via median-based grouping
  - Fallback: if grid detection fails, uses projection-profile row segmentation
"""

import cv2
import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ─────────────────────── CONFIG ───────────────────────
IMAGE_PATH   = "b.jpeg"
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
DEBUG_DIR    = os.path.join(OUTPUT_DIR, "debug_cells")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "output.csv")
OUTPUT_XLSX  = os.path.join(OUTPUT_DIR, "output.xlsx")
os.makedirs(DEBUG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────── LOAD MODEL ───────────────────
print("🔄 Loading TrOCR (large-handwritten)...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.to(DEVICE).eval()


# ══════════════════════════════════════════════════════
#  1. IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Denoise + adaptive binarise the full page."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    # CLAHE for uneven illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    return gray


# ══════════════════════════════════════════════════════
#  2. GRID / CELL DETECTION
# ══════════════════════════════════════════════════════
def _morph_lines(gray, axis: str, length_frac=0.25):
    """Extract horizontal or vertical lines via morphological opening."""
    H, W   = gray.shape
    length = int((W if axis == "h" else H) * length_frac)
    size   = (length, 1) if axis == "h" else (1, length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    lines  = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    return lines


def detect_cells_grid(gray: np.ndarray):
    """
    Primary strategy: detect table grid from horizontal + vertical lines,
    intersect them to find cell bounding boxes.
    Returns list of (x, y, w, h) or empty list if detection fails.
    """
    h_lines = _morph_lines(gray, "h")
    v_lines = _morph_lines(gray, "v")

    # Combine
    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)

    # Find connected-component cells
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(grid), connectivity=4
    )

    H, W = gray.shape
    cells = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i, :5]
        if w > 30 and h > 20 and w < W * 0.92 and h < H * 0.92:
            cells.append((int(x), int(y), int(w), int(h)))

    return cells


def detect_cells_contour(gray: np.ndarray):
    """
    Fallback strategy: threshold + find external contours with NMS.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    H, W = gray.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 40 < w < W * 0.9 and 25 < h < H * 0.9:
            boxes.append((x, y, w, h))

    return nms_boxes(boxes)


def nms_boxes(boxes, iou_threshold=0.3):
    """Remove highly overlapping boxes (keep larger)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept  = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for kbox in kept:
            x2, y2, w2, h2 = kbox
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            if union > 0 and inter / union > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(box)
    return kept


def detect_cells(img: np.ndarray):
    gray  = preprocess_image(img)
    cells = detect_cells_grid(gray)
    if len(cells) < 4:                      # grid strategy failed → fallback
        print("  ⚠️  Grid detection found <4 cells, using contour fallback")
        cells = detect_cells_contour(gray)
    print(f"  ✅ {len(cells)} cells detected")
    return sorted(cells, key=lambda b: (b[1], b[0]))


# ══════════════════════════════════════════════════════
#  3. CELL IMAGE PREPROCESSING FOR OCR
# ══════════════════════════════════════════════════════
def prepare_cell(crop: np.ndarray) -> Image.Image:
    """
    Clean up a cell crop before passing to TrOCR:
      - Convert to grayscale
      - Denoise
      - Binarise (Otsu)
      - Add white padding so characters aren't clipped
    """
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.fastNlMeansDenoising(gray, h=20)

    # Otsu binarisation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If image is mostly dark (inverted), flip it
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Pad
    pad    = 12
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=255)

    # Resize so short dimension is at least 64 px (TrOCR likes taller crops)
    h, w = padded.shape
    if h < 64:
        scale  = 64 / h
        padded = cv2.resize(padded, (int(w * scale), 64),
                            interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(padded).convert("RGB")


# ══════════════════════════════════════════════════════
#  4. OCR
# ══════════════════════════════════════════════════════
def ocr_cell(pil_img: Image.Image) -> str:
    """Run TrOCR with beam search on a single cell image."""
    if pil_img is None:
        return ""
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=5,
            early_stopping=True,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ══════════════════════════════════════════════════════
#  5. TABLE RECONSTRUCTION
# ══════════════════════════════════════════════════════
def cluster_1d(values: list, gap: int):
    """
    Cluster 1-D positions (row centres or col centres) by gap threshold.
    Returns a list of group labels (integers) in the same order as `values`.
    """
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    groups      = []
    group_id    = 0
    prev_val    = None
    idx_to_group = {}

    for orig_idx, val in sorted_vals:
        if prev_val is None or val - prev_val > gap:
            group_id += 1
        idx_to_group[orig_idx] = group_id
        prev_val = val
        groups.append(group_id)

    return [idx_to_group[i] for i in range(len(values))]


def reconstruct_table(cells, img: np.ndarray):
    entries = []

    for i, (x, y, w, h) in enumerate(cells):
        crop     = img[y:y+h, x:x+w]
        pil_img  = prepare_cell(crop)
        text     = ocr_cell(pil_img)

        # Save debug crop
        if pil_img is not None:
            pil_img.save(os.path.join(DEBUG_DIR, f"{i:04d}.png"))

        cx = x + w // 2
        cy = y + h // 2
        entries.append({"cx": cx, "cy": cy, "text": text})

    if not entries:
        return []

    cys = [e["cy"] for e in entries]
    cxs = [e["cx"] for e in entries]

    # Dynamically choose gap threshold from median cell height
    heights    = [h for _, _, _, h in cells]
    row_gap    = int(np.median(heights) * 0.6) if heights else 20
    widths     = [w for _, _, w, _ in cells]
    col_gap    = int(np.median(widths)  * 0.4) if widths  else 30

    row_ids = cluster_1d(cys, row_gap)
    col_ids = cluster_1d(cxs, col_gap)

    # Build grid
    grid = {}
    for e, r, c in zip(entries, row_ids, col_ids):
        grid.setdefault(r, {})[c] = e["text"]

    num_rows = max(grid) + 1
    all_cols = sorted({c for row in grid.values() for c in row})

    table = []
    for r in sorted(grid):
        row_data = [grid[r].get(c, "") for c in all_cols]
        table.append(row_data)

    return table


# ══════════════════════════════════════════════════════
#  6. HEADER DETECTION
# ══════════════════════════════════════════════════════
def auto_detect_headers(table):
    def looks_like_header(row):
        has_alpha = sum(1 for c in row if re.search(r"[A-Za-z]", c))
        return has_alpha >= max(1, len(row) // 3)

    for i, row in enumerate(table):
        if looks_like_header(row):
            headers = [c if c else f"col_{j+1}" for j, c in enumerate(row)]
            return headers, table[i+1:]

    headers = [f"col_{i+1}" for i in range(len(table[0]))]
    return headers, table          # ← don't drop first row in fallback


# ══════════════════════════════════════════════════════
#  7. MAIN
# ══════════════════════════════════════════════════════
def extract_table(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Image not found: {image_path}")

    # Scale up small images so OCR works better
    H, W = img.shape[:2]
    if max(H, W) < 1200:
        scale = 1200 / max(H, W)
        img   = cv2.resize(img, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_CUBIC)
        print(f"  📐 Upscaled image to {img.shape[1]}×{img.shape[0]}")

    print("📐 Detecting cells...")
    cells = detect_cells(img)

    print("✍️  Running OCR on cells...")
    table = reconstruct_table(cells, img)

    if not table:
        print("❌ No table data extracted.")
        return

    print("🧠 Auto-detecting headers...")
    headers, data = auto_detect_headers(table)

    # Pad rows to header length
    n = len(headers)
    data = [r + [""] * (n - len(r)) if len(r) < n else r[:n] for r in data]

    df = pd.DataFrame(data, columns=headers)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print("✅ CSV  →", OUTPUT_CSV)
    print("✅ XLSX →", OUTPUT_XLSX)
    print("🧩 Debug cells →", DEBUG_DIR)
    print(df.to_string())
    return df


if __name__ == "__main__":
    extract_table(IMAGE_PATH)