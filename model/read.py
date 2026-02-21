import os
import re
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from openpyxl import Workbook
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------- LOAD MODEL ----------------

print("Loading TrOCR model...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded.\n")


# ---------------- BLANK CELL CHECK ----------------

def is_blank_cell(img, threshold_ratio=0.01):
    """
    Detect if cell is mostly blank.
    Prevents TrOCR from hallucinating text like '0 0'.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    total_pixels = gray.shape[0] * gray.shape[1]
    dark_pixels = total_pixels - cv2.countNonZero(gray)

    ratio = dark_pixels / total_pixels

    return ratio < threshold_ratio


# ---------------- TrOCR FUNCTION ----------------

def read_with_trocr(img):

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Resize to improve small handwriting reading
    pil_img = pil_img.resize(
        (pil_img.width * 2, pil_img.height * 2),
        Image.BICUBIC
    )

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


# ---------------- TABLE EXTRACTION ----------------

def extract_table_from_cells(folder="cells"):

    if not os.path.exists(folder):
        print("Cells folder not found.")
        return []

    files = os.listdir(folder)

    pattern = r"cell_r(\d+)_c(\d+)\.jpg"
    cell_data = []

    for file in files:
        match = re.match(pattern, file)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            cell_data.append((row, col, file))

    if not cell_data:
        print("No cell images found.")
        return []

    # Sort row-wise
    cell_data.sort(key=lambda x: (x[0], x[1]))

    max_row = max(x[0] for x in cell_data)
    max_col = max(x[1] for x in cell_data)

    table = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    print("Reading cells using TrOCR...\n")

    for row, col, file in cell_data:
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        # Blank detection
        if is_blank_cell(img):
            text = ""
        else:
            text = read_with_trocr(img)

        table[row][col] = text

        print(f"Cell ({row},{col}) -> '{text}'")

    return table


# ---------------- EXPORT FUNCTIONS ----------------

def export_to_csv(table, filename="table_output.csv"):
    if not table:
        print("Empty table. CSV not created.")
        return

    df = pd.DataFrame(table)
    df.to_csv(filename, index=False, header=False)
    print(f"\nCSV saved as {filename}")


def export_to_excel(table, filename="table_output.xlsx"):
    if not table:
        print("Empty table. Excel not created.")
        return

    wb = Workbook()
    ws = wb.active

    for i, row in enumerate(table):
        for j, value in enumerate(row):
            ws.cell(row=i+1, column=j+1, value=value)

    wb.save(filename)
    print(f"Excel saved as {filename}")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    table = extract_table_from_cells("cells")

    export_to_csv(table)
    export_to_excel(table)