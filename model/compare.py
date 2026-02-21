import os
import re
import cv2
import torch
import pandas as pd
from PIL import Image
from openpyxl import Workbook
import pytesseract
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------- LOAD MODELS ----------------

print("Loading TrOCR...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trocr_model.to(device)
trocr_model.eval()

print("Loading PaddleOCR...")
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("Tesseract ready.")


# ---------------- OCR FUNCTIONS ----------------

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    return gray


def read_trocr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((pil_img.width * 2, pil_img.height * 2))

    pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)

    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def read_paddle(img):

    result = paddle_ocr.ocr(img)

    if not result:
        return ""

    # Case 1: Old format -> [[[box], (text, score)], ...]
    if isinstance(result[0], list):
        texts = []
        for line in result[0]:
            if isinstance(line, list) and len(line) > 1:
                texts.append(line[1][0])
        return " ".join(texts)

    # Case 2: New dict format
    if isinstance(result[0], dict):
        texts = []
        for line in result:
            if "rec_text" in line:
                texts.append(line["rec_text"])
        return " ".join(texts)

    return ""

def read_tesseract(img):
    img = preprocess(img)
    text = pytesseract.image_to_string(img, config='--psm 7')
    return text.strip()


# ---------------- TABLE RECONSTRUCTION ----------------

def extract_cells(folder="cells"):

    files = os.listdir(folder)
    pattern = r"cell_r(\d+)_c(\d+)\.jpg"
    cell_data = []

    for file in files:
        match = re.match(pattern, file)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            cell_data.append((row, col, file))

    cell_data.sort(key=lambda x: (x[0], x[1]))

    max_row = max(x[0] for x in cell_data)
    max_col = max(x[1] for x in cell_data)

    tables = {
        "trocr": [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)],
        "paddle": [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)],
        "tesseract": [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)],
    }

    for row, col, file in cell_data:
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        print(f"\nCell ({row},{col})")

        tables["trocr"][row][col] = read_trocr(img)
        print("TrOCR:", tables["trocr"][row][col])

        tables["paddle"][row][col] = read_paddle(img)
        print("Paddle:", tables["paddle"][row][col])

        tables["tesseract"][row][col] = read_tesseract(img)
        print("Tesseract:", tables["tesseract"][row][col])

    return tables


def export_table(table, filename):
    df = pd.DataFrame(table)
    df.to_excel(filename, index=False, header=False)
    print(f"Saved {filename}")


# ---------------- RUN ----------------

if __name__ == "__main__":

    tables = extract_cells("cells")

    export_table(tables["trocr"], "trocr_output.xlsx")
    export_table(tables["paddle"], "paddle_output.xlsx")
    export_table(tables["tesseract"], "tesseract_output.xlsx")