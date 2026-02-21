# detection.py

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

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


def detect_rows_columns(image_path, threshold=0.4):

    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    h, w = img_cv.shape[:2]

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[(h, w)]
    )[0]

    rows = []
    columns = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        label_name = model.config.id2label[label.item()]
        box = box.cpu().numpy().astype(int)

        if label_name == "table row":
            rows.append(box)

        elif label_name == "table column":
            columns.append(box)

    # Sort rows top → bottom
    rows = sorted(rows, key=lambda x: x[1])

    # Sort columns left → right
    columns = sorted(columns, key=lambda x: x[0])

    print("Rows detected:", len(rows))
    print("Columns detected:", len(columns))
    print("Expected cells:", len(rows) * len(columns))

    return rows, columns