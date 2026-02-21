import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

IMAGE_PATH = "h.jpg"
OUTPUT_IMAGE = "final_cells_from_model.jpg"

# ---------------- LOAD MODEL ----------------
processor = DetrImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def detect_cells(image_path):

    image = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    h, w = img_cv.shape[:2]

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=0.7,
        target_sizes=[(h, w)]
    )[0]

    rows = []
    cols = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):

        label_name = model.config.id2label[label.item()]
        box = box.cpu().numpy().astype(int)

        if label_name == "table row":
            rows.append(box)

        if label_name == "table column":
            cols.append(box)

    print("Rows detected:", len(rows))
    print("Columns detected:", len(cols))

    if len(rows) == 0 or len(cols) == 0:
        print("No rows/columns found.")
        return

    # Sort rows top to bottom
    rows = sorted(rows, key=lambda b: b[1])
    # Sort columns left to right
    cols = sorted(cols, key=lambda b: b[0])

    output = img_cv.copy()

    # Compute intersections
    cell_count = 0
    for r in rows:
        for c in cols:

            x1 = max(r[0], c[0])
            y1 = max(r[1], c[1])
            x2 = min(r[2], c[2])
            y2 = min(r[3], c[3])

            if x2 > x1 and y2 > y1:
                cell_count += 1
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("Cells reconstructed:", cell_count)

    cv2.imwrite(OUTPUT_IMAGE, output)
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    detect_cells(IMAGE_PATH)