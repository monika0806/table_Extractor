import cv2
import numpy as np

IMAGE_PATH = "b.jpeg"
OUTPUT_IMAGE = "edge_based_grid.jpg"


def cluster(vals, thresh=15):
    vals = sorted(vals)
    result = []
    for v in vals:
        if not result or abs(v - result[-1]) > thresh:
            result.append(v)
    return result


def detect_grid(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Contrast enhancement ---
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)

    # --- Edge detection ---
    edges = cv2.Canny(gray, 40, 120)

    # --- Dilate edges slightly to connect gaps ---
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # --- Extract horizontal lines ---
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60,1))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

    # --- Extract vertical lines ---
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,60))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

    # --- Find horizontal contours ---
    contours_h, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_lines = []

    for cnt in contours_h:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100:
            y_lines.append(y)

    # --- Find vertical contours ---
    contours_v, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_lines = []

    for cnt in contours_v:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 100:
            x_lines.append(x)

    # Cluster lines
    x_lines = cluster(x_lines)
    y_lines = cluster(y_lines)

    print("Vertical lines:", x_lines)
    print("Horizontal lines:", y_lines)

    if len(x_lines) < 2 or len(y_lines) < 2:
        print("❌ Not enough lines detected")
        return

    cells = []

    for i in range(len(y_lines)-1):
        for j in range(len(x_lines)-1):
            x1 = x_lines[j]
            x2 = x_lines[j+1]
            y1 = y_lines[i]
            y2 = y_lines[i+1]

            w = x2 - x1
            h = y2 - y1

            if w > 30 and h > 30:
                cells.append((x1, y1, w, h))

    print("✅ Cells detected:", len(cells))

    for (x,y,w,h) in cells:
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imwrite(OUTPUT_IMAGE, output)
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    detect_grid(IMAGE_PATH)