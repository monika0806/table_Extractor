import cv2
import numpy as np

IMAGE_PATH = "d.jpeg"
OUTPUT_IMAGE = "traced_cells.jpg"


def cluster(values, thresh=15):
    values = sorted(values)
    result = []
    for v in values:
        if not result or abs(v - result[-1]) > thresh:
            result.append(v)
    return result


def detect_table_by_line_tracing(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 1. Edge detection ----
    edges = cv2.Canny(gray, 50, 150)

    # ---- 2. Hough line detection ----
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=80,
        maxLineGap=20
    )

    if lines is None:
        print("No lines detected")
        return

    horizontal = []
    vertical = []

    # ---- 3. Separate horizontal & vertical ----
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if abs(y1 - y2) < 10:
            horizontal.append((y1, x1, x2))

        elif abs(x1 - x2) < 10:
            vertical.append((x1, y1, y2))

    # ---- 4. Get unique line positions ----
    y_lines = cluster([y for y, _, _ in horizontal])
    x_lines = cluster([x for x, _, _ in vertical])

    print("Horizontal lines:", y_lines)
    print("Vertical lines:", x_lines)

    if len(x_lines) < 2 or len(y_lines) < 2:
        print("Not enough lines")
        return

    # ---- 5. Build cells from intersections ----
    cells = []

    for i in range(len(y_lines)-1):
        for j in range(len(x_lines)-1):

            x1 = x_lines[j]
            x2 = x_lines[j+1]
            y1 = y_lines[i]
            y2 = y_lines[i+1]

            w = x2 - x1
            h = y2 - y1

            if w > 20 and h > 20:
                cells.append((x1, y1, w, h))

    print("Cells detected:", len(cells))

    # ---- 6. Draw final cell borders ----
    for (x, y, w, h) in cells:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(OUTPUT_IMAGE, output)
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    detect_table_by_line_tracing(IMAGE_PATH)