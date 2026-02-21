import cv2
import numpy as np

IMAGE_PATH = "b.jpeg"
OUTPUT_IMAGE = "projection_grid.jpg"


def find_peaks(projection, threshold_ratio=0.5, min_gap=20):
    threshold = max(projection) * threshold_ratio
    peaks = []
    for i, val in enumerate(projection):
        if val > threshold:
            if not peaks or abs(i - peaks[-1]) > min_gap:
                peaks.append(i)
    return peaks


def detect_grid_projection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Strong binary
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Horizontal projection (sum of white pixels per row)
    horizontal_projection = np.sum(th, axis=1)

    # Vertical projection (sum of white pixels per column)
    vertical_projection = np.sum(th, axis=0)

    # Detect line positions
    y_lines = find_peaks(horizontal_projection, threshold_ratio=0.6)
    x_lines = find_peaks(vertical_projection, threshold_ratio=0.6)

    print("Horizontal lines:", y_lines)
    print("Vertical lines:", x_lines)

    if len(x_lines) < 2 or len(y_lines) < 2:
        print("❌ Not enough lines detected")
        return

    cells = []

    for i in range(len(y_lines) - 1):
        for j in range(len(x_lines) - 1):
            x1 = x_lines[j]
            x2 = x_lines[j + 1]
            y1 = y_lines[i]
            y2 = y_lines[i + 1]

            w = x2 - x1
            h = y2 - y1

            if w > 20 and h > 20:
                cells.append((x1, y1, w, h))

    print("✅ Cells detected:", len(cells))

    for (x, y, w, h) in cells:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(OUTPUT_IMAGE, output)
    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    detect_grid_projection(IMAGE_PATH)