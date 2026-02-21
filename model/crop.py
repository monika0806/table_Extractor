# crop.py

import cv2
import os


def crop_by_intersection(image_path, rows, columns, output_dir="cells"):

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    count = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(columns):

            # Intersection coordinates
            x1 = max(row[0], col[0])
            y1 = max(row[1], col[1])
            x2 = min(row[2], col[2])
            y2 = min(row[3], col[3])

            if x2 <= x1 or y2 <= y1:
                continue

            cropped = img[y1:y2, x1:x2]

            save_path = os.path.join(output_dir, f"cell_r{i}_c{j}.jpg")
            cv2.imwrite(save_path, cropped)

            count += 1

    print(f"Total cells saved: {count}")