# app.py

from detection import detect_rows_columns
from crop import crop_by_intersection

IMAGE_PATH = "b.jpeg"


def main():

    print("\nSTEP 1: Detecting rows & columns...")
    rows, columns = detect_rows_columns(IMAGE_PATH, threshold=0.3)

    if len(rows) == 0 or len(columns) == 0:
        print("Detection failed.")
        return

    print("\nSTEP 2: Cropping using intersection...")
    crop_by_intersection(IMAGE_PATH, rows, columns)


if __name__ == "__main__":
    main()