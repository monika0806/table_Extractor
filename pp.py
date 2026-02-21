import cv2
import json
from paddleocr import PPStructure, save_structure_res

# Load PP-Structure engine
table_engine = PPStructure(
    show_log=False,
    use_gpu=False
)

# Read Image
img_path = "a.jpg"
img = cv2.imread(img_path)

# Run Table Structure Detection
result = table_engine(img)

# Save detected structure (cell boxes)
save_structure_res(result, "./output", img_path)

print("\nTable Structure Detected\n")

# Extract Text from Each Cell
table_data = []

for res in result:
    if res['type'] == 'table':
        html = res['res']['html']
        table_data.append(html)

# Convert to JSON
with open("table_data.json", "w") as f:
    json.dump(table_data, f, indent=4)

print("Cell Data saved to table_data.json")