import torch
import pandas as pd
import io
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------- CONFIG ----------------
IMAGE_PATH = "e.jpeg"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Detect hardware (Apple Silicon MPS, Nvidia CUDA, or standard CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Loading local Vision Model on {device.upper()} (this may take a moment)...")

# ---------------- LOAD MODEL ----------------
# We use torch_dtype="auto" to load it efficiently based on your hardware
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map=device
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------- EXTRACT TABLE ----------------
def extract_table(image_path):
    print("Analyzing image...")
    
    # We construct a prompt telling the model exactly what we want
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text", 
                    "text": "Extract the handwritten table from this image. Ignore the lined notebook paper, date, and signature. Return ONLY a valid CSV format representing the table contents. Do not include markdown formatting or extra text."
                },
            ],
        }
    ]

    # Prepare inputs for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        
    # Isolate just the generated text (ignoring the prompt tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Convert the raw CSV string directly into a Pandas DataFrame
    try:
        csv_data = io.StringIO(output_text)
        df = pd.read_csv(csv_data)
        print("\nFinal Digitized Table:")
        print(df)
        return df
    except Exception as e:
        print("\nFailed to parse as CSV. Here is the raw model output:")
        print(output_text)

if __name__ == "__main__":
    extract_table(IMAGE_PATH)