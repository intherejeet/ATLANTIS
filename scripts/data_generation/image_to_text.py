import os
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

# Define dataset path: Make sure to define training data path only 
base_path = 'path/to/cars196_reformatted'  # Replace with your actual dataset path

# List all car categories
car_categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Dictionary for storing image paths and captions
image_captions = {}

# Use tqdm to show progress
for category in tqdm(car_categories, desc="Processing Categories"):
    
    # Check if category belongs to the first 98 classes
    class_number = int(category.split('.')[0])
    # if class_number > 98:
    #     continue

    category_path = os.path.join(base_path, category)

    # List all images in this category
    images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]

    for img_name in images:
        image_path = os.path.join(category_path, img_name)
        relative_path = os.path.join(category, img_name)

        raw_image = Image.open(image_path).convert('RGB')

        # Perform unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=15)

        caption = processor.decode(out[0], skip_special_tokens=True)
        image_captions[relative_path] = caption

# Save captions to a JSON file
output_json_path = 'path/to/save/train_captions.json'  # Replace with your desired output path
with open(output_json_path, 'w') as f:
    json.dump(image_captions, f)

print("Captions generation completed!")
