import json
import os
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("ByteDance/SDXL-Lightning")

input_captions = 'path/to/cleaned_enhanced_captions_train.json'  # Replace with your actual JSON file path

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp32")
pipe.to("cuda")

def process_image(caption, folder_name, image_index):
    try:
        image = pipe(prompt=caption).images[0]
        # Save image
        image_name = f"{image_index}.jpg"
        image_path = os.path.join(folder_name, image_name)
        image.save(image_path)
    except Exception as e:
        print(f"Error processing {folder_name}/{image_name}: {e}")

# Create data directory if not exists
data_folder = 'path/to/syn_train'  # Replace with your actual data folder path
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# Import captions from JSON file
with open(input_captions, 'r') as f:
    captions_dict = json.load(f)

for folder_name, captions in captions_dict.items():
    parent_dir = os.path.join(data_folder, folder_name)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    for i, caption in enumerate(captions):
        process_image(caption, parent_dir, i)
