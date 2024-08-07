import json
from tqdm import tqdm
import os
from openai import AzureOpenAI
import time

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    api_key="your_api_key_here",  # Replace with your actual API key
    azure_endpoint='https://your_azure_endpoint_here/'  # Replace with your actual Azure endpoint
)

# Define dataset path: Make sure to define training data path only 
base_path = 'path/to/cars196_reformatted/images'  # Replace with your actual dataset path

output_json_path = "path/to/category_mapping.json"  # Replace with your desired output path

# List all car categories
car_categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

category_mapping = {}

# Use tqdm to show progress
for category in tqdm(car_categories, desc="Processing Categories"):
    class_number, class_name = category.split('.', 1)
        
    response = client.chat.completions.create(
        model="gpt4v",
        messages=[
            {
                "role": "system",
                "content": "Given the name of a vehicle, which includes the brand, model, and year, determine the type of vehicle. The only possible classifications are: Sedan, SUV-Crossover, or Sports Car. Even if the vehicle does not fall into these classes, choose the nearest class for it among these three. Provide only the class name in your response without any additional text. For example, Sedan or SUV-Crossover or Sports Car."
            },
            {
                "role": "user",
                "content": f"The vehicle name is {class_name}."
            }
        ],
        max_tokens=20,
    )
    answer = json.loads(response.model_dump_json(indent=2))['choices'][0]['message']['content'].strip("'").split("', '")[0]
    category_mapping[category] = answer

    with open(output_json_path, 'w') as json_file:
        json.dump(category_mapping, json_file)

    time.sleep(0.5)
