import openai
import json
from tqdm import tqdm
import os
import time
from ii_insight_generator import get_class_names

openai.api_type = "azure"
openai.api_key = "your_api_key_here"  # Replace with your actual API key
openai.api_base = "https://your_azure_endpoint_here/"  # Replace with your actual Azure endpoint
openai.api_version = '2023-08-01-preview'

# File paths
output_file_name = 'enhanced_captions_train.json'
output_file_path = "./captions/" + output_file_name

# Initialize existing enhanced captions
if os.path.exists(output_file_path):
    with open(output_file_path, "r") as infile:
        enhanced_captions = json.load(infile)
else:
    enhanced_captions = {}

def batch_data(data, batch_size):
    return (data[i:i + batch_size] for i in range(0, len(data), batch_size))

def get_bird_identity(key):
    try:
        bird_identity = key.split("/")[0].split(".")[1]
        return bird_identity.replace("_", " ")
    except IndexError:
        return None
    
def clean_captions(captions_dict):
    for key in captions_dict:
        # Check if the value is a string
        if isinstance(captions_dict[key], str):
            # Remove newlines, extra spaces, and escape characters from the JSON string
            cleaned_string = captions_dict[key].replace('\n', '').replace('  ', ' ').replace('\\"', '')

            try:
                # Try to parse the cleaned string into a list and replace the original value
                captions_dict[key] = json.loads(cleaned_string)
            except json.JSONDecodeError:
                # If there's an error, leave the value as is
                pass
    return captions_dict

class_names = get_class_names()
sorted_class_names = sorted(class_names, key=lambda x: int(x.split(".")[0]))

# Process in batches
for cl in tqdm(sorted_class_names[0:98]):
    try:
        user_messages = [f'The car is a {get_bird_identity(cl)}.']
        user_message_str = ", ".join(user_messages)
        
        response = openai.ChatCompletion.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a highly specialized image caption augmentation engine, designed for optimizing input prompts in text-to-image models with a focus on vehicle or car subjects. Your assignment is to create 60 domain-diverse, enhanced captions for each given car class name. Each caption should be formulated exclusively around the specified car class, without incorporating any other car in the scene. Craft your captions to depict varied yet plausible scenarios that would be valuable for generating a diverse range of images via data augmentation techniques. Note that each scenario must only feature a single, prominently visible car belonging to the class name in focus. Your output should maintain high grammatical standards, be clear, and concise. Strictly follow the following: your generated enhanced captions set must be in a list format of comma separated strings of captions without numbering, bullet points, or any additional markers like slashes \\ or newlines \\n. For instance, the output must be in the following format: ['caption1', 'caption2', ...]. Also the list must not have any external single or double quotes for example '[]', because it changes the format of the value.",
                },
                {"role": "user", "content": user_message_str},
            ],
            deployment_id="gpt4"
        )

        enhanced_captions_list = response["choices"][0]["message"]["content"]
        enhanced_captions[cl] = enhanced_captions_list
        enhanced_captions = clean_captions(enhanced_captions)
        
        with open(output_file_path, "w") as f:
            json.dump(enhanced_captions, f)
        # time.sleep(0.5)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

### cleaning generated json
def clean_and_convert_to_list(value):
    """
    Ensures that the value is a list of strings.
    If the value is not a list, it returns an empty list.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    return []

def clean_json_data(input_file, output_file):
    """
    Reads JSON data from input_file, cleans it, and writes to output_file.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    cleaned_data = {}
    for key, value in data.items():
        cleaned_data[key] = clean_and_convert_to_list(value)

    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

# Clean and convert the data
clean_json_data(output_file_path, './captions/cleaned_' + output_file_name)
