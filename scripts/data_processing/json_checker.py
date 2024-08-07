import json

file_path = 'path/to/your/json_file.json'  # Replace with your actual JSON file path

with open(file_path, 'r') as f:
    data = json.load(f)
    num_keys = len(data.keys())
    print(f"Number of keys: {num_keys}")

    # If you want to iterate over each key-value pair and print the number of values in each list:
    # for key, value in data.items():
    #     num_values = len(value) if isinstance(value, list) else 0
    #     print(f"Key: {key}, Number of values: {num_values}")
