import json

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

# Define file paths
input_file_path = 'path/to/your/enhanced_captions_test.json'  # Replace with your actual input file path
output_file_path = 'path/to/your/cleaned_enhanced_captions_test.json'  # Replace with your actual output file path

# Clean and convert the data
clean_json_data(input_file_path, output_file_path)
