import json

def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a dictionary.
    
    Parameters:
    file_path (str): The path to the JSON file.
    
    Returns:
    dict: The JSON data as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def merge_dictionaries(dict1, dict2):
    """
    Merge two dictionaries into a single dictionary.
    If there are overlapping keys, values from dict2 will overwrite those from dict1.

    Parameters:
    dict1 (dict): The first dictionary
    dict2 (dict): The second dictionary

    Returns:
    dict: Merged dictionary
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict

def write_json_file(data, file_path):
    """
    Writes a dictionary to a JSON file.

    Parameters:
    data (dict): The data to write to the file.
    file_path (str): The path to the JSON file to be written.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    # Paths to the JSON files
    file_path1 = '/home/inderjeet/robustretrieval/data_generator/captions/cleaned_enhanced_captions_train.json'
    file_path2 = '/home/inderjeet/robustretrieval/data_generator/captions/cleaned_enhanced_captions_test.json'

    # Read the JSON files
    dict1 = read_json_file(file_path1)
    dict2 = read_json_file(file_path2)

    # Merge the dictionaries
    combined_dict = merge_dictionaries(dict1, dict2)

    # Write the combined dictionary to a new JSON file
    output_file_path = '/home/inderjeet/robustretrieval/data_generator/captions/combined_cleaned_enhanced_captions.json'
    write_json_file(combined_dict, output_file_path)

    print("Combined JSON file created at:", output_file_path)
