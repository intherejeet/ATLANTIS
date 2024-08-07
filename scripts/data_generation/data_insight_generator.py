import json
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import openai
import os

openai.api_type = "azure"
openai.api_key = "your_api_key_here"  # Replace with your actual API key
openai.api_base = "https://your_azure_endpoint_here/"  # Replace with your actual Azure endpoint
openai.api_version = '2023-08-01-preview'

def count_images_in_classes(root_dir, extensions=['.jpg', '.png', '.gif']):
    class_count = {}

    for class_dir in os.listdir(root_dir):
        full_class_path = os.path.join(root_dir, class_dir)
        
        if os.path.isdir(full_class_path):
            image_count = 0
            for file_name in os.listdir(full_class_path):
                if any(file_name.endswith(ext) for ext in extensions):
                    image_count += 1
            
            class_count[class_dir] = image_count

    min_class = min(class_count, key=class_count.get)
    max_class = max(class_count, key=class_count.get)
    avg_count = sum(class_count.values()) / len(class_count)
    
    sorted_counts = dict(sorted(class_count.items(), key=lambda item: item[1]))
    least_10_classes = dict(list(sorted_counts.items())[:10])

    return len(class_count), sum(class_count.values()), avg_count, min_class, class_count[min_class], max_class, class_count[max_class], least_10_classes

def tokenize_caption(caption):
    words = re.sub(r'[^a-zA-Z\s]', '', caption).lower().split()
    return [word for (word, pos) in pos_tag(words) if pos[:2] in ['NN', 'VB']]

def get_class_names(root_dir="path/to/cars196_reformatted"):  # Replace with your actual dataset path
    class_names = []

    for class_dir in os.listdir(root_dir):
        full_class_path = os.path.join(root_dir, class_dir)
        
        if os.path.isdir(full_class_path):
            class_names.append(class_dir)

    return class_names

def main():
    class_counter = Counter()
    domain_counter = Counter()

    # Load data
    input_file_path = "path/to/train_captions.json"  # Replace with your actual file path
    with open(input_file_path, "r") as infile:
        data = json.load(infile)

    # Process captions
    for key in tqdm(data.keys(), desc='Processing captions'):
        caption = data[key]
        class_id = key.split("/")[0].split(".")[1]
        class_counter[class_id] += 1
        
        relevant_words = tokenize_caption(caption)
        domain_counter.update(relevant_words)

    # Extract top stop words
    top_stop_words = domain_counter.most_common(100)

    # Formulate prompt for ChatGPT
    user_message_str = f"These are the top stop words with their occurrences: {top_stop_words}."

    response = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "You are a sophisticated AI model, trained to discern context and patterns within an image dataset. You are proficient in understanding stop words extracted from image descriptions or captions. Your expertise extends to deep metric learning and image retrieval datasets. Based on these descriptions, infer a renowned dataset to which this could be a subset of or bear similarity. Identify the common classification or discrimination task associated with the inferred dataset in the image retrieval domain. Based on this task information and input stop words, deduce key semantic domains that specifically represent the subject's location, action, background or scene in general, while also adding count their occurrences in bracket, to comprehend how the domain distribution varies across different classes."},
            {"role": "user", "content": user_message_str}
        ],
        deployment_id="gpt4"
    )

    root_directory = "path/to/CUB_200_2011/images"  # Replace with your actual dataset path
    num_classes, total, avg, min_class, min_count, max_class, max_count, least_10 = count_images_in_classes(root_directory)

    return {
        'num_classes': num_classes,
        'total': total,
        'avg': avg,
        'min_class': min_class,
        'min_count': min_count,
        'max_class': max_class,
        'max_count': max_count,
        'least_10': least_10,
        'class_names': list(class_counter.keys()),
        'data_summary': response["choices"][0]["message"]["content"]
    }

if __name__ == '__main__':
    results = main()
    print('\n\nData Class Distribution Information \n')
    print(f"Total classes: {results['num_classes']}")
    print(f"Total images: {results['total']}")
    print(f"Average images per class: {results['avg']}")
    print(f"Class with maximum images: {results['max_class']} ({results['max_count']} images)")
    print(f"Class with minimum images: {results['min_class']} ({results['min_count']} images)")
    print(f"10 classes with least images: {results['least_10']}")
    print("\n\nData Overview:", results['data_summary'])
