import json
from collections import Counter
from tqdm import tqdm
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to request domain categorization from GPT for a batch of captions
def get_domain_categories(caption_batch):
    try:
        user_message_str = "Categorize the following contexts into bird-specific domains:\n" + "\n".join(caption_batch)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI trained to categorize context into general domains, specifically focused on bird species. Upon analyzing the captions, identify five key domains that are most prominent in the dataset. For each domain, provide the number of images associated with it. Exclude generic terms like 'animal' or 'bird.' Your output should be concise, listing only the domain names and their corresponding image counts."
                },
                {"role": "user", "content": user_message_str}
            ],
        )
        return response["choices"][0]["message"]["content"].split("\n")
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(caption_batch)

# Function to process the data in batches and update class and domain statistics
def process_data_in_batches(data, batch_size):
    class_counter = Counter()
    domain_counter = Counter()

    caption_keys = list(data.keys())
    for i in tqdm(range(0, len(caption_keys), batch_size), desc='Processing batches'):
        caption_batch_keys = caption_keys[i:i + batch_size]
        caption_batch_values = [data[k] for k in caption_batch_keys]

        # Update class imbalance statistics
        for key in caption_batch_keys:
            bird_identity = key.split("/")[0].split(".")[1]
            class_counter[bird_identity] += 1

        # Request domain categories from GPT for the batch
        domain_categories = get_domain_categories(caption_batch_values)
        if domain_categories:
            domain_counter.update(domain_categories)

    return class_counter, domain_counter

# Function to output the statistics
def output_statistics(class_counter, domain_counter):
    # Identify classes with extreme sample counts
    max_class = class_counter.most_common(1)[0]
    min_class = class_counter.most_common()[:-2:-1][0]
    bottom_classes = class_counter.most_common()[:-6:-1]  # Bottom 5 classes

    # Output the statistics
    print("\nClass imbalance:")
    print("Max:", max_class)
    print("Min:", min_class)
    print("Bottom 5:", bottom_classes)
    print("\nDomain imbalance:")
    print(domain_counter)

def main():
    input_file_path = "./captions/train_captions.json"

    # Load input JSON file
    with open(input_file_path, "r") as infile:
        data = json.load(infile)

    batch_size = 50  # Adjust based on token limitations

    # Process data and update counters
    class_counter, domain_counter = process_data_in_batches(data, batch_size)

    # Output the final statistics
    output_statistics(class_counter, domain_counter)

if __name__ == "__main__":
    main()
