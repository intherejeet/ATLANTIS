import json
from collections import Counter, defaultdict
import re
from nltk.corpus import stopwords
from tqdm import tqdm

# Define the input file path
input_file_path = "./captions/train_captions.json"

# Load input JSON file
with open(input_file_path, "r") as infile:
    data = json.load(infile)

# Initialize counters and stop words
class_counter = Counter()
word_counter = Counter()
stop_words = set(stopwords.words('english'))
co_occurrence = defaultdict(Counter)

# Function to preprocess and tokenize captions
def tokenize_caption(caption):
    caption = re.sub(r'[^a-zA-Z\s]', '', caption).lower()
    return [word for word in caption.split() if word not in stop_words]

# Count occurrences and co-occurrences
for key, value in tqdm(data.items(), desc="Processing captions"):
    bird_identity = key.split("/")[0].split(".")[1]
    class_counter[bird_identity] += 1
    
    tokens = tokenize_caption(value)
    word_counter.update(tokens)

    for word in tokens:
        co_occurrence[word].update(tokens)

# Identify frequent domain features and cluster them
def cluster_words(co_occurrence, threshold=2):
    clusters = defaultdict(list)
    for word, neighbors in co_occurrence.items():
        for neighbor, count in neighbors.items():
            if word != neighbor and count > threshold:
                clusters[word].append(neighbor)
    return clusters

clusters = cluster_words(co_occurrence)
top_domain_features = [item[0] for item in word_counter.most_common(10)]

# Initialize domain counter
domain_counter = Counter()

# Count domain features considering clusters
for key, value in tqdm(data.items(), desc="Counting domain features"):
    tokens = tokenize_caption(value)
    domain_features = []
    for word in tokens:
        if word in top_domain_features:
            domain_features.append(word)
        elif word in clusters:
            domain_features.extend(clusters[word])
    domain_counter.update(domain_features)

# Identify classes with extreme sample counts
def get_class_statistics(class_counter):
    max_class = class_counter.most_common(1)[0]
    min_class = class_counter.most_common()[:-2:-1][0]
    bottom_classes = class_counter.most_common()[:-6:-1]
    return max_class, min_class, bottom_classes

max_class, min_class, bottom_classes = get_class_statistics(class_counter)

# Output the statistics
def output_statistics(max_class, min_class, bottom_classes, domain_counter):
    print("\nClass imbalance:")
    print("Max:", max_class)
    print("Min:", min_class)
    print("Bottom 5:", bottom_classes)
    print("\nDomain imbalance:")
    print(domain_counter)

output_statistics(max_class, min_class, bottom_classes, domain_counter)
