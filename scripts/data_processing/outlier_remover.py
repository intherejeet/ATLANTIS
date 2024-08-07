from sklearn.cluster import KMeans
import numpy as np
import torch
from tap import Tap
from torchvision import datasets, transforms as T
import sys
import os
import shutil
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from hyp_metric.model import init_model

sys.path.append('path/to/hyp_metric')  # Replace with your actual path

class Config(Tap):
    path_orig: str = "path/to/test_cars196_reformatted"  # Replace with your actual path
    path_syn: str = "path/to/syn_test"  # Replace with your actual path
    bs: int = 900
    model: str = "dino_vits16"
    resize: int = 256
    crop: int = 224
    threshold_factors: list = [1.5]
    seed: int = 42
    original_embeddings_path: str = "path/to/original_embeddings.npy"  # Replace with your actual path
    original_labels_path: str = "path/to/original_labels.npy"  # Replace with your actual path
    synthetic_embeddings_path: str = "path/to/synthetic_embeddings.npy"  # Replace with your actual path
    synthetic_labels_path: str = "path/to/synthetic_labels.npy"  # Replace with your actual path

def get_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    for x, y in tqdm(dataloader, desc='Generating embeddings'):
        x = x.to(device)
        with torch.no_grad():
            z = model(x)
        embeddings.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())
        
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

def dist_matrix(x, y):
    return euclidean_distances(x, y)

def calculate_class_centers(embeddings, labels):
    class_centers = {}
    unique_labels = set(labels)
    for label in unique_labels:
        class_samples = embeddings[labels == label]
        class_center = np.mean(class_samples, axis=0)
        class_centers[label] = class_center
    return class_centers

def calculate_mean_distances(embeddings, labels, class_centers):
    mean_distances = {}
    unique_labels = set(labels)
    for label in unique_labels:
        class_samples = embeddings[labels == label]
        distances = dist_matrix(class_samples, np.expand_dims(class_centers[label], axis=0))
        mean_distances[label] = np.mean(distances)
    return mean_distances

def detect_and_remove_outliers(threshold_factor, original_embeddings, original_labels, synthetic_embeddings, synthetic_labels, synthetic_data):
    original_class_centers = calculate_class_centers(original_embeddings, original_labels)
    original_mean_distances = calculate_mean_distances(original_embeddings, original_labels, original_class_centers)

    outlier_indices = []

    new_synthetic_folder = f'path/to/cleaned/syn_test_threshold_{threshold_factor}'  # Replace with your actual path

    if not os.path.exists(new_synthetic_folder):
        os.makedirs(new_synthetic_folder)

    for i, (sample, label) in enumerate(zip(synthetic_data.samples, synthetic_labels)):
        distance_to_center = dist_matrix(np.expand_dims(synthetic_embeddings[i], axis=0), 
                                         np.expand_dims(original_class_centers[label], axis=0))[0][0]

        if distance_to_center > original_mean_distances[label] * threshold_factor:
            outlier_indices.append(i)
        else:
            original_path = sample[0]
            filename = os.path.basename(original_path)
            class_folder = os.path.basename(os.path.dirname(original_path))
            new_folder_path = os.path.join(new_synthetic_folder, class_folder)

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            new_path = os.path.join(new_folder_path, filename)
            shutil.copy2(original_path, new_path)

if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize(cfg.resize),
        T.CenterCrop(cfg.crop),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ])

    original_data = datasets.ImageFolder(cfg.path_orig, transform=transform)
    synthetic_data = datasets.ImageFolder(cfg.path_syn, transform=transform)

    model = init_model(cfg).to(device)
    model.load_state_dict(torch.load("path/to/pretrained_params/model_dino_vits16_best_r1_case_cars196_reformated_seed_42.pth"))  # Replace with your actual path
    model.eval()

    if os.path.exists(cfg.original_embeddings_path) and os.path.exists(cfg.original_labels_path):
        original_embeddings = np.load(cfg.original_embeddings_path, allow_pickle=True)
        original_labels = np.load(cfg.original_labels_path, allow_pickle=True)
    else:
        original_loader = torch.utils.data.DataLoader(original_data, batch_size=cfg.bs, shuffle=False)
        original_embeddings, original_labels = get_embeddings(model, original_loader, device)
        np.save(cfg.original_embeddings_path, original_embeddings)
        np.save(cfg.original_labels_path, original_labels)

    if os.path.exists(cfg.synthetic_embeddings_path) and os.path.exists(cfg.synthetic_labels_path):
        synthetic_embeddings = np.load(cfg.synthetic_embeddings_path, allow_pickle=True)
        synthetic_labels = np.load(cfg.synthetic_labels_path, allow_pickle=True)
    else:
        synthetic_loader = torch.utils.data.DataLoader(synthetic_data, batch_size=cfg.bs, shuffle=False)
        synthetic_embeddings, synthetic_labels = get_embeddings(model, synthetic_loader, device)
        np.save(cfg.synthetic_embeddings_path, synthetic_embeddings)
        np.save(cfg.synthetic_labels_path, synthetic_labels)

    for threshold_factor in cfg.threshold_factors:
        detect_and_remove_outliers(threshold_factor, original_embeddings, original_labels, synthetic_embeddings, synthetic_labels, synthetic_data)
