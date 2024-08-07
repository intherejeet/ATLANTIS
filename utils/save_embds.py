from sklearn.cluster import KMeans
import numpy as np
import torch
from tap import Tap
from torchvision import datasets, transforms as T
import sys
sys.path.append('/path/to/robustretrieval/main_refined/hyp_metric')
import os
import shutil
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from hyp_metric.model import init_model


class Config(Tap):
    # data_path: str = "/path/to/datasets/cycle2/syn"
    data_path: str = "/path/to/datasets/cycle2/cleaned/threshold_1"
    bs: int = 900
    model: str = "vit_small_patch16_224"
    resize: int = 256
    crop: int = 224
    seed: int = 42
    embeddings_path: str = "./embeddings/synthetic_embeddings_delta_1.npy"
    labels_path: str = "./embeddings/synthetic_labels_delta_1.npy"


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
    # Assuming x and y are of shape [num_samples, num_features]
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



if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
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

    # Data transformations and loaders
    transform = T.Compose([
        T.Resize(cfg.resize),
        T.CenterCrop(cfg.crop),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ])

    # Original code line
    data = datasets.ImageFolder(cfg.data_path, transform=transform)

    # Adjust class_to_idx based on custom folder naming
    new_class_to_idx = {}
    for folder_name in os.listdir(cfg.data_path):
        if os.path.isdir(os.path.join(cfg.data_path, folder_name)):
            class_number, class_name = folder_name.split(".")
            new_class_to_idx[class_name] = int(class_number) - 1
    data.class_to_idx = new_class_to_idx

    # Loading model
    model = init_model(cfg).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./hyp_metric/model_parameters.pth"))
    model.eval()

    # Generate or load original embeddings and labels
    if os.path.exists(cfg.embeddings_path) and os.path.exists(cfg.labels_path):
        original_embeddings = np.load(cfg.embeddings_path, allow_pickle=True)
        original_labels = np.load(cfg.labels_path, allow_pickle=True)
    else:
        original_loader = torch.utils.data.DataLoader(data, batch_size=cfg.bs, shuffle=False)
        original_embeddings, original_labels = get_embeddings(model, original_loader, device)
        np.save(cfg.embeddings_path, original_embeddings)
        np.save(cfg.labels_path, original_labels)
