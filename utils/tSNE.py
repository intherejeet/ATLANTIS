from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

class Config(Tap):
    original_embeddings_path: str = "./embeddings/original_embeddings.npy"
    original_labels_path: str = "./embeddings/original_labels.npy"
    # synthetic_embeddings_path: str = "./embeddings/synthetic_embeddings.npy"
    # synthetic_labels_path: str = "./embeddings/synthetic_labels.npy"
    synthetic_embeddings_path: str = "./embeddings/synthetic_embeddings_delta_1.npy"
    synthetic_labels_path: str = "./embeddings/synthetic_labels_delta_1.npy"
    output_plot_path: str = "./visualization/tsne_overlay_delta_1.png"


def load_embeddings_and_labels(embedding_path, label_path):
    embeddings = np.load(embedding_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    return embeddings, labels


def filter_first_n_classes(embeddings, labels, n_classes=100):
    indices = np.isin(labels, np.arange(n_classes))
    return embeddings[indices], labels[indices]


def plot_tsne_overlay(original_embeddings, original_labels, synthetic_embeddings, synthetic_labels, title, output_plot_path):
    tsne = TSNE(n_components=2, random_state=0)
    
    all_embeddings = np.vstack([original_embeddings, synthetic_embeddings])
    all_labels = np.concatenate([original_labels, synthetic_labels])
    
    all_embeddings_2d = tsne.fit_transform(all_embeddings)
    
    original_embeddings_2d = all_embeddings_2d[:len(original_embeddings)]
    synthetic_embeddings_2d = all_embeddings_2d[len(original_embeddings):]
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(original_embeddings_2d[:, 0], original_embeddings_2d[:, 1], c=original_labels, cmap='tab10', s=10, marker='o', alpha=0.5, label="Original")
    plt.scatter(synthetic_embeddings_2d[:, 0], synthetic_embeddings_2d[:, 1], c=synthetic_labels, cmap='tab10', s=10, marker='x', alpha=0.5, label="Synthetic")
    
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    plt.savefig(output_plot_path, dpi=1200)


if __name__ == "__main__":
    cfg = Config().parse_args()
    
    original_embeddings, original_labels = load_embeddings_and_labels(cfg.original_embeddings_path, cfg.original_labels_path)
    synthetic_embeddings, synthetic_labels = load_embeddings_and_labels(cfg.synthetic_embeddings_path, cfg.synthetic_labels_path)
    
    original_embeddings, original_labels = filter_first_n_classes(original_embeddings, original_labels, 100)
    synthetic_embeddings, synthetic_labels = filter_first_n_classes(synthetic_embeddings, synthetic_labels, 100)
    
    plot_tsne_overlay(original_embeddings, original_labels, synthetic_embeddings, synthetic_labels, 't-SNE Overlay of Original and Cleaned Synthetic Data', cfg.output_plot_path)
