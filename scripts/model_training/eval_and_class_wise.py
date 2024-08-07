import sys
import random
from functools import partial
from tap import Tap
from typing_extensions import Literal
from hyp_metric.helpers import (
    get_emb,
    get_recall,
    get_class_wise_recall,
    get_recall_inshop,
)
from hyp_metric.proxy_anchor.dataset import CUBirds, SOP, Cars
from hyp_metric.proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyp_metric.model import init_model
from torch.nn.parallel import DistributedDataParallel
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

random.seed(42)

class Config(Tap):
    path: str = "path/to/dataset"  # path to dataset
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "CUB"  # dataset name
    model: str = (
        "dino_vits16"  # model name (see train.py), dino_vits16, vit_small_patch16_224
    )
    resize: int = 256  # image resize
    crop: int = 224  # center crop after resize
    class_wise_stats: bool = True
    visualization: bool = False
    params: str = "./saved_params/improved_model.pth"

def get_min_recall_scores(recall_dict, num_classes=20):
    # Sort the recall dictionary by values
    sorted_recall_dict = sorted(recall_dict.items(), key=lambda item: item[1])

    # Get the first 'num_classes' items from the sorted dictionary
    min_recall_classes = sorted_recall_dict[:num_classes]

    return min_recall_classes

if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    model = init_model(cfg)
    model.load_state_dict(torch.load(cfg.params))
    model.eval()

    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_list[cfg.ds],
        path=cfg.path,
        mean_std=mean_std,
        world_size=1,
        resize=cfg.resize,
        crop=cfg.crop,
        skip_head=False,
    )

    if cfg.ds != "Inshop":
        emb = get_emb_f(ds_type="eval")
        get_recall(*emb, cfg.ds, 0)
        if cfg.class_wise_stats:
            recall_dict = get_class_wise_recall(*emb, cfg.ds, 0)
            min_recall_classes = get_min_recall_scores(recall_dict)
            # Print the classes with minimum recall scores
            for class_label, recall_score in min_recall_classes:
                print(f"Class: {class_label}, Recall Score: {recall_score}")
    else:
        emb_query = get_emb_f(ds_type="query")
        emb_gal = get_emb_f(ds_type="gallery")
        get_recall_inshop(*emb_query, *emb_gal, 0)

    if cfg.class_wise_stats and cfg.visualization:
        # Create a 2D NumPy array from your dictionary
        k_values = [1]
        # Get the number of classes and k-values
        num_classes = len(recall_dict)
        num_k_values = len(next(iter(recall_dict.values())))

        # Prepare the data for the heatmap
        heatmap_data = np.zeros((num_classes, num_k_values))
        class_labels = list(recall_dict.keys())

        for i, (class_label, recalls) in enumerate(recall_dict.items()):
            heatmap_data[i, :] = recalls

        # heatmap generation for class-wise recall@k statistics
        plt.figure(figsize=(14, 26), dpi=300)
        sns.set(font_scale=1.4, style="whitegrid")  # Set style and font scale

        cmap = sns.cm.rocket_r  # High contrast color map
        ax = sns.heatmap(
            heatmap_data,
            annot=False,
            cmap=cmap,
            xticklabels=k_values,
            yticklabels=True,
            cbar_kws={
                "label": "Recall Score",
                "shrink": 0.5,
                "ticks": [0, 0.25, 0.5, 0.75, 1],
            },
            linewidths=1,
            linecolor="gray",
        )

        # Labels and title
        plt.xlabel("k-values", fontsize=18)
        plt.ylabel("Class Labels", fontsize=18)
        plt.title("Class-wise Recall for Different k-values", fontsize=20)

        # Filter to display only every 5th y-tick label and add grid
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::5])
        ax.set_yticklabels(class_labels[::5])

        # Add minor ticks at every class label to improve readability
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25)

        # Show the plot
        plt.tight_layout()

        plt.savefig(f"./test.png", dpi=600, format="png")
