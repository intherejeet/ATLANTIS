import sys
import torch
import torchvision.transforms as T
import os
from tap import Tap
from typing_extensions import Literal
import numpy as np
import PIL
from hyp_metric_allshot.helpers import get_emb, evaluate
from hyp_metric_allshot.proxy_anchor.dataset import CUBirds, SOP, Cars
from hyp_metric_allshot.proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyp_metric_allshot.model import init_model
from functools import partial
import pandas as pd

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Config(Tap):
    train_path: str = "path/to/train/data"  # Path to train data
    test_path: str = 'path/to/test/data'  # Path to testing data
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "Cars"  # dataset name
    emb: int = 128  # output embedding size
    hyp_c: float = 0.1  # hyperbolic c, "0" enables sphere mode
    model: str = "dino_vits16"  # model name dino_vits16 or vit_small_patch16_224 or ...
    resize: int = 256  # image resize
    crop: int = 224  # center crop after resize
    gpu: int = 0
    seed: int = 21
    checkpoint_path: str = "path/to/checkpoint.pth"  # Path to the saved best model checkpoint

if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    set_seeds(cfg.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    assert os.path.isfile(cfg.checkpoint_path), "Checkpoint file not found."

    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}

    model = init_model(cfg)
    model = model.cuda()
    
    # Load pre-trained model parameters
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint)

    model.eval()  # Set the model to evaluation mode

    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_list[cfg.ds],
        train_path=cfg.train_path,
        test_path=cfg.test_path,
        mean_std=mean_std,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    
    # Run evaluation
    rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

    # Print out evaluation metrics  
    recall_keys = [1, 2, 4, 8]  
    for i, k in enumerate(recall_keys):  
        print(f"Recall@{k}: {rh[i]*100:.2f}")  
  
    # Create a DataFrame with recall scores as a column.  
    df_recall = pd.DataFrame({'Scores': [rh[i]*100 for i in range(len(recall_keys))]}, index=recall_keys)  
      
    # Transpose the DataFrame to have recall scores in a column.  
    df_recall = df_recall.T  
      
    # Save the transposed DataFrame to an Excel file.  
    df_recall.to_excel('recall_scores.xlsx', index=False)
