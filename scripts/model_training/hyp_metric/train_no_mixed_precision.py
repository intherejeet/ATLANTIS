import os
import random
import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from functools import partial
import numpy as np
import PIL
from tap import Tap
from typing_extensions import Literal
from sampler import UniqueClassSempler
from helpers import get_emb, evaluate
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyptorch.pmath import dist_matrix
from model import init_model
import wandb
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class Config(Tap):
    path: str = "/path/to/datasets/merged/threshold_1.5"  # Path to datasets
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "CUB"  # Dataset name
    num_samples: int = 21  # Samples per category in batch
    bs: int = 2100  # Batch size per GPU
    lr: float = 3e-5  # Learning rate
    t: float = 0.2  # Cross-entropy temperature
    emb: int = 128  # Output embedding size
    freeze: int = 0  # Number of transformer blocks to freeze
    ep: int = 50  # Number of epochs
    hyp_c: float = 0.1  # Hyperbolic c, "0" enables sphere mode
    eval_ep: str = "[10,20,30,40,50]"  # Epochs for evaluation
    model: str = "vit_small_patch16_224"  # Model name
    save_emb: bool = False  # Save embeddings after training
    emb_name: str = "emb"  # Filename for embeddings
    clip_r: float = 2.3  # Feature clipping radius
    resize: int = 256  # Image resize
    crop: int = 224  # Center crop after resize
    local_rank: int = 0  # Set automatically for distributed training

def contrastive_loss(x0, x1, tau, hyp_c):
    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
    }
    return loss, stats

def main():
    cfg: Config = Config().parse_args()
    wandb.init(project="hyp_metric", config=cfg.as_dict(), mode='dryrun')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiple_gpus = torch.cuda.device_count() > 1

    mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) if cfg.model.startswith("vit") else (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tr = T.Compose([
        T.RandomResizedCrop(cfg.crop, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ])

    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[cfg.ds]
    ds_train = ds_class(cfg.path, "train", train_tr)
    sampler = UniqueClassSempler(ds_train.ys, cfg.num_samples, 0, 1)
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )

    model = init_model(cfg).to(device)
    if multiple_gpus:
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_class,
        path=cfg.path,
        mean_std=mean_std,
        world_size=1,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

    for ep in trange(cfg.ep):
        sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            y = y.view(len(y) // cfg.num_samples, cfg.num_samples)
            assert (y[:, 0] == y[:, -1]).all()
            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)

            x = x.to(device, non_blocking=True)
            z = model(x).view(len(x) // cfg.num_samples, cfg.num_samples, cfg.emb)
            loss = 0
            for i in range(cfg.num_samples):
                for j in range(cfg.num_samples):
                    if i != j:
                        l, s = loss_f(z[:, i], z[:, j])
                        loss += l
                        stats_ep.append({**s, "loss": l.item()})

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

        if (ep + 1) in eval_ep:
            rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

        stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
        if (ep + 1) in eval_ep:
            stats_ep = {"recall": rh, "recall_b": rb, **stats_ep}
        wandb.log({**stats_ep, "ep": ep})

    torch.save(model.state_dict(), "./model_parameters.pth")

    if cfg.save_emb:
        ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
        x, y = get_emb_f(ds_type=ds_type)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), os.path.join(cfg.path, f"{cfg.emb_name}_eval.pt"))

        x, y = get_emb_f(ds_type="train")
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), os.path.join(cfg.path, f"{cfg.emb_name}_train.pt"))

if __name__ == "__main__":
    main()
