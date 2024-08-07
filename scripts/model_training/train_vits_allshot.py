import sys
import random
import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
import os
from tqdm import trange
import wandb
import multiprocessing
from functools import partial
import numpy as np
import PIL
from tap import Tap
from typing_extensions import Literal
import seaborn as sns
import matplotlib.pyplot as plt
from hyp_metric_allshot.sampler_class_imbalance import UniqueClassSempler
from hyp_metric_allshot.helpers import get_emb, evaluate
from hyp_metric_allshot.proxy_anchor.dataset import CUBirds, SOP, Cars
from hyp_metric_allshot.proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyp_metric_allshot.hyptorch.pmath import dist_matrix
from hyp_metric_allshot.model import init_model
import pandas as pd

sys.path.append("path/to/hyp_metric_allshot")  # Replace with your actual path

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you have more than one GPU

# flake8: noqa: E501
class Config(Tap):
    train_path: str = "path/to/train_data"  # Replace with your actual training data path
    test_path: str = "path/to/test_data"  # Replace with your actual testing data path
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "Cars"  # dataset name
    num_samples: int = 10  # how many samples per each category in batch
    bs: int = 500  # batch size per GPU, e.g. --num_samples 3 --bs 900 means each iteration we sample 300 categories with 3 samples
    lr: float = 3e-5  # learning rate
    t: float = 0.2  # cross-entropy temperature
    emb: int = 128  # output embedding size
    freeze: int = 0  # number of blocks in transformer to freeze, None - freeze nothing, 0 - freeze only patch_embed
    ep: int = 300  # number of epochs
    hyp_c: float = 0.1  # hyperbolic c, "0" enables sphere mode
    eval_ep: str = "r(30,500,5)"  # epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""
    model: str = "vit_small_patch16_224"  # model name from timm or torch.hub, i.e. deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16
    save_emb: bool = False  # save embeddings of the dataset after training
    emb_name: str = "emb"  # filename for embeddings
    clip_r: float = 2.3  # feature clipping radius
    resize: int = 256  # image resize
    crop: int = 224  # center crop after resize
    local_rank: int = 0  # set automatically for distributed training
    gpu: int = 0
    seed: int = 21
    project: str = "class_imbalance_cars"
    run_name: str = "vit_orig_class_100_seed_21"

def _init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)

def contrastive_loss(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

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

if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    # Set all the seeds for reproducibility
    set_seeds(cfg.seed)

    def _init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    if cfg.local_rank == 0:
        wandb.init(project=cfg.project, config=cfg.as_dict(), name=cfg.run_name)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()

    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tr = T.Compose(
        [
            T.RandomResizedCrop(
                cfg.crop, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )

    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[cfg.ds]
    ds_train = ds_class(cfg.train_path, "train", train_tr)
    assert len(ds_train.ys) * cfg.num_samples >= cfg.bs * world_size
    sampler = UniqueClassSempler(
        ds_train.ys, cfg.num_samples, cfg.local_rank, world_size
    )
    # sampler = RandomSampler(ds_train)

    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_init_fn,
    )

    model = init_model(cfg)
    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_class,
        train_path=cfg.train_path,  # path to training data
        test_path=cfg.test_path,    # path to testing data
        mean_std=mean_std,
        world_size=world_size,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

    cudnn.benchmark = True

    best_recall_at_1 = 0.0  # Step 1

    for ep in trange(cfg.ep):
        # sampler.set_epoch(ep)
        stats_ep = []
        ## original training loop
        for x, y in dl_train:
            y = y.view(len(y) // cfg.num_samples, cfg.num_samples)

            # assert (y[:, 0] == y[:, -1]).all() # since class-imbalance case have only 2 samples for some classes

            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)
            x = x.cuda(non_blocking=True)
            z = model(x).view(len(x) // cfg.num_samples, cfg.num_samples, cfg.emb)
            loss = 0
            optimizer.zero_grad()
            with autocast():  # Native AMP context manager
                for i in range(cfg.num_samples):
                    for j in range(cfg.num_samples):
                        if i != j:
                            l, s = loss_f(z[:, i], z[:, j])
                            loss += l
                            stats_ep.append({**s, "loss": l.item()})

            # Backward and optimizer step using scaler
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            scaler.step(optimizer)
            scaler.update()

        if (ep + 1) in eval_ep:
            rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) in eval_ep:
                # Create the recall keys and values
                recall_keys = [1, 2, 4, 8]
                new_stats_ep = {f"recall@{k}": rh[i] for i, k in enumerate(recall_keys)}
                new_stats_ep.update(
                    {f"recall_b@{k}": rb[i] for i, k in enumerate(recall_keys)}
                )
                # Merge new_stats_ep into stats_ep
                stats_ep = {**new_stats_ep, **stats_ep}
                current_recall_at_1 = rh[0]
                if current_recall_at_1 > best_recall_at_1:
                    best_recall_at_1 = current_recall_at_1
                    best_epoch = ep
                    recall_scores_at_best_r1 = {f"recall@{k}": rh[i] for i, k in enumerate(recall_keys)}

            wandb.log({**stats_ep, "ep": ep})

    if cfg.local_rank == 0:
        print(f"Best Recall@1 (epoch {best_epoch}): {best_recall_at_1*100:.2f}")

        # Prepare data for DataFrame
        data = {}
        data.update({f"{k}": [v*100] for k, v in recall_scores_at_best_r1.items()})

        # Create DataFrame
        df = pd.DataFrame(data)

        # Print the recall scores
        for k, v in recall_scores_at_best_r1.items():
            print(f"{k}: {v*100:.2f}")

        # Save DataFrame to an Excel file
        excel_file = "best_recall_scores.xlsx"
        df.to_excel(excel_file, index=False)

        # Log the Excel file to wandb
        wandb.save(excel_file)

    if cfg.save_emb:
        ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
        x, y = get_emb_f(ds_type=ds_type)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_eval.pt")

        x, y = get_emb_f(ds_type="train")
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_train.pt")
