import os
import random
import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange
import wandb
import multiprocessing
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

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you have more than one GPU

class Config(Tap):
    path: str = "/path/to/datasets/orig"  # Path to datasets
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "CUB"  # Dataset name
    num_samples: int = 3  # Samples per category in batch
    bs: int = 300  # Batch size per GPU
    lr: float = 3e-5  # Learning rate
    t: float = 0.2  # Cross-entropy temperature
    emb: int = 128  # Output embedding size
    freeze: int = 0  # Number of transformer blocks to freeze
    ep: int = 40  # Number of epochs
    hyp_c: float = 0.1  # Hyperbolic c, "0" enables sphere mode
    eval_ep: str = "r(5,100,5)"  # Epochs for evaluation
    model: str = "vit_small_patch16_224"  # Model name
    save_emb: bool = False  # Save embeddings after training
    emb_name: str = "emb"  # Filename for embeddings
    clip_r: float = 2.3  # Feature clipping radius
    resize: int = 256  # Image resize
    crop: int = 224  # Center crop after resize
    local_rank: int = 0  # Set automatically for distributed training
    gpu: int = 0
    project: str = 'hyp_metric_ours'

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

def initialize_wandb(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    if cfg.local_rank == 0:
        wandb.init(project=cfg.project, config=cfg.as_dict())

def initialize_distributed_world():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return world_size

def prepare_transforms(cfg):
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.RandomResizedCrop(cfg.crop, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ])

def prepare_dataloader(cfg, ds_train, sampler, world_size):
    return DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
    )

def main():
    cfg: Config = Config().parse_args()
    initialize_wandb(cfg)
    world_size = initialize_distributed_world()
    transform = prepare_transforms(cfg)

    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[cfg.ds]
    ds_train = ds_class(cfg.path, "train", transform)
    assert len(ds_train.ys) * cfg.num_samples >= cfg.bs * world_size
    sampler = UniqueClassSempler(ds_train.ys, cfg.num_samples, cfg.local_rank, world_size)
    dl_train = prepare_dataloader(cfg, ds_train, sampler, world_size)

    model = init_model(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    get_emb_f = partial(get_emb, model=model, ds=ds_class, path=cfg.path, mean_std=transform.transforms[-1].mean, world_size=world_size, resize=cfg.resize, crop=cfg.crop)
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

    cudnn.benchmark = True

    for ep in trange(cfg.ep):
        sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            y = y.view(len(y) // cfg.num_samples, cfg.num_samples)
            assert (y[:, 0] == y[:, -1]).all()
            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)

            x = x.cuda(non_blocking=True)
            z = model(x).view(len(x) // cfg.num_samples, cfg.num_samples, cfg.emb)
            if world_size > 1:
                with torch.no_grad():
                    all_z = [torch.zeros_like(z) for _ in range(world_size)]
                    torch.distributed.all_gather(all_z, z)
                all_z[cfg.local_rank] = z
                z = torch.cat(all_z)
            loss = 0
            for i in range(cfg.num_samples):
                for j in range(cfg.num_samples):
                    if i != j:
                        l, s = loss_f(z[:, i], z[:, j])
                        loss += l
                        stats_ep.append({**s, "loss": l.item()})

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            scaler.step(optimizer)
            scaler.update()

        if (ep + 1) in eval_ep:
            rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) in eval_ep:
                stats_ep.update({"recall": rh, "recall_b": rb})
            wandb.log({**stats_ep, "ep": ep})

    if cfg.save_emb:
        save_embeddings(cfg, get_emb_f)

def save_embeddings(cfg, get_emb_f):
    ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
    x, y = get_emb_f(ds_type=ds_type)
    x, y = x.float().cpu(), y.long().cpu()
    torch.save((x, y), os.path.join(cfg.path, f"{cfg.emb_name}_eval.pt"))

    x, y = get_emb_f(ds_type="train")
    x, y = x.float().cpu(), y.long().cpu()
    torch.save((x, y), os.path.join(cfg.path, f"{cfg.emb_name}_train.pt"))

if __name__ == "__main__":
    main()
