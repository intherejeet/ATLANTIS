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
from hyp_metric.sampler import UniqueClassSempler
from hyp_metric.helpers import get_emb, evaluate, evaluate_class_wise
from hyp_metric.proxy_anchor.dataset import CUBirds, SOP, Cars
from hyp_metric.proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyp_metric.hyptorch.pmath import dist_matrix
from hyp_metric.model import init_model
from torchvision.datasets import ImageFolder
import pandas as pd

sys.path.append("path/to/hyp_metric")  # Replace with your actual path

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you have more than one GPU

class Config(Tap):
    path: str = "path/to/cars196_reformatted"  # Replace with your actual dataset path
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "Cars"  # dataset name
    num_samples: int = 11  # how many samples per each category in batch
    bs: int = 1078  # batch size per GPU, e.g. --num_samples 3 --bs 900 means each iteration we sample 300 categories with 3 samples
    lr: float = 3e-5  # learning rate
    t: float = 0.2  # cross-entropy temperature
    emb: int = 128  # output embedding size
    freeze: int = 0  # number of blocks in transformer to freeze, None - freeze nothing, 0 - freeze only patch_embed
    ep: int = 300  # number of epochs
    hyp_c: float = 0.1  # hyperbolic c, "0" enables sphere mode
    eval_ep: str = "r(0,1000,5)"  # epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""
    model: str = "dino_vits16"  # model name from timm or torch.hub, i.e. deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16
    save_emb: bool = False  # save embeddings of the dataset after training
    emb_name: str = "emb"  # filename for embeddings
    clip_r: float = 2.3  # feature clipping radius
    resize: int = 256  # image resize
    crop: int = 224  # center crop after resize
    local_rank: int = 0  # set automatically for distributed training
    gpu: int = 0
    seed: int = 42
    project: str = "car_data_experiments"
    run_name: str = "default"
    log_class_wise: bool = False

def _init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)

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

if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    set_seeds(cfg.seed)

    # Calculating number of training classes and setting batch size
    all_class_labels = [i[1] for i in ImageFolder(root=os.path.join(cfg.path, 'images')).imgs]
    total_num_classes = max(all_class_labels) + 1  # Assuming labels are 0-indexed
    test_num_classes = 98  # fixed as per your condition
    train_num_classes = total_num_classes - test_num_classes

    print(f'Number of train classes: {train_num_classes}')
    print(f'Batch size: {cfg.bs}')

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
    ds_train = ds_class(cfg.path, "train", train_tr)
    assert len(ds_train.ys) * cfg.num_samples >= cfg.bs * world_size
    sampler = UniqueClassSempler(
        ds_train.ys, cfg.num_samples, cfg.local_rank, world_size
    )
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
        path=cfg.path,
        mean_std=mean_std,
        world_size=world_size,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

    cudnn.benchmark = True

    best_recall_at_1 = 0.0  # Step 1
    recall_keys = [1, 2, 4, 8]
    recall_scores_at_best_r1 = {}

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
            if cfg.log_class_wise:
                rh, rb, class_wise_rh, train_class_wise_rh = evaluate_class_wise(
                    get_emb_f, cfg.ds, cfg.hyp_c
                )
            else:
                rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

            current_recall_at_1 = rh[0]  # Recall@1
            if current_recall_at_1 > best_recall_at_1:
                best_recall_at_1 = current_recall_at_1
                best_epoch = ep
                recall_scores_at_best_r1 = {f"recall@{k}": rh[i] for i, k in enumerate(recall_keys)}

                model_save_path = f'./trained_parameters/model_{cfg.model}_best_r1_case_{cfg.path.split("/")[-1]}_seed_{cfg.seed}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"Model saved with Recall@1: {current_recall_at_1:.2f} at epoch {ep + 1}"
                )

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) in eval_ep:
                new_stats_ep = {f"recall@{k}": rh[i] for i, k in enumerate(recall_keys)}
                new_stats_ep.update(
                    {f"recall_b@{k}": rb[i] for i, k in enumerate(recall_keys)}
                )
                stats_ep = {**new_stats_ep, **stats_ep}
                current_recall_at_1 = rh[0]

                if cfg.log_class_wise:
                    if current_recall_at_1 > best_recall_at_1:
                        k_values = [1, 2, 4, 8]
                        num_classes = len(train_class_wise_rh)
                        num_k_values = len(next(iter(train_class_wise_rh.values())))

                        heatmap_data = np.zeros((num_classes, num_k_values))
                        class_labels = list(train_class_wise_rh.keys())

                        for i, (class_label, recalls) in enumerate(
                            train_class_wise_rh.items()
                        ):
                            heatmap_data[i, :] = recalls

                        plt.figure(figsize=(14, 26), dpi=300)
                        sns.set(
                            font_scale=1.4, style="whitegrid"
                        )

                        cmap = sns.cm.rocket_r
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

                        plt.xlabel("k-values", fontsize=18)
                        plt.ylabel("Class Labels", fontsize=18)
                        plt.title(
                            "Train Class-wise Recall for Different k-values",
                            fontsize=20,
                        )

                        yticks = ax.get_yticks()
                        ax.set_yticks(yticks[::5])
                        ax.set_yticklabels(class_labels[::5])

                        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
                        ax.grid(
                            which="minor", color="gray", linestyle="-", linewidth=0.25
                        )

                        plt.tight_layout()
                        training_heatmap_filename = f'./visualization/class_wise_recall/train_heatmap_model_{cfg.model}_case_{cfg.path.split("/")[-1]}_seed_{cfg.seed}.png'
                        plt.savefig(training_heatmap_filename, dpi=200, format="png")
                        plt.close()

                        wandb.log(
                            {
                                "Train Class-wise Recall Heatmap": wandb.Image(
                                    training_heatmap_filename
                                ),
                                "ep": ep,
                            }
                        )

                        num_classes = len(class_wise_rh)
                        num_k_values = len(next(iter(class_wise_rh.values())))

                        heatmap_data = np.zeros((num_classes, num_k_values))
                        class_labels = list(class_wise_rh.keys())

                        for i, (class_label, recalls) in enumerate(
                            class_wise_rh.items()
                        ):
                            heatmap_data[i, :] = recalls

                        plt.figure(figsize=(14, 26), dpi=300)
                        sns.set(
                            font_scale=1.4, style="whitegrid"
                        )

                        cmap = sns.cm.rocket_r
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

                        plt.xlabel("k-values", fontsize=18)
                        plt.ylabel("Class Labels", fontsize=18)
                        plt.title(
                            "Test Class-wise Recall for Different k-values", fontsize=20
                        )

                        yticks = ax.get_yticks()
                        ax.set_yticks(yticks[::5])
                        ax.set_yticklabels(class_labels[::5])

                        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
                        ax.grid(
                            which="minor", color="gray", linestyle="-", linewidth=0.25
                        )

                        plt.tight_layout()
                        test_heatmap_filename = f'./visualization/class_wise_recall/test_heatmap_model_{cfg.model}_case_{cfg.path.split("/")[-1]}_seed_{cfg.seed}.png'
                        plt.savefig(test_heatmap_filename, dpi=200, format="png")
                        plt.close()

                        wandb.log(
                            {
                                "Test Class-wise Recall Heatmap": wandb.Image(
                                    test_heatmap_filename
                                ),
                                "ep": ep,
                            }
                        )

            wandb.log({**stats_ep, "ep": ep})

    if cfg.local_rank == 0:
        print(f"Best Recall@1 (epoch {best_epoch}): {best_recall_at_1*100:.2f}")

        data = {}
        data.update({f"{k}": [v*100] for k, v in recall_scores_at_best_r1.items()})

        df = pd.DataFrame(data)

        for k, v in recall_scores_at_best_r1.items():
            print(f"{k}: {v*100:.2f}")

        excel_file = "best_recall_scores.xlsx"
        df.to_excel(excel_file, index=False)

        wandb.save(excel_file)

    if cfg.save_emb:
        ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
        x, y = get_emb_f(ds_type=ds_type)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_eval.pt")

        x, y = get_emb_f(ds_type="train")
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_train.pt")
