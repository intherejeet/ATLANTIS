import sys
import random
from tap import Tap
from typing_extensions import Literal
from hyp_metric.model import init_model
import torch
import os
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import PIL
from torchvision.utils import save_image
from tqdm import tqdm

random.seed(42)

class Config(Tap):
    path: str = None  # path to image folder
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "Cars"  # dataset name
    model: str = "dino_vits16"  # model name: deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16
    resize: int = 256  # image resize
    crop: int = 224  # center crop after resize
    epsilon: float = 0.05
    num_iter: int = 1
    targets_num = 1
    mode: str = "baseline"  # baseline or ours
    data_mode: str = 'zeroshot_orig' # zeroshot_orig or zeroshot_syn or fullshot_orig or fullshot_syn
    gpu: int = 0

cfg: Config = Config().parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths based on data_mode
if cfg.data_mode == 'zeroshot_orig':
    cfg.path = "path/to/zeroshot/orig/test"
elif cfg.data_mode == 'zeroshot_syn':
    cfg.path = "path/to/syn/test/threshold_1.5"
elif cfg.data_mode == 'fullshot_orig':
    cfg.path = "path/to/fullshot/orig/split1/test"
elif cfg.data_mode == 'fullshot_syn':
    cfg.path = "path/to/syn/all/threshold_1.5"
else:
    raise Exception("Sorry, data_mode is not defined properly")

class ClassWiseSampler(Sampler):
    def __init__(self, class_to_idx):
        self.class_to_idx = class_to_idx

    def __iter__(self):
        for label in self.class_to_idx.keys():
            indices = self.class_to_idx[label]
            yield indices

    def __len__(self):
        return len(self.class_to_idx)

def attack_feature_evasion(cfg, imgs, labels, model, target_nums=1):
    images_clean_saved = imgs.clone()
    embeds = model(images_clean_saved.to(device))

    embeddings_tar_list = []
    for q in range(target_nums):
        embeddings_tar_list.append(
            embeds.detach().clone()[torch.randperm(embeds.size()[0])]
        )

    lower_bound = torch.clamp(
        imgs - cfg.epsilon, min=imgs.min().item(), max=imgs.max().item()
    )
    upper_bound = torch.clamp(
        imgs + cfg.epsilon, min=imgs.min().item(), max=imgs.max().item()
    )

    init_start = torch.empty_like(imgs).uniform_(-0.0005, 0.0005)
    start_adv = imgs + init_start

    adv = start_adv
    for i in range(cfg.num_iter):
        adv.requires_grad = True
        adv_feat = model(adv.to(device))

        loss_attack = torch.Tensor([0]).to(device)
        for embeds_tar in embeddings_tar_list:
            diff = embeds_tar - adv_feat
            loss_attack += torch.mean(torch.norm(diff, p=2, dim=1))
        loss_attack = (loss_attack / target_nums).pow(2)

        g = torch.autograd.grad(
            loss_attack, adv, retain_graph=False, create_graph=False
        )[0]

        adv = adv + torch.sign(g) * (cfg.epsilon / cfg.num_iter)
        adv = torch.where(adv > lower_bound, adv, lower_bound)
        adv = torch.where(adv < upper_bound, adv, upper_bound).detach()

    return adv, labels

if __name__ == "__main__":
    # Image transformations
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    model = init_model(cfg)
    if cfg.mode == "baseline":
        print("loading baseline model...")
        param_path = "./saved_params/baseline_dino.pth"
    else:
        print("loading our model...")
        param_path = "./saved_params/ours_best_dino.pth"

    model.load_state_dict(torch.load(param_path))
    model.eval()

    ### Data loading
    eval_tr = T.Compose(
        [
            T.Resize(cfg.resize, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(cfg.crop),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    dataset = ImageFolder(root=cfg.path, transform=eval_tr)

    # Build an index for each class
    class_to_idx = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_to_idx:
            class_to_idx[label] = []
        class_to_idx[label].append(idx)

    # Create a ClassWiseSampler instance
    sampler = ClassWiseSampler(class_to_idx)

    # Create DataLoader with ClassWiseSampler
    data_loader = DataLoader(dataset, batch_sampler=sampler)

    adversarial_root = f"path/to/adversarial/{cfg.mode}_{cfg.data_mode}testdata_{cfg.ds}_{cfg.model}_targets_{cfg.targets_num}_epsilon_{cfg.epsilon}_itr_{cfg.num_iter}"  # Adversarial root directory

    num_batches = len(data_loader)
    pbar = tqdm(
        total=num_batches, desc="Generating adversarial examples"
    )  # Initialize tqdm progress bar

    # Iterate over batches
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        adv_images, adv_labels = attack_feature_evasion(
            cfg, images, labels, model, target_nums=cfg.targets_num
        )

        # Find corresponding file paths from the original dataset
        batch_size = adv_images.size(0)  # Get the batch size
        batch_indices = sampler.class_to_idx[labels.cpu().numpy()[0]][:batch_size]

        for j, idx in enumerate(batch_indices):
            _, original_label = dataset[idx]
            original_path = dataset.imgs[idx][0]
            original_filename = original_path.split("/")[-1]
            new_dir = os.path.join(adversarial_root, f"{original_path.split('/')[-2]}")
            os.makedirs(new_dir, exist_ok=True)
            if isinstance(
                original_filename, str
            ):  # Check if original_filename is a string
                new_path = os.path.join(new_dir, f"adv_{original_filename}")
            else:
                print(
                    f"Warning: original_filename is not a string. It is: {original_filename}"
                )
                continue  # Skip to next iteration if filename is not as expected

            # Save adversarial image
            save_image(adv_images[j].cpu(), new_path, normalize=True)

        pbar.update(1)  # Update tqdm progress by 1
    pbar.close()  # Close the tqdm progress bar after loop ends
