import os
import json
import torchvision.transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import numpy as np

def center_crop_fn(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

from torchvision import transforms

def build_old_imagenet_transform(image_size: int, center_crop: bool = True):
    """
    Matches the old ImageNetTarIterable preprocessing:
    Resize(image_size, BICUBIC) -> CenterCrop(image_size) or Resize((image_size,image_size))
    -> ToTensor()
    """
    resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)
    if center_crop:
        crop = transforms.CenterCrop(image_size)
    else:
        crop = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
    return transforms.Compose([resize, crop, transforms.ToTensor()])

def load_imagenet100_wnids(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    # if your json top-level key is "root" as shown
    wnids = list(data["root"].keys())
    return set(wnids)

class PixImageNet(ImageFolder):
    def __init__(
        self,
        root,
        resolution=256,
        random_crop=False,
        imagenet100_json=None,
    ):
        super().__init__(root)

        self.img_tfm = build_old_imagenet_transform(
            image_size=resolution,
            center_crop=True,        # <- change to False if you ever want full resize
        )

        # ---- filter to ImageNet-100 classes if json is provided ----
        if imagenet100_json is not None:
            allowed_wnids = load_imagenet100_wnids(imagenet100_json)
          
            filtered_samples = []
            kept_wnids = set()
            skipped_wnids = set()

            for image_path, target in self.samples:
                wnid = os.path.basename(os.path.dirname(image_path))
                if wnid in allowed_wnids:
                    filtered_samples.append((image_path, target))
                    kept_wnids.add(wnid)
                else:
                    skipped_wnids.add(wnid)
                    
            self.samples = filtered_samples
            # keep targets consistent with samples
            self.targets = [t for _, t in self.samples]

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]

        img = Image.open(image_path).convert("RGB")

        img_t = self.img_tfm(img)      

        raw_image = img_t              
        normalized_image = img_t

        wnid = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)

        metadata = {
            "raw_image": raw_image,
            "class": target,
            "wnid": wnid,
            "filename": filename,
        }
        return normalized_image, target, metadata
    
def make_imagenet_loader(
    split: str,
    base_dir: str,
    bs: int = 8,
    workers: int = 8,
    image_size: int = 256,
    shuffle: bool = False,
    imagenet100_json: str | None = None,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Build a DataLoader over an ImageNet-style folder structure using PixImageNet.

    When `world_size > 1`, each rank's loader only iterates that rank's
    1/world_size share of the dataset via a DistributedSampler. Without this,
    every rank's dataloader workers would independently read the entire
    1.28M-file ImageNet tree from lustre — the per-rank lustre read traffic
    would be world_size larger than needed and bottleneck encoding on
    small-file metadata throughput. The encoder loop no longer needs the
    `b_idx % world_size == rank` skip when this is used.

    Expected directory layout:
        base_dir/
            train/
                n01440764/
                    xxx.JPEG
                ...
            val/
                n01440764/
                    yyy.JPEG
                ...
    """
    if split.lower() in "train":
        split_dir = "train"
        random_crop = True
    else:
        split_dir = "val"
        random_crop = False

    root = os.path.join(base_dir, split_dir)

    dataset = PixImageNet(
        root=root,
        resolution=image_size,
        random_crop=random_crop,
        imagenet100_json=imagenet100_json,
    )

    sampler = None
    if world_size > 1:
        # DistributedSampler interleaves indices across ranks (rank r gets
        # idx r, r+W, r+2W, ...). This keeps each rank's WNID coverage even,
        # which is what we want for the per-rank wnid->label map.
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        # DataLoader can't combine sampler + shuffle.
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader
