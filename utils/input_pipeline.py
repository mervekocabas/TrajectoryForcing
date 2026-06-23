"""ImageNet input pipeline."""

import os
import random
import json
import io
import pickle
import zipfile
from collections import OrderedDict
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

from utils.logging_util import log_for_0

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def loader(path: str):
    return pil_loader(path)

def _region_mean_canvas_from_ids(z_hwc, region_ids_hw):
    """Average z tokens per region id and broadcast the mean back to each region."""
    h, w, c = z_hwc.shape
    hw = h * w

    z_flat = z_hwc.reshape((hw, c))
    ids_flat = region_ids_hw.reshape((hw,)).astype(jnp.int32)

    valid = ids_flat >= 0
    safe_ids = jnp.where(valid, ids_flat, 0)
    safe_ids = jnp.clip(safe_ids, 0, hw - 1)

    valid_f = valid.astype(z_flat.dtype)
    sums = jnp.zeros((hw, c), dtype=z_flat.dtype).at[safe_ids].add(
        z_flat * valid_f[:, None]
    )
    counts = jnp.zeros((hw,), dtype=z_flat.dtype).at[safe_ids].add(valid_f)
    means = sums / jnp.maximum(counts, jnp.array(1.0, dtype=z_flat.dtype))[:, None]

    out_flat = means[safe_ids]
    out_flat = jnp.where(valid[:, None], out_flat, 0)
    return out_flat.reshape((h, w, c))


def _region_mean_canvas_batch(z, region_ids):
    return jax.vmap(_region_mean_canvas_from_ids)(z, region_ids)


def _select_level_batch(level_tensors, levels):
    """Select a level tensor per sample from a tuple of [B, ...] tensors."""
    stacked = jnp.stack(level_tensors, axis=1)  # [B, L, ...]
    idx = jnp.clip(levels.astype(jnp.int32), 0, stacked.shape[1] - 1)
    batch_idx = jnp.arange(stacked.shape[0], dtype=jnp.int32)
    return stacked[batch_idx, idx]


def _override_levels_from_device_axis(levels, num_levels, devices_per_level):
    """Pin one level id per device using pmap axis index."""
    axis_idx = jax.lax.axis_index("batch")
    level_id = jnp.minimum(axis_idx // devices_per_level, num_levels - 1)
    return jnp.full_like(levels, level_id, dtype=levels.dtype)


def _reconstruct_latent_levels_on_device(z, levels, objbg_ids, parts_ids, subparts_ids):
    """Build (cur, prev, region_ids) from z + hierarchical region-id maps on device."""
    z = z.astype(jnp.float32)
    levels = levels.astype(jnp.int32)
    objbg_ids = objbg_ids.astype(jnp.int32)
    parts_ids = parts_ids.astype(jnp.int32)
    subparts_ids = subparts_ids.astype(jnp.int32)

    zero = jnp.zeros_like(z)
    neg_region = -jnp.ones_like(objbg_ids, dtype=jnp.int32)

    def mixed_level_path(_):
        level0 = _region_mean_canvas_batch(z, objbg_ids)
        level1 = _region_mean_canvas_batch(z, parts_ids)
        level2 = _region_mean_canvas_batch(z, subparts_ids)

        cur = _select_level_batch((level0, level1, level2, z), levels)
        prev = _select_level_batch((zero, level0, level1, level2), levels)
        region_ids = _select_level_batch(
            (objbg_ids, parts_ids, subparts_ids, neg_region), levels
        )
        return cur, prev, region_ids

    def single_level_path(_):
        level_id = jnp.clip(levels[0], 0, 3)

        def build_level0(_):
            level0 = _region_mean_canvas_batch(z, objbg_ids)
            return level0, zero, objbg_ids

        def build_level1(_):
            level0 = _region_mean_canvas_batch(z, objbg_ids)
            level1 = _region_mean_canvas_batch(z, parts_ids)
            return level1, level0, parts_ids

        def build_level2(_):
            level1 = _region_mean_canvas_batch(z, parts_ids)
            level2 = _region_mean_canvas_batch(z, subparts_ids)
            return level2, level1, subparts_ids

        def build_level3(_):
            level2 = _region_mean_canvas_batch(z, subparts_ids)
            return z, level2, neg_region

        return jax.lax.switch(
            level_id,
            (build_level0, build_level1, build_level2, build_level3),
            operand=None,
        )

    same_level = jnp.all(levels == levels[0])
    return jax.lax.cond(same_level, single_level_path, mixed_level_path, operand=None)


def process_image_on_tpu(image, use_flip=True, flip_key=None):
    """
    Process a single image on TPU: convert to float, normalize, flip.
    Center crop is already done on CPU to ensure uniform batch size.

    Args:
        image: uint8 array of shape (image_size, image_size, C)
        use_flip: whether to apply random horizontal flip
        flip_key: JAX random key for flipping (required if use_flip=True)

    Returns:
        Processed image as float32 array of shape (image_size, image_size, C)
        normalized to [-1, 1]
    """
    # Convert to float [0, 1]
    image = image.astype(jnp.float32) / 255.0

    # Random horizontal flip
    if use_flip and flip_key is not None:
        should_flip = jax.random.bernoulli(flip_key, p=0.5)
        image = jnp.where(should_flip, jnp.fliplr(image), image)

    # Normalize to [-1, 1]
    image = (image - 0.5) / 0.5

    return image


def process_batch_on_tpu(
    batch_dict,
    use_flip=True,
    rng_key=None,
    pin_levels_to_device_groups=False,
    num_levels=4,
    devices_per_level=1,
):
    """
    Process a batch of images on TPU (designed to be used with pmap).
    This function processes one device's batch at a time (called by pmap).
    Images are already center-cropped on CPU to uniform size.

    Args:
        batch_dict: dict with 'image' (uint8) and 'label'
                   image shape: (device_batch_size, image_size, image_size, C)
        use_flip: whether to apply random horizontal flip
        rng_key: JAX random key for this device's batch

    Returns:
        Processed batch with images as float32 normalized to [-1, 1]
        image shape: (device_batch_size, image_size, image_size, C)
    """
    labels = batch_dict["label"]

    if (
        pin_levels_to_device_groups
        and "level" in batch_dict
    ):
        levels = _override_levels_from_device_axis(
            batch_dict["level"],
            int(num_levels),
            int(devices_per_level),
        )
    else:
        levels = batch_dict.get("level", None)

    # Latent mode (raw z + ids): build level canvases on device.
    if "z" in batch_dict:
        cur, prev, region_ids = _reconstruct_latent_levels_on_device(
            z=batch_dict["z"],
            levels=levels,
            objbg_ids=batch_dict["objbg_ids"],
            parts_ids=batch_dict["parts_ids"],
            subparts_ids=batch_dict["subparts_ids_global"],
        )
        return {
            "image": cur,
            "label": labels,
            "prev": prev,
            "level": levels,
            "region_ids": region_ids,
        }

    # Latent mode (legacy dense cur/prev): no image-space normalization or random flips.
    if "prev" in batch_dict:
        images = batch_dict["image"]
        out = {
            "image": images.astype(jnp.float32),
            "label": labels,
            "prev": batch_dict["prev"].astype(jnp.float32),
            "level": levels,
        }
        if "region_ids" in batch_dict:
            out["region_ids"] = batch_dict["region_ids"]
        return out

    images = batch_dict["image"]

    # Generate flip keys for each image if needed
    if use_flip and rng_key is not None:
        device_batch_size = images.shape[0]
        flip_keys = jax.random.split(rng_key, device_batch_size)
    else:
        flip_keys = None

    # Process each image in the batch
    def process_single(image, flip_key):
        return process_image_on_tpu(image, use_flip, flip_key)

    if use_flip and flip_keys is not None:
        processed_images = jax.vmap(process_single)(images, flip_keys)
    else:
        processed_images = jax.vmap(lambda img: process_image_on_tpu(img, False, None))(
            images
        )
    
    return {
        "image": processed_images,
        "label": labels,
    }


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def prepare_batch_data(batch, batch_size=None):
    """
    Reformat a input batch from PyTorch Dataloader.

    Args: (torch)
      batch = (image, label)
        image: shape (host_batch_size, H, W, C) - uint8 numpy arrays
        label: shape (host_batch_size)
      batch_size = expected batch_size of this node, for eval's drop_last=False only

    Returns: a dict (numpy)
      image shape (local_devices, device_batch_size, H, W, C) - uint8
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 6:
        z, label, level, objbg_ids, parts_ids, subparts_ids = batch

        if batch_size is not None and batch_size > z.shape[0]:
            pad_n = batch_size - z.shape[0]
            z = torch.cat(
                [z, torch.zeros((pad_n,) + z.shape[1:], dtype=z.dtype)],
                axis=0,
            )
            label = torch.cat(
                [label, -torch.ones((pad_n,), dtype=label.dtype)],
                axis=0,
            )
            level = torch.cat(
                [level, torch.zeros((pad_n,), dtype=level.dtype)],
                axis=0,
            )
            objbg_ids = torch.cat(
                [objbg_ids, -torch.ones((pad_n,) + objbg_ids.shape[1:], dtype=objbg_ids.dtype)],
                axis=0,
            )
            parts_ids = torch.cat(
                [parts_ids, -torch.ones((pad_n,) + parts_ids.shape[1:], dtype=parts_ids.dtype)],
                axis=0,
            )
            subparts_ids = torch.cat(
                [
                    subparts_ids,
                    -torch.ones((pad_n,) + subparts_ids.shape[1:], dtype=subparts_ids.dtype),
                ],
                axis=0,
            )

        local_device_count = jax.local_device_count()
        z = z.reshape((local_device_count, -1) + z.shape[1:])
        label = label.reshape(local_device_count, -1)
        level = level.reshape(local_device_count, -1)
        objbg_ids = objbg_ids.reshape((local_device_count, -1) + objbg_ids.shape[1:])
        parts_ids = parts_ids.reshape((local_device_count, -1) + parts_ids.shape[1:])
        subparts_ids = subparts_ids.reshape((local_device_count, -1) + subparts_ids.shape[1:])

        return {
            "z": z.numpy(),
            "label": label.numpy(),
            "level": level.numpy(),
            "objbg_ids": objbg_ids.numpy(),
            "parts_ids": parts_ids.numpy(),
            "subparts_ids_global": subparts_ids.numpy(),
        }

    if isinstance(batch, (tuple, list)) and len(batch) == 5:
        image, label, level, prev, region_ids = batch

        if batch_size is not None and batch_size > image.shape[0]:
            pad_n = batch_size - image.shape[0]
            image = torch.cat(
                [image, torch.zeros((pad_n,) + image.shape[1:], dtype=image.dtype)],
                axis=0,
            )
            prev = torch.cat(
                [prev, torch.zeros((pad_n,) + prev.shape[1:], dtype=prev.dtype)],
                axis=0,
            )
            label = torch.cat(
                [label, -torch.ones((pad_n,), dtype=label.dtype)],
                axis=0,
            )
            level = torch.cat(
                [level, torch.zeros((pad_n,), dtype=level.dtype)],
                axis=0,
            )
            region_ids = torch.cat(
                [
                    region_ids,
                    -torch.ones((pad_n,) + region_ids.shape[1:], dtype=region_ids.dtype),
                ],
                axis=0,
            )

        local_device_count = jax.local_device_count()
        image = image.reshape((local_device_count, -1) + image.shape[1:])
        prev = prev.reshape((local_device_count, -1) + prev.shape[1:])
        label = label.reshape(local_device_count, -1)
        level = level.reshape(local_device_count, -1)
        region_ids = region_ids.reshape((local_device_count, -1) + region_ids.shape[1:])

        return {
            "image": image.numpy(),
            "label": label.numpy(),
            "prev": prev.numpy(),
            "level": level.numpy(),
            "region_ids": region_ids.numpy(),
        }

    image, label = batch

    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > image.shape[0]:
        image = torch.cat(
            [
                image,
                torch.zeros(
                    (batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype
                ),
            ],
            axis=0,
        )
        label = torch.cat(
            [label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)],
            axis=0,
        )

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    local_device_count = jax.local_device_count()
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    image = image.numpy()
    label = label.numpy()

    return_dict = {
        "image": image,
        "label": label,
    }

    return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

from torchvision.datasets.folder import pil_loader

class SyntheticImageDataset(Dataset):
    """Synthetic dataset of random uint8 images and random labels."""

    def __init__(self, size, image_size, num_classes, seed=0):
        self.size = int(size)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Deterministic per-index generation keeps samples stable across workers.
        rng = np.random.default_rng(self.seed + int(index))
        image = rng.integers(
            low=0,
            high=256,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        label = int(rng.integers(low=0, high=self.num_classes))
        return image, label


class SyntheticLatentHierDataset(Dataset):
    """Synthetic latent-hier dataset yielding random z + region ids."""

    def __init__(self, size, image_size, latent_dim, num_levels=4, num_classes=1000, seed=0):
        self.size = int(size)
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.num_levels = int(num_levels)
        self.num_classes = int(num_classes)
        self.seed = int(seed)

    def __len__(self):
        return self.size

    def _random_region_ids(self, rng, h, w, max_regions):
        # Keep ids compact and valid for device-side reconstruction: [0, H*W-1].
        n_regions = int(rng.integers(low=1, high=max_regions + 1))
        ids = rng.integers(low=0, high=n_regions, size=(h, w), dtype=np.int16)
        return torch.from_numpy(ids)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)) and len(index) == 2:
            sample_idx, level_id = int(index[0]), int(index[1])
        else:
            sample_idx = int(index)
            # Match LatentHierDataset behavior when not using LevelBatchSampler.
            level_id = int(np.random.randint(0, self.num_levels))

        if level_id < 0 or level_id >= self.num_levels:
            raise ValueError(f"level_id must be in [0, {self.num_levels - 1}], got {level_id}")

        rng = np.random.default_rng(self.seed + sample_idx)
        h = self.image_size
        w = self.image_size
        hw = h * w

        z = rng.standard_normal(size=(self.latent_dim, h, w), dtype=np.float32)
        label = int(rng.integers(low=0, high=max(1, self.num_classes)))

        # Coarser to finer region counts.
        objbg_ids = self._random_region_ids(rng, h, w, max_regions=min(2, hw))
        parts_ids = self._random_region_ids(rng, h, w, max_regions=min(16, hw))
        subparts_ids = self._random_region_ids(rng, h, w, max_regions=min(64, hw))

        return torch.from_numpy(z), label, level_id, objbg_ids, parts_ids, subparts_ids


def _list_latent_samples(root_dir):
    manifest_path = os.path.join(root_dir, ".latent_sample_index.pkl")
    if os.path.exists(manifest_path):
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        samples = manifest.get("samples", manifest if isinstance(manifest, list) else None)
        if not isinstance(samples, list):
            raise RuntimeError(
                f"Invalid latent sample manifest format at {manifest_path}: expected list under 'samples'"
            )
        out = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            kind = str(s.get("kind", "")).lower()
            if kind not in {"pt", "npz", "tar"}:
                continue
            e = dict(s)
            e["kind"] = kind
            if "path" in e:
                e["path"] = str(e["path"])
            if "member" in e and e["member"] is not None:
                e["member"] = str(e["member"])
            if "offset" in e and e["offset"] is not None:
                e["offset"] = int(e["offset"])
            if "size" in e and e["size"] is not None:
                e["size"] = int(e["size"])
            out.append(e)
        if len(out) == 0:
            raise RuntimeError(f"No usable samples found in latent sample manifest {manifest_path}")
        return out

    pt_files = []
    npz_files = []
    for root, _dirs, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".pt"):
                pt_files.append(os.path.join(root, name))
            elif name.endswith(".npz"):
                npz_files.append(os.path.join(root, name))

    samples = []
    for path in sorted(pt_files):
        wnid = os.path.basename(os.path.dirname(path)) or "unknown"
        samples.append({"kind": "pt", "path": path, "member": None, "wnid": wnid})

    for path in sorted(npz_files):
        wnid = os.path.splitext(os.path.basename(path))[0] or "unknown"
        with zipfile.ZipFile(path, mode="r") as zf:
            members = sorted(name for name in zf.namelist() if name.endswith(".npy"))
        for member in members:
            samples.append({"kind": "npz", "path": path, "member": member, "wnid": wnid})

    return samples


def _to_torch_recursive(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            return obj
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _to_torch_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_torch_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_torch_recursive(v) for v in obj)
    return obj


def _load_npz_member(path, member, zf=None, keys=None):
    if zf is None:
        with zipfile.ZipFile(path, mode="r") as local_zf:
            with local_zf.open(member, mode="r") as fh:
                payload = np.load(fh, allow_pickle=True)
    else:
        with zf.open(member, mode="r") as fh:
            payload = np.load(fh, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == np.object_ and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected dict payload in {path}:{member}, got {type(payload)}")
    if keys is not None:
        payload = {k: payload[k] for k in keys if k in payload}
    return _to_torch_recursive(payload)


def _torch_load_trusted(obj):
    """Load trusted torch payloads across PyTorch versions (2.6 changed defaults)."""
    try:
        return torch.load(obj, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(obj, map_location="cpu")


def _load_tar_member_by_index(sample, fh=None, keys=None):
    path = sample["path"]
    offset = int(sample["offset"])
    size = int(sample["size"])

    if fh is None:
        with open(path, "rb") as local_fh:
            local_fh.seek(offset)
            payload = local_fh.read(size)
    else:
        fh.seek(offset)
        payload = fh.read(size)

    if len(payload) != size:
        raise RuntimeError(
            f"Short read for tar sample {path}:{sample.get('member', '')} at offset={offset}: "
            f"expected {size} bytes, got {len(payload)}"
        )

    data = _torch_load_trusted(io.BytesIO(payload))
    if keys is not None and isinstance(data, dict):
        data = {k: data[k] for k in keys if k in data}
    return data


def _load_latent_sample(sample, zf=None, tar_fh=None, keys=None):
    if sample["kind"] == "pt":
        data = _torch_load_trusted(sample["path"])
        if keys is not None and isinstance(data, dict):
            return {k: data[k] for k in keys if k in data}
        return data
    if sample["kind"] == "npz":
        return _load_npz_member(sample["path"], sample["member"], zf=zf, keys=keys)
    if sample["kind"] == "tar":
        return _load_tar_member_by_index(sample, fh=tar_fh, keys=keys)
    raise ValueError(f"Unknown sample kind: {sample['kind']}")


def _build_canvas(ids, content, latent_dim):
    ids = torch.as_tensor(ids, dtype=torch.int64)
    content = torch.as_tensor(content, dtype=torch.float32)
    h, w = ids.shape
    if content.numel() == 0:
        return torch.zeros((latent_dim, h, w), dtype=torch.float32)
    safe_ids = ids.clone()
    invalid = safe_ids < 0
    safe_ids[invalid] = 0
    canvas = content[safe_ids]
    if invalid.any():
        canvas[invalid] = 0
    return canvas.permute(2, 0, 1).contiguous()


def _to_chw(z, latent_dim):
    z = torch.as_tensor(z, dtype=torch.float32)
    if z.ndim != 3:
        raise RuntimeError(f"Expected z to be 3D, got shape {tuple(z.shape)}")
    if z.shape[0] == latent_dim:
        return z.contiguous()
    if z.shape[-1] == latent_dim:
        return z.permute(2, 0, 1).contiguous()
    raise RuntimeError(f"Could not infer channel dimension for z with shape {tuple(z.shape)}")


def _extract_level_canvases(data, latent_dim):
    level0 = _build_canvas(data["objbg_ids"], data["obj_content"], latent_dim)
    level1 = _build_canvas(data["parts_ids"], data["parts_content"], latent_dim)
    level2 = _build_canvas(data["subparts_ids_global"], data["subparts_content_global"], latent_dim)
    level3 = _to_chw(data["z"], latent_dim)
    return [level0, level1, level2, level3]


_LEVEL_REGION_KEYS = (
    ("objbg_ids", "obj_content"),
    ("parts_ids", "parts_content"),
    ("subparts_ids_global", "subparts_content_global"),
)


def _extract_single_level_canvas(data, latent_dim, level_id):
    if level_id == 3:
        return _to_chw(data["z"], latent_dim)
    ids_key, content_key = _LEVEL_REGION_KEYS[level_id]
    return _build_canvas(data[ids_key], data[content_key], latent_dim)


class LatentHierDataset(Dataset):
    """Hierarchical latent dataset yielding z + region ids for device-side canvas build."""

    def __init__(
        self,
        data_path,
        split="train",
        latent_dim=768,
        num_levels=4,
        class_map=None,
        class_map_path=None,
        max_open_npz=32,
    ):
        split_root = os.path.join(data_path, split)
        self.data_root = split_root if os.path.exists(split_root) else data_path
        self.latent_dim = int(latent_dim)
        self.num_levels = int(num_levels)
        self.max_open_npz = max(0, int(max_open_npz))
        self.samples = _list_latent_samples(self.data_root)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No latent samples (.pt/.npz/.tar or manifest) found under {self.data_root}"
            )

        self._runtime_pid = None
        self._npz_readers = OrderedDict()
        self._tar_readers = OrderedDict()

        self.class_map = class_map
        if self.class_map is None and class_map_path and os.path.exists(class_map_path):
            with open(class_map_path, "r") as f:
                self.class_map = json.load(f)

        if self.class_map is None:
            self.class_map = {}
            for sample in self.samples:
                wnid = sample.get("wnid", "unknown")
                if sample["kind"] == "pt" and wnid in {"", "train", "validation", "val"}:
                    data = _load_latent_sample(sample)
                    wnid = data.get("meta", {}).get("wnid", "unknown")
                if wnid not in self.class_map:
                    self.class_map[wnid] = len(self.class_map)
            if class_map_path:
                with open(class_map_path, "w") as f:
                    json.dump(self.class_map, f, indent=2)

    def __len__(self):
        return len(self.samples)

    def _ensure_runtime_state(self):
        pid = os.getpid()
        if self._runtime_pid != pid:
            self._close_npz_readers()
            self._close_tar_readers()
            self._runtime_pid = pid
            self._npz_readers = OrderedDict()
            self._tar_readers = OrderedDict()

    def _get_npz_reader(self, path):
        if self.max_open_npz <= 0:
            return None
        self._ensure_runtime_state()
        reader = self._npz_readers.pop(path, None)
        if reader is None:
            reader = zipfile.ZipFile(path, mode="r")
        self._npz_readers[path] = reader
        while len(self._npz_readers) > self.max_open_npz:
            _old_path, old_reader = self._npz_readers.popitem(last=False)
            old_reader.close()
        return reader

    def _close_npz_readers(self):
        for reader in self._npz_readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self._npz_readers = OrderedDict()

    def _get_tar_reader(self, path):
        if self.max_open_npz <= 0:
            return None
        self._ensure_runtime_state()
        reader = self._tar_readers.pop(path, None)
        if reader is None or reader.closed:
            reader = open(path, "rb")
        self._tar_readers[path] = reader
        while len(self._tar_readers) > self.max_open_npz:
            _old_path, old_reader = self._tar_readers.popitem(last=False)
            try:
                old_reader.close()
            except Exception:
                pass
        return reader

    def _close_tar_readers(self):
        for reader in self._tar_readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self._tar_readers = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime_pid"] = None
        state["_npz_readers"] = OrderedDict()
        state["_tar_readers"] = OrderedDict()
        return state

    def __del__(self):
        if hasattr(self, "_npz_readers"):
            self._close_npz_readers()
        if hasattr(self, "_tar_readers"):
            self._close_tar_readers()

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)) and len(idx) == 2:
            sample_idx, level_id = int(idx[0]), int(idx[1])
        else:
            sample_idx = int(idx)
            level_id = random.randint(0, self.num_levels - 1)

        if level_id < 0 or level_id >= self.num_levels:
            raise ValueError(f"level_id must be in [0, {self.num_levels - 1}], got {level_id}")

        sample = self.samples[sample_idx]
        zf = self._get_npz_reader(sample["path"]) if sample["kind"] == "npz" else None
        tar_fh = self._get_tar_reader(sample["path"]) if sample["kind"] == "tar" else None
        data = _load_latent_sample(
            sample,
            zf=zf,
            tar_fh=tar_fh,
            keys=(
                "z",
                "levels_chw",
                "objbg_ids",
                "parts_ids",
                "subparts_ids_global",
                "meta",
                "label",
            ),
        )

        wnid = data.get("meta", {}).get("wnid", sample.get("wnid", "unknown"))
        if "label" in data:
            label = int(data["label"])
        else:
            label = self.class_map.get(wnid, 0)

        missing = [
            k
            for k in ("objbg_ids", "parts_ids", "subparts_ids_global")
            if k not in data
        ]
        if ("z" not in data) and ("levels_chw" not in data):
            missing.append("z|levels_chw")
        if missing:
            raise RuntimeError(
                f"Sample {sample['path']} missing required keys for latent reconstruction: {missing}"
            )

        if "z" in data:
            z_src = data["z"]
        else:
            z_src = data["levels_chw"][-1]
        z = _to_chw(z_src, self.latent_dim)
        h, w = z.shape[1], z.shape[2]

        objbg_ids = torch.as_tensor(data["objbg_ids"], dtype=torch.int16)
        parts_ids = torch.as_tensor(data["parts_ids"], dtype=torch.int16)
        subparts_ids = torch.as_tensor(data["subparts_ids_global"], dtype=torch.int16)

        for name, ids in (
            ("objbg_ids", objbg_ids),
            ("parts_ids", parts_ids),
            ("subparts_ids_global", subparts_ids),
        ):
            if ids.shape != (h, w):
                raise RuntimeError(
                    f"{name} shape mismatch at sample {sample_idx}: expected {(h, w)}, got {tuple(ids.shape)}"
                )
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < -1 or max_id >= (h * w):
                raise RuntimeError(
                    f"{name} id range invalid at sample {sample_idx}: expected ids in [-1, {h*w - 1}], "
                    f"got [{min_id}, {max_id}]"
                )

        return z, int(label), int(level_id), objbg_ids, parts_ids, subparts_ids


def collate_latent_hier(batch):
    z, labels, levels, objbg_ids, parts_ids, subparts_ids = zip(*batch)
    z = torch.stack(z, dim=0).permute(0, 2, 3, 1).contiguous().to(torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)
    levels = torch.tensor(levels, dtype=torch.int32)
    objbg_ids = torch.stack([torch.as_tensor(x, dtype=torch.int16) for x in objbg_ids], dim=0)
    parts_ids = torch.stack([torch.as_tensor(x, dtype=torch.int16) for x in parts_ids], dim=0)
    subparts_ids = torch.stack(
        [torch.as_tensor(x, dtype=torch.int16) for x in subparts_ids], dim=0
    )
    return z, labels, levels, objbg_ids, parts_ids, subparts_ids


class LevelBatchSampler(Sampler):
    """Wrap a base sampler and emit batches with a single sampled level id."""

    def __init__(self, sampler, batch_size, drop_last, num_levels=4, seed=0,
                 round_robin=False):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_levels = int(num_levels)
        self.seed = int(seed)
        self.round_robin = bool(round_robin)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be > 0, got {self.num_levels}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batch = []
        batch_level = None
        batch_idx = 0
        # For round-robin: count total batches so leftover ones use last level.
        if self.round_robin:
            total_batches = len(self)
            full_cycles = total_batches // self.num_levels
            base_count = full_cycles * self.num_levels
        for idx in self.sampler:
            if batch_level is None:
                if self.round_robin:
                    if batch_idx < base_count:
                        batch_level = batch_idx % self.num_levels
                    else:
                        batch_level = self.num_levels - 1
                else:
                    batch_level = int(rng.integers(0, self.num_levels))
            batch.append((int(idx), batch_level))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                batch_level = None
                batch_idx += 1
        if batch and not self.drop_last:
            if batch_level is None:
                if self.round_robin:
                    if batch_idx < base_count:
                        batch_level = batch_idx % self.num_levels
                    else:
                        batch_level = self.num_levels - 1
                else:
                    batch_level = int(rng.integers(0, self.num_levels))
            yield batch

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


def create_latent_hier_split(dataset_cfg, batch_size, split):
    rank = jax.process_index()
    num_levels = int(dataset_cfg.get("num_levels", 4))
    if dataset_cfg.get("use_synthetic", False):
        ds = SyntheticLatentHierDataset(
            size=int(dataset_cfg.get("synthetic_size", 1281167)),
            image_size=int(dataset_cfg.image_size),
            latent_dim=int(dataset_cfg.image_channels),
            num_levels=num_levels,
            num_classes=int(dataset_cfg.get("num_classes", 1000)),
            seed=int(dataset_cfg.get("synthetic_seed", 0)),
        )
    else:
        ds = LatentHierDataset(
            data_path=dataset_cfg.root,
            split=split,
            latent_dim=int(dataset_cfg.image_channels),
            num_levels=num_levels,
            class_map_path=dataset_cfg.get("class_map_path", None),
            max_open_npz=int(dataset_cfg.get("max_open_npz", 32)),
        )

    log_for_0(
        "Using %s latent hierarchical dataset: split=%s, size=%d, latent_dim=%d, levels=%d",
        "synthetic" if dataset_cfg.get("use_synthetic", False) else "real",
        split,
        len(ds),
        int(dataset_cfg.image_channels),
        num_levels,
    )

    sampler = DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=True,
    )

    if dataset_cfg.get("sample_single_level_per_batch", True):
        batch_sampler = LevelBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            num_levels=num_levels,
            seed=int(dataset_cfg.get("level_seed", 0)),
            round_robin=bool(dataset_cfg.get("level_round_robin", False)),
        )
        it = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
            collate_fn=collate_latent_hier,
        )
    else:
        it = DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
            collate_fn=collate_latent_hier,
        )

    return it, len(it)


def create_imagenet_split(dataset_cfg, batch_size, split):
    """
    Creates a split for either ImageNet or latent hierarchical data.

    Args:
      dataset_cfg: Configurations for the dataset.
      batch_size: Batch size for the dataloader.
      split: 'train' or 'val'.
    Returns:
      it: A PyTorch Dataloader.
      steps_per_epoch: Number of steps to loop through the DataLoader.
    """
    dataset_kind = str(dataset_cfg.get("kind", "imagenet")).lower()
    if dataset_kind in {"latent_hier", "latent"}:
        return create_latent_hier_split(dataset_cfg, batch_size, split)

    rank = jax.process_index()
    if dataset_cfg.get("use_synthetic", False):
        ds = SyntheticImageDataset(
            size=dataset_cfg.get("synthetic_size", 50000),
            image_size=dataset_cfg.image_size,
            num_classes=dataset_cfg.num_classes,
            seed=dataset_cfg.get("synthetic_seed", 0),
        )
        log_for_0(
            "Using synthetic random dataset: size=%d, image_size=%d, num_classes=%d",
            len(ds),
            dataset_cfg.image_size,
            dataset_cfg.num_classes,
        )
    else:
        # Create a loader that applies center crop on CPU
        # This is necessary to ensure all images have uniform size for batching
        def loader_with_crop(path: str):
            img = pil_loader(path)
            img_cropped = center_crop_arr(img, dataset_cfg.image_size)
            return np.array(
                img_cropped
            )  # Returns uint8 array (image_size, image_size, C)

        root = os.path.join(dataset_cfg.root, split)
        ds = datasets.ImageFolder(
            root,
            transform=None,  # No transforms - crop is done in loader
            loader=loader_with_crop,  # Returns uint8 numpy arrays (image_size, image_size, 3)
        )

    log_for_0(ds)
    sampler = DistributedSampler(
        ds,
        num_replicas=jax.process_count(),
        rank=rank,
        shuffle=True,
    )
    it = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=True,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=(
            dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
        ),
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    return it, steps_per_epoch
