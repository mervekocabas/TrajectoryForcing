#!/usr/bin/env python3
"""Encode images into DINO-style latents and hierarchical region ids.

This exporter is tailored for the pMF latent dataloader (`utils/input_pipeline.py`):
- saves only `z` + region id maps (+ small metadata/label)
- optionally archives per-class `.pt` samples into uncompressed `.tar`
- writes `.latent_sample_index.pkl` with tar byte offsets for fast random access
- writes `class_map.json` (wnid -> integer label)
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
import queue
import tarfile
import threading
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from fastcluster import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from torchvision import transforms as T
from tqdm.auto import tqdm

from dataloader.imagenet_loader import make_imagenet_loader
from encoder import EncoderOnly


def build_coords(h: int, w: int) -> np.ndarray:
    """Normalized grid coordinates [N,2] in [0,1]^2."""
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    return np.stack([xs, ys], axis=-1).reshape(-1, 2)


def feature_distance_matrix(feats: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Full pairwise distance matrix [N,N]."""
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    if metric == "cosine":
        u = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        dist = 1.0 - (u @ u.T)
    elif metric == "l2":
        a = (feats**2).sum(axis=1, keepdims=True)
        dist = np.sqrt(np.maximum(a + a.T - 2.0 * (feats @ feats.T), 0.0))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    np.fill_diagonal(dist, 0.0)
    return dist


def add_spatial_term(df: np.ndarray, coords: np.ndarray, lam: float) -> np.ndarray:
    """Add lam * spatial L2 distance to a feature distance matrix."""
    if lam <= 0:
        return df
    b = (coords**2).sum(axis=1, keepdims=True)
    ds = np.sqrt(np.maximum(b + b.T - 2.0 * (coords @ coords.T), 0.0))
    return df + lam * ds


def condensed_from_full(dist: np.ndarray) -> np.ndarray:
    """Upper-triangular condensed distance vector for scipy linkage."""
    i, j = np.triu_indices(dist.shape[0], k=1)
    return dist[i, j]


def split_object_bg(
    feats: np.ndarray, coords: np.ndarray, alpha: float = 0.0, seed: int = 0
) -> np.ndarray:
    """KMeans(K=2) over [feats, alpha*coords]; returns bool mask for object cluster.
    """
    x = np.concatenate([feats, alpha * coords], axis=1)
    km = KMeans(n_clusters=2, n_init=1, random_state=seed).fit(x)
    labs = km.labels_

    center = np.array([[0.5, 0.5]])
    d0 = (
        np.mean(np.linalg.norm(coords[labs == 0] - center, axis=1))
        if np.any(labs == 0)
        else np.inf
    )
    d1 = (
        np.mean(np.linalg.norm(coords[labs == 1] - center, axis=1))
        if np.any(labs == 1)
        else np.inf
    )
    obj_id = 0 if d0 < d1 else 1
    return labs == obj_id


def parts_labels_with_normalized_cuts(
    feats_obj: np.ndarray,
    coords_obj: np.ndarray,
    metric: str = "cosine",
    lam_spatial: float = 0.2,
    linkage_method: str = "average",
    cut_mode: str = "height",
    h_sub: float = 0.35,
    h_part: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical clustering on object tokens; returns part and subpart labels."""
    n = feats_obj.shape[0]
    if n <= 1:
        zeros = np.zeros((n,), dtype=np.int32)
        return zeros, zeros

    df = feature_distance_matrix(feats_obj, metric=metric)
    dist = add_spatial_term(df, coords_obj, lam_spatial)
    condensed = condensed_from_full(dist)

    method = linkage_method
    if linkage_method == "ward" and metric != "l2":
        method = "average"

    z_link = scipy_linkage(condensed, method=method)
    heights = z_link[:, 2]
    hmin, hmax = heights.min(), heights.max()
    denom = (hmax - hmin) + 1e-8
    z_norm = z_link.copy()
    z_norm[:, 2] = (heights - hmin) / denom

    def cut_at(threshold: float) -> np.ndarray:
        return fcluster(z_norm, threshold, criterion="distance") - 1

    if cut_mode == "height":
        labs_part = cut_at(h_part)
        labs_sub = cut_at(h_sub)
    elif cut_mode == "quantile":
        t_part = np.quantile(z_norm[:, 2], h_part) if n > 2 else 0.0
        t_sub = np.quantile(z_norm[:, 2], h_sub) if n > 2 else 0.0
        labs_part = cut_at(t_part)
        labs_sub = cut_at(t_sub)
    else:
        raise ValueError(f"Unsupported cut_mode: {cut_mode}")

    return labs_part.astype(np.int32), labs_sub.astype(np.int32)


def global_labels_with_normalized_cuts(
    feats_obj: np.ndarray,
    coords_obj: np.ndarray,
    h_cuts: list[float],
    metric: str = "cosine",
    lam_spatial: float = 0.2,
    linkage_method: str = "average",
    cut_mode: str = "height",
) -> list[np.ndarray]:
    """Hierarchical clustering on object tokens; returns global labels per cut."""
    n = feats_obj.shape[0]
    if n <= 1:
        return [np.zeros((n,), dtype=np.int32) for _ in h_cuts]

    df = feature_distance_matrix(feats_obj, metric=metric)
    dist = add_spatial_term(df, coords_obj, lam_spatial)
    condensed = condensed_from_full(dist)

    method = linkage_method
    if linkage_method == "ward" and metric != "l2":
        method = "average"

    z_link = scipy_linkage(condensed, method=method)
    heights = z_link[:, 2]
    hmin, hmax = heights.min(), heights.max()
    denom = (hmax - hmin) + 1e-8
    z_norm = z_link.copy()
    z_norm[:, 2] = (heights - hmin) / denom

    def cut_at(threshold: float) -> np.ndarray:
        return fcluster(z_norm, threshold, criterion="distance") - 1

    out: list[np.ndarray] = []
    for h in h_cuts:
        if cut_mode == "height":
            labs = cut_at(h)
        elif cut_mode == "quantile":
            t = np.quantile(z_norm[:, 2], h) if n > 2 else 0.0
            labs = cut_at(float(t))
        else:
            raise ValueError(f"Unsupported cut_mode: {cut_mode}")
        out.append(labs.astype(np.int32))
    return out


def compute_hierarchy_from_z_chw(
    z_chw: torch.Tensor,
    metric: str = "cosine",
    lambda_spatial: float = 0.2,
    linkage_method: str = "average",
    cut_mode: str = "height",
    h_cuts: list[float] | None = None,
    alpha_obj_kmeans: float = 0.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute obj/bg and global multi-level label maps from [C,H,W] features."""
    if z_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(z_chw.shape)}")
    if h_cuts is None:
        # parts (coarser cut) then subparts (finer cut); this matches the
        # 4-level objbg -> parts -> subparts -> z hierarchy consumed by training.
        h_cuts = [0.65, 0.35]
    if len(h_cuts) < 1:
        raise ValueError(f"Expected at least one hierarchy cut level, got {len(h_cuts)}")

    c, h, w = z_chw.shape
    z_np = (
        z_chw.detach()
        .cpu()
        .permute(1, 2, 0)
        .reshape(h * w, c)
        .float()
        .numpy()
    )
    coords = build_coords(h, w)

    if metric == "cosine":
        nrm = np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-8
        feats = z_np / nrm
    else:
        feats = z_np

    obj_mask_flat = split_object_bg(feats, coords, alpha=alpha_obj_kmeans, seed=0)
    objbg_ids_hw = np.where(obj_mask_flat.reshape(h, w), 1, 0).astype(np.int64)

    idx_obj = np.where(obj_mask_flat)[0]
    level_labs_hw = [np.zeros((h, w), dtype=np.int64) for _ in h_cuts]

    if idx_obj.size == 1:
        for level_idx in range(len(level_labs_hw)):
            level_flat = np.zeros((h * w,), dtype=np.int64)
            level_flat[idx_obj] = 1
            level_labs_hw[level_idx] = level_flat.reshape(h, w)
    elif idx_obj.size >= 2:
        feats_obj = feats[idx_obj]
        coords_obj = coords[idx_obj]
        level_labs = global_labels_with_normalized_cuts(
            feats_obj,
            coords_obj,
            h_cuts=h_cuts,
            metric=metric,
            lam_spatial=lambda_spatial,
            linkage_method=linkage_method,
            cut_mode=cut_mode,
        )

        for level_idx, labs in enumerate(level_labs):
            level_flat = np.zeros((h * w,), dtype=np.int64)
            level_flat[idx_obj] = labs + 1
            level_labs_hw[level_idx] = level_flat.reshape(h, w)

    objbg_ids = torch.from_numpy(objbg_ids_hw)
    level_ids = [torch.from_numpy(level_map) for level_map in level_labs_hw]
    return objbg_ids, level_ids


def load_single_image(image_path: str, img_size: int, center_crop: bool = True) -> torch.Tensor:
    """Load one image and preprocess like the ImageNet loader."""
    img = Image.open(image_path).convert("RGB")
    resize = T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC)
    crop = (
        T.CenterCrop(img_size)
        if center_crop
        else T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)
    )
    transform = T.Compose([resize, crop, T.ToTensor()])
    return transform(img).unsqueeze(0)


def _cast_z_dtype(z_chw: torch.Tensor, z_dtype: str) -> torch.Tensor:
    if z_dtype == "float16":
        return z_chw.to(dtype=torch.float16)
    if z_dtype == "bfloat16":
        return z_chw.to(dtype=torch.bfloat16)
    if z_dtype == "float32":
        return z_chw.to(dtype=torch.float32)
    raise ValueError(f"Unsupported z_dtype: {z_dtype}")


def _torch_load_trusted(obj):
    """Load trusted torch payloads across PyTorch versions."""
    try:
        return torch.load(obj, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(obj, map_location="cpu")


def _resolve_device(save_cfg: dict) -> str:
    device = str(save_cfg.get("device", "") or "").strip()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_torch_runtime(save_cfg: dict, device: str) -> None:
    if not str(device).startswith("cuda"):
        return

    if bool(save_cfg.get("tf32", True)) and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        torch.backends.cudnn.benchmark = bool(save_cfg.get("cudnn_benchmark", True))
    except Exception:
        pass


def _get_encoder_amp_dtype(save_cfg: dict, device: str):
    if not str(device).startswith("cuda"):
        return None
    amp = str(save_cfg.get("encoder_amp_dtype", "none") or "none").strip().lower()
    if amp in {"", "none", "off", "false", "0"}:
        return None
    if amp in {"float16", "fp16", "half"}:
        return torch.float16
    if amp in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported encoder_amp_dtype: {amp}")


def _autocast_ctx_for(device: str, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=str(device).split(":", 1)[0], dtype=amp_dtype)


def _save_pt_file(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _archive_pt_dir_to_tar(
    class_dir: Path, out_path: Path, keep_pt: bool = False
) -> Path | None:
    pt_files = sorted(class_dir.glob("*.pt"))
    if not pt_files:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, mode="w:") as tar:
        for p in pt_files:
            tar.add(p, arcname=p.name)

    if not keep_pt:
        for p in pt_files:
            p.unlink()
        try:
            class_dir.rmdir()
        except OSError:
            pass
    return out_path


def _iter_tar_manifest_entries(tar_path: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(tar_path, mode="r:") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".pt"):
                continue
            member_name = str(m.name)
            if "/" in member_name:
                wnid = member_name.split("/", 1)[0] or "unknown"
            else:
                wnid = tar_path.stem or "unknown"
            rows.append(
                {
                    "kind": "tar",
                    "path": str(tar_path.resolve()),
                    "member": member_name,
                    "wnid": wnid,
                    "offset": int(m.offset_data),
                    "size": int(m.size),
                }
            )
    return rows


def _iter_tar_manifest_entries_path(tar_path_str: str) -> list[dict]:
    """Pool-friendly wrapper: takes a path string so it pickles cheaply."""
    return _iter_tar_manifest_entries(Path(tar_path_str))


def _write_latent_sample_index(
    split_root: Path,
    tar_paths: list[Path] | None = None,
    workers: int | None = None,
) -> None:
    if tar_paths is None:
        iter_paths = sorted(split_root.glob("*.tar"))
    else:
        iter_paths = sorted({p.resolve() for p in tar_paths})

    if not iter_paths:
        return

    # Reading tar headers from lustre serially is dominated by seek latency.
    # With one process per shard we parallelize the lustre reads — the actual
    # TarInfo parsing is trivial. Each spawned worker costs ~2-3s to import
    # torch/scipy/etc, so we only parallelize once there's enough work to
    # amortize that startup. With ~0.2s of serial work per shard, a 16-worker
    # pool needs >~25 shards to come out ahead of serial.
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 1)))
    # Don't spawn more workers than shards — extras just pay spawn cost.
    workers = min(workers, len(iter_paths))

    if workers <= 1 or len(iter_paths) < 32:
        all_rows = [_iter_tar_manifest_entries(p) for p in iter_paths]
    else:
        path_strs = [str(p) for p in iter_paths]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            # chunksize=1 — each shard is hundreds of ms; chunking would
            # serialize multiple shards per IPC roundtrip needlessly.
            all_rows = list(pool.map(_iter_tar_manifest_entries_path, path_strs, chunksize=1))

    samples: list[dict] = [row for chunk in all_rows for row in chunk]
    if not samples:
        return

    manifest = {"samples": samples}
    with open(split_root / ".latent_sample_index.pkl", "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)


def _write_class_map_json(split_root: Path, wnid_to_label: dict[str, int]) -> None:
    if not wnid_to_label:
        return
    ordered = dict(sorted(((k, int(v)) for k, v in wnid_to_label.items()), key=lambda kv: kv[1]))
    with open(split_root / "class_map.json", "w") as f:
        json.dump(ordered, f, indent=2)


def _rebuild_class_map_for_tar(tar_path_str: str) -> dict[str, int]:
    """Pool worker: scan one tar, return wnid->label.

    Reads at most one payload per wnid via the in-shard short-circuit, which
    keeps the per-tar cost ~O(unique wnids in shard) instead of O(members).
    """
    out: dict[str, int] = {}
    tar_path = Path(tar_path_str)
    with tarfile.open(tar_path, mode="r:") as tf:
        for m in tf:
            if not m.isfile() or not str(m.name).endswith(".pt"):
                continue
            member_name = str(m.name)
            wnid_guess = (
                member_name.split("/", 1)[0]
                if "/" in member_name
                else (tar_path.stem or "unknown")
            )
            if wnid_guess in out:
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            payload = _torch_load_trusted(io.BytesIO(fh.read()))
            if not isinstance(payload, dict):
                continue
            meta_v = payload.get("meta", {})
            meta = meta_v if isinstance(meta_v, dict) else {}
            wnid = str(meta.get("wnid") or wnid_guess)
            label = payload.get("label", None)
            if label is None:
                continue
            out[wnid] = int(label)
    return out


def _rebuild_class_map_from_tars(
    split_root: Path, workers: int | None = None
) -> dict[str, int]:
    """Rebuild wnid->label mapping by reading labels from tar members.

    Parallelized across shards because each tar.extractfile() pulls a full
    400KB payload off lustre — serially this dominates --finalize_only wall
    time on ImageNet-scale outputs.
    """
    tar_paths = sorted(split_root.glob("*.tar"))
    if not tar_paths:
        return {}

    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 1)))
    workers = min(workers, len(tar_paths))

    # Same heuristic as _write_latent_sample_index: spawn cost (importing
    # torch/scipy/etc per worker, ~2-3s) dominates below ~32 shards of work.
    if workers <= 1 or len(tar_paths) < 32:
        chunks = [_rebuild_class_map_for_tar(str(p)) for p in tar_paths]
    else:
        path_strs = [str(p) for p in tar_paths]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            chunks = list(pool.map(_rebuild_class_map_for_tar, path_strs, chunksize=1))

    wnid_to_label: dict[str, int] = {}
    for chunk in chunks:
        for wnid, label in chunk.items():
            prev = wnid_to_label.get(wnid)
            if prev is None:
                wnid_to_label[wnid] = label
            elif prev != label:
                raise RuntimeError(
                    f"Inconsistent labels while rebuilding class_map for wnid={wnid}: {prev} vs {label}"
                )
    return wnid_to_label


def finalize_existing_export(split_root: Path, save_cfg: dict | None = None) -> None:
    """Rebuild manifest (and optionally class_map) from existing tar shards."""
    save_cfg = save_cfg or {}
    split_root = Path(split_root)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")
    manifest_workers = save_cfg.get("manifest_workers")
    if manifest_workers is not None:
        manifest_workers = int(manifest_workers)
    _write_latent_sample_index(split_root, workers=manifest_workers)
    if bool(save_cfg.get("write_class_map_json", True)):
        wnid_to_label = _rebuild_class_map_from_tars(split_root, workers=manifest_workers)
        if wnid_to_label:
            _write_class_map_json(split_root, wnid_to_label)


def _iter_tmp_pt_samples(tmp_root: Path):
    for class_dir in sorted(p for p in tmp_root.iterdir() if p.is_dir()):
        wnid = class_dir.name
        for pt_path in sorted(class_dir.glob("*.pt")):
            yield wnid, class_dir, pt_path


def _archive_all_tmp_dirs(root: Path, save_cfg: dict) -> list[Path]:
    archive_tmp = save_cfg.get("archive_tmp", "_pt_tmp")
    keep_pt = bool(save_cfg.get("archive_keep_pt", False))
    tmp_root = root / archive_tmp
    if not tmp_root.exists():
        return []

    tar_paths: list[Path] = []
    for class_dir in sorted(p for p in tmp_root.iterdir() if p.is_dir()):
        wnid = class_dir.name
        out_path = root / f"{wnid}.tar"
        tar_path = _archive_pt_dir_to_tar(class_dir=class_dir, out_path=out_path, keep_pt=keep_pt)
        if tar_path is not None:
            tar_paths.append(tar_path)

    if not keep_pt:
        try:
            tmp_root.rmdir()
        except OSError:
            pass
    return tar_paths


def _archive_tmp_dirs_to_shards(root: Path, save_cfg: dict) -> list[Path]:
    archive_tmp = save_cfg.get("archive_tmp", "_pt_tmp")
    keep_pt = bool(save_cfg.get("archive_keep_pt", False))
    samples_per_tar = max(1, int(save_cfg.get("samples_per_tar", 1000)))
    tmp_root = root / archive_tmp
    if not tmp_root.exists():
        return []

    shard_prefix = str(save_cfg.get("shard_prefix") or root.name or "shard")
    tar_paths: list[Path] = []
    tar = None
    shard_idx = 0
    shard_count = 0

    def _open_next_shard(idx: int):
        out_path = root / f"{shard_prefix}_{idx:05d}.tar"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tar_fh = tarfile.open(out_path, mode="w:")
        tar_paths.append(out_path)
        return tar_fh

    try:
        for wnid, class_dir, pt_path in _iter_tmp_pt_samples(tmp_root):
            if tar is None or shard_count >= samples_per_tar:
                if tar is not None:
                    tar.close()
                tar = _open_next_shard(shard_idx)
                shard_idx += 1
                shard_count = 0

            # Use wnid-prefixed member names so manifest can recover true class ids from shards.
            tar.add(pt_path, arcname=f"{wnid}/{pt_path.name}")
            shard_count += 1

            if not keep_pt:
                pt_path.unlink()
                if not any(class_dir.iterdir()):
                    try:
                        class_dir.rmdir()
                    except OSError:
                        pass
    finally:
        if tar is not None:
            tar.close()

    if not keep_pt:
        try:
            tmp_root.rmdir()
        except OSError:
            pass
    return tar_paths


class _DirectTarWriter:
    """Write per-image payloads directly into rotating tar shards.

    Replaces the original (.pt-per-image -> tmp dir -> end-of-run rollup) path
    in sharded mode. Lustre caps small-file metadata ops at ~700/s, so writing
    + reading + unlinking 1.28M .pt files added 5+ hours on top of encoding.
    Here workers return serialized payload bytes and this writer appends them
    straight into the open tar shard from the main thread — no intermediate
    files, only one open file per shard, all sequential.

    Output layout matches `_archive_tmp_dirs_to_shards`:
      <root>/<shard_prefix>_<idx:05d>.tar with member names <wnid>/<stem>.pt.
    """

    def __init__(self, split_root: Path, shard_prefix: str, samples_per_tar: int):
        self.split_root = split_root
        self.shard_prefix = shard_prefix
        self.samples_per_tar = max(1, int(samples_per_tar))
        self.shard_idx = 0
        self.tar: tarfile.TarFile | None = None
        self.shard_count = 0
        self.created_tar_paths: list[Path] = []

    def _open_next_shard(self) -> None:
        out_path = self.split_root / f"{self.shard_prefix}_{self.shard_idx:05d}.tar"
        if out_path.exists():
            raise RuntimeError(
                f"Shard already exists: {out_path}. Remove stale shards before re-running."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.tar = tarfile.open(out_path, mode="w:")
        self.created_tar_paths.append(out_path)
        self.shard_idx += 1
        self.shard_count = 0

    def add(self, wnid: str, name: str, data: bytes) -> None:
        """Append one serialized payload to the current shard (rotating if full)."""
        if self.tar is None or self.shard_count >= self.samples_per_tar:
            if self.tar is not None:
                self.tar.close()
                self.tar = None
            self._open_next_shard()
        assert self.tar is not None
        info = tarfile.TarInfo(name=f"{wnid}/{name}")
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        self.shard_count += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


class _AsyncTarWriter:
    """Drives a _DirectTarWriter from a background thread.

    The main encoder loop's per-image cost includes a tar.addfile() call which
    blocks on lustre I/O. Moving the actual tar write to a worker thread lets
    the main loop continue submitting/draining work (and the GPU keep
    forwarding) while the lustre write happens in parallel.

    Thread safety: only the worker thread touches the underlying tarfile, so
    no locking needed on the _DirectTarWriter. The main thread only enqueues
    (wnid, name, bytes) tuples via add().

    Backpressure: queue is bounded so a slow writer applies backpressure to
    the main loop (it'll block on enqueue), which is the desired behavior.

    Exceptions in the writer thread are stashed and re-raised on the next
    add() or close() call, so failures are not silently swallowed.
    """

    _STOP = object()  # sentinel passed via queue to signal shutdown

    def __init__(self, writer: _DirectTarWriter, max_queue: int = 512):
        self.writer = writer
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="tar-writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is self._STOP:
                    return
                wnid, name, data = item
                self.writer.add(wnid, name, data)
        except BaseException as e:  # noqa: BLE001 — propagate everything
            self._exc = e
            # Drain remaining queue items so the producer doesn't block forever
            # waiting on a full queue after we've died.
            while True:
                try:
                    item = self.queue.get_nowait()
                    if item is self._STOP:
                        return
                except queue.Empty:
                    return

    def add(self, wnid: str, name: str, data: bytes) -> None:
        if self._exc is not None:
            raise self._exc
        self.queue.put((wnid, name, data))

    def close(self) -> None:
        self.queue.put(self._STOP)
        self._thread.join()
        if self._exc is not None:
            raise self._exc
        self.writer.close()

    @property
    def created_tar_paths(self) -> list[Path]:
        return self.writer.created_tar_paths


def _finalize_tar_export(
    root: Path,
    save_cfg: dict,
    wnid_to_label: dict[str, int] | None = None,
    precreated_tar_paths: list[Path] | None = None,
    streamed: bool = False,
) -> list[Path]:
    archive_mode = str(save_cfg.get("archive_mode", "sharded")).lower()
    created_tar_paths = list(precreated_tar_paths or [])

    if archive_mode == "class":
        created_tar_paths.extend(_archive_all_tmp_dirs(root, save_cfg))
    elif archive_mode == "sharded":
        # When streaming has already drained tmp -> shards during encoding, skip
        # the legacy end-of-run rollup (it would re-open shard_idx=0 and clobber
        # the streamed shards if any stale .pt files were left over).
        if not streamed:
            created_tar_paths.extend(_archive_tmp_dirs_to_shards(root, save_cfg))
    else:
        raise ValueError(f"Unsupported archive_mode: {archive_mode}")

    if bool(save_cfg.get("write_manifest_pkl", True)):
        manifest_workers = save_cfg.get("manifest_workers")
        if manifest_workers is not None:
            manifest_workers = int(manifest_workers)
        _write_latent_sample_index(
            root, tar_paths=created_tar_paths or None, workers=manifest_workers
        )
    if bool(save_cfg.get("write_class_map_json", True)) and wnid_to_label is not None:
        _write_class_map_json(root, wnid_to_label)
    return created_tar_paths


def save_triplet(
    root: Path,
    wnid: str,
    filename: str,
    z_bchw: torch.Tensor,  # [1,C,H,W]
    save_cfg: dict | None = None,
    label: int | None = None,
) -> None:
    """Save a minimal latent payload consumable by pMF LatentHierDataset."""
    save_cfg = save_cfg or {}
    archive = save_cfg.get("archive", "tar")
    archive_tmp = save_cfg.get("archive_tmp", "_pt_tmp")
    z_dtype = save_cfg.get("z_dtype", "float16")

    stem, _ext = os.path.splitext(filename)
    z_chw = z_bchw[0]

    # 4-level hierarchy consumed by training: objbg -> parts -> subparts -> z.
    # h_part is the coarser cut (parts), h_sub the finer cut (subparts); both
    # follow the repo's part/subpart defaults and can be overridden via save_cfg.
    h_part = float(save_cfg.get("h_part", 0.65))
    h_sub = float(save_cfg.get("h_sub", 0.35))
    objbg_ids, level_ids = compute_hierarchy_from_z_chw(
        z_chw,
        metric="cosine",
        lambda_spatial=0.2,
        linkage_method="average",
        cut_mode="height",
        h_cuts=[h_part, h_sub],
        alpha_obj_kmeans=0.0,
    )
    parts_ids, subparts_ids = level_ids[0], level_ids[1]

    out_dict: dict = {
        "z": _cast_z_dtype(z_chw.cpu(), z_dtype),  # [C,H,W] (level 3)
        "objbg_ids": objbg_ids.to(torch.int16),  # [H,W] obj/bg (level 0)
        "parts_ids": parts_ids.to(torch.int16),  # [H,W] parts (level 1)
        "subparts_ids_global": subparts_ids.to(torch.int16),  # [H,W] subparts (level 2)
        "meta": {
            "stem": stem,
            "wnid": wnid,
            "shape": list(z_chw.shape),
        },
    }
    if label is not None:
        out_dict["label"] = int(label)

    if archive == "tar":
        tmp_root = root / archive_tmp
        _save_pt_file(out_dict, tmp_root / wnid / f"{stem}.pt")
    else:
        _save_pt_file(out_dict, root / wnid / f"{stem}.pt")


def _save_triplet_worker(
    root_str: str,
    wnid: str,
    filename: str,
    z_bchw: torch.Tensor,
    save_cfg: dict,
    label: int,
) -> None:
    """Process-pool worker: compute the per-image hierarchy and write the .pt file.

    Module-level so it pickles cleanly under the 'spawn' start method.
    """
    save_triplet(
        root=Path(root_str),
        wnid=wnid,
        filename=filename,
        z_bchw=z_bchw,
        save_cfg=save_cfg,
        label=label,
    )


def build_triplet_payload(
    z_bchw: torch.Tensor,
    wnid: str,
    filename: str,
    save_cfg: dict | None = None,
    label: int | None = None,
) -> tuple[str, str, bytes]:
    """Compute the hierarchy and serialize the per-image payload to bytes.

    Returns (wnid, "<stem>.pt", serialized_bytes) so the caller can append
    directly into a tar shard without ever materializing a .pt file on disk.
    Mirrors `save_triplet` in terms of dict contents.
    """
    save_cfg = save_cfg or {}
    z_dtype = save_cfg.get("z_dtype", "float16")

    stem, _ext = os.path.splitext(filename)
    z_chw = z_bchw[0]

    h_part = float(save_cfg.get("h_part", 0.65))
    h_sub = float(save_cfg.get("h_sub", 0.35))
    objbg_ids, level_ids = compute_hierarchy_from_z_chw(
        z_chw,
        metric="cosine",
        lambda_spatial=0.2,
        linkage_method="average",
        cut_mode="height",
        h_cuts=[h_part, h_sub],
        alpha_obj_kmeans=0.0,
    )
    parts_ids, subparts_ids = level_ids[0], level_ids[1]

    out_dict: dict = {
        "z": _cast_z_dtype(z_chw.cpu(), z_dtype),
        "objbg_ids": objbg_ids.to(torch.int16),
        "parts_ids": parts_ids.to(torch.int16),
        "subparts_ids_global": subparts_ids.to(torch.int16),
        "meta": {
            "stem": stem,
            "wnid": wnid,
            "shape": list(z_chw.shape),
        },
    }
    if label is not None:
        out_dict["label"] = int(label)

    buf = io.BytesIO()
    torch.save(out_dict, buf)
    return wnid, f"{stem}.pt", buf.getvalue()


def _build_triplet_payload_worker(
    z_bchw: torch.Tensor,
    wnid: str,
    filename: str,
    save_cfg: dict,
    label: int,
) -> tuple[str, str, bytes]:
    """Process-pool worker for the direct-to-tar path.

    Module-level so it pickles cleanly under the 'spawn' start method.
    """
    return build_triplet_payload(z_bchw, wnid, filename, save_cfg, label)


def build_encoder_from_cfg(encoder_cfg: dict) -> torch.nn.Module:
    """Construct the project's EncoderOnly wrapper from config."""
    return EncoderOnly(
        encoder_cls=encoder_cfg["cls"],
        encoder_config_path=encoder_cfg["config_path"],
        encoder_input_size=encoder_cfg["input_size"],
        encoder_params=encoder_cfg["params"],
        reshape_to_2d=True,
        normalization_stat_path=encoder_cfg.get("normalization_stat_path") or None,
    )


def _extract_meta_field(metas, idx: int, key: str):
    if isinstance(metas, dict):
        return metas[key][idx]
    return metas[idx][key]


def _extract_label(targets, idx: int) -> int:
    if torch.is_tensor(targets):
        return int(targets[idx].item())
    return int(targets[idx])


def run(
    split: str,
    base_dir: str,
    out_root: str,
    encoder_cfg: dict,
    dataloader_cfg: dict,
    save_cfg: dict | None = None,
    max_batches: int = -1,
) -> None:
    """Encode a dataset split and export minimal latent payloads."""
    save_cfg = dict(save_cfg or {})
    device = _resolve_device(save_cfg)
    _configure_torch_runtime(save_cfg, device)
    amp_dtype = _get_encoder_amp_dtype(save_cfg, device)
    world_size = int(save_cfg.get("world_size", 1))
    process_rank = int(save_cfg.get("process_rank", 0))
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if process_rank < 0 or process_rank >= world_size:
        raise ValueError(
            f"process_rank must be in [0, {world_size - 1}], got {process_rank}"
        )
    if world_size > 1 and bool(dataloader_cfg.get("shuffle", False)):
        raise ValueError(
            "dataloader.shuffle=True with world_size>1 can cause overlap/missing samples because "
            "batch-index sharding assumes identical batch order across ranks. Set shuffle=False."
        )

    loader = make_imagenet_loader(
        split=split,
        base_dir=base_dir,
        bs=dataloader_cfg.get("batch_size", 8),
        workers=dataloader_cfg.get("workers", 8),
        image_size=dataloader_cfg.get("image_size", encoder_cfg["input_size"]),
        shuffle=dataloader_cfg.get("shuffle", False),
        imagenet100_json=dataloader_cfg.get("imagenetsmall"),
        rank=process_rank,
        world_size=world_size,
    )

    enc = build_encoder_from_cfg(encoder_cfg).to(device).eval()

    split_root = Path(out_root) / split
    split_root.mkdir(parents=True, exist_ok=True)

    processed_batches = 0
    wnid_to_label: dict[str, int] = {}
    archive_mode = str(save_cfg.get("archive_mode", "sharded")).lower()
    archive_on_class_change = bool(save_cfg.get("archive_on_class_change", False))
    # cluster_workers: -1 (or missing) => auto, 0 => serial, >0 => explicit.
    raw_cw = int(save_cfg.get("cluster_workers", -1))
    if raw_cw < 0:
        cpu_total = os.cpu_count() or 1
        # Cap so we don't oversubscribe on huge nodes; per-image work is short
        # enough that IPC overhead dominates past ~16 workers per rank.
        n_cluster_workers = max(1, min(16, cpu_total // max(world_size, 1)))
    else:
        n_cluster_workers = raw_cw
    print(
        f"[encode] cluster_workers={n_cluster_workers} "
        f"(raw={raw_cw}, world_size={world_size}, cpu_count={os.cpu_count()})",
        flush=True,
    )
    pool: ProcessPoolExecutor | None = None
    pending: deque[Future] = deque()
    max_pending = 0
    if n_cluster_workers > 0:
        pool = ProcessPoolExecutor(
            max_workers=n_cluster_workers,
            mp_context=mp.get_context("spawn"),
        )
        max_pending = 4 * n_cluster_workers

    # `direct_writer` is set below if we're in archive=tar + sharded mode. When
    # active, workers return serialized payload bytes (via _build_triplet_payload_worker)
    # and the main thread appends them straight into rotating tar shards — no
    # intermediate .pt files. `_drain_to` routes future results to the writer.
    # `_AsyncTarWriter` moves the actual tar.addfile() lustre I/O to a worker
    # thread so the main loop doesn't block on it.
    direct_writer: _DirectTarWriter | _AsyncTarWriter | None = None

    def _drain_to(target_size: int) -> None:
        # FIFO drain so worker exceptions surface in submission order.
        while len(pending) > target_size:
            result = pending.popleft().result()
            if direct_writer is not None and result is not None:
                direct_writer.add(*result)

    if world_size > 1 and save_cfg.get("archive") == "tar" and archive_mode != "sharded":
        raise ValueError(
            "Multi-process export with shared output requires archive_mode='sharded'."
        )
    if save_cfg.get("archive") == "tar" and archive_mode == "sharded" and archive_on_class_change:
        raise ValueError(
            "archive_on_class_change=True is incompatible with archive_mode='sharded'. "
            "Use archive_mode='class' or disable archive_on_class_change."
        )
    if world_size > 1 and save_cfg.get("archive") == "tar" and archive_mode == "sharded":
        base_prefix = str(save_cfg.get("shard_prefix") or split_root.name or "shard")
        save_cfg["shard_prefix"] = f"{base_prefix}_r{process_rank}"
        base_tmp = str(save_cfg.get("archive_tmp", "_pt_tmp"))
        save_cfg["archive_tmp"] = f"{base_tmp}_r{process_rank}"
        # Avoid races and incomplete metadata writes. Run --finalize_only once after all ranks finish.
        save_cfg["write_manifest_pkl"] = False
        save_cfg["write_class_map_json"] = False
    last_wnid: str | None = None
    created_tar_paths: list[Path] = []

    # Direct-to-tar: in sharded mode, workers serialize each per-image payload
    # to bytes and the main thread appends them straight into rotating tar
    # shards. Replaces the older write-.pt-then-rollup path that hit lustre's
    # ~700/s small-file metadata cap with 1.28M files, adding 5+ hours of
    # finalize after a 1h encode. Class mode still uses the .pt+_pt_tmp path
    # because it tars per-class via archive_on_class_change.
    samples_per_tar_int = max(1, int(save_cfg.get("samples_per_tar", 1000)))
    if save_cfg.get("archive") == "tar" and archive_mode == "sharded":
        direct_prefix = str(save_cfg.get("shard_prefix") or split_root.name or "shard")
        direct_writer = _AsyncTarWriter(
            _DirectTarWriter(split_root, direct_prefix, samples_per_tar_int)
        )

    # Progress bar: count *this rank's* batches. With DistributedSampler the
    # loader already only yields this rank's batches, so len(loader) is the
    # per-rank count directly.
    try:
        total_loader_batches = len(loader)
    except TypeError:
        total_loader_batches = None
    if total_loader_batches is not None:
        my_total_batches = total_loader_batches
        if max_batches is not None and max_batches > 0:
            my_total_batches = min(my_total_batches, max_batches)
    else:
        my_total_batches = max_batches if (max_batches and max_batches > 0) else None
    pbar = tqdm(
        total=my_total_batches,
        unit="batch",
        desc=f"encode r{process_rank}/{world_size}",
        smoothing=0.1,
        mininterval=2.0,
        dynamic_ncols=True,
    )

    try:
        with torch.inference_mode():
            for b_idx, (imgs, targets, metas) in enumerate(loader):
                # Multi-rank sharding now happens at the sampler level
                # (DistributedSampler), so every batch the loader yields is
                # for this rank — no more `b_idx % world_size` skip needed.
                imgs = imgs.to(device, non_blocking=True)
                with _autocast_ctx_for(device, amp_dtype):
                    z = enc(imgs)  # [B,C,H,W]
                z_cpu = z.detach().cpu()

                for i in range(imgs.size(0)):
                    wnid = str(_extract_meta_field(metas, i, "wnid") or "unknown")
                    filename = str(_extract_meta_field(metas, i, "filename"))
                    label_i = _extract_label(targets, i)

                    prev = wnid_to_label.get(wnid)
                    if prev is None:
                        wnid_to_label[wnid] = label_i
                    elif prev != label_i:
                        raise RuntimeError(
                            f"Inconsistent label for wnid={wnid}: saw {prev} and {label_i}"
                        )

                    if (
                        save_cfg.get("archive") == "tar"
                        and archive_mode == "class"
                        and archive_on_class_change
                    ):
                        if last_wnid is None:
                            last_wnid = wnid
                        elif wnid != last_wnid:
                            # Flush in-flight writes for the closing class before tarring it.
                            _drain_to(0)
                            archive_tmp = save_cfg.get("archive_tmp", "_pt_tmp")
                            keep_pt = bool(save_cfg.get("archive_keep_pt", False))
                            tmp_root = split_root / archive_tmp
                            class_dir = tmp_root / last_wnid
                            out_path = split_root / f"{last_wnid}.tar"
                            if out_path.exists():
                                raise RuntimeError(
                                    f"Archive already exists for wnid={last_wnid}. "
                                    "Classes may not be contiguous in the loader."
                                )
                            tar_path = _archive_pt_dir_to_tar(
                                class_dir=class_dir, out_path=out_path, keep_pt=keep_pt
                            )
                            if tar_path is not None:
                                created_tar_paths.append(tar_path)
                            last_wnid = wnid

                    if pool is not None:
                        # clone() so the submitted tensor owns its own storage —
                        # pickling a view would otherwise serialize the full batch.
                        z_slice = z_cpu[i : i + 1].clone()
                        if direct_writer is not None:
                            pending.append(
                                pool.submit(
                                    _build_triplet_payload_worker,
                                    z_slice,
                                    wnid,
                                    filename,
                                    save_cfg,
                                    label_i,
                                )
                            )
                        else:
                            pending.append(
                                pool.submit(
                                    _save_triplet_worker,
                                    str(split_root),
                                    wnid,
                                    filename,
                                    z_slice,
                                    save_cfg,
                                    label_i,
                                )
                            )
                        _drain_to(max_pending)
                    elif direct_writer is not None:
                        result = build_triplet_payload(
                            z_cpu[i : i + 1], wnid, filename, save_cfg, label_i
                        )
                        direct_writer.add(*result)
                    else:
                        save_triplet(
                            root=split_root,
                            wnid=wnid,
                            filename=filename,
                            z_bchw=z_cpu[i : i + 1],
                            save_cfg=save_cfg,
                            label=label_i,
                        )

                processed_batches += 1
                pbar.update(1)
                if pool is not None:
                    pbar.set_postfix(in_flight=len(pending), refresh=False)

                if max_batches is not None and max_batches > 0 and processed_batches >= max_batches:
                    break

        # Drain remaining workers; their results land in the open tar shard.
        pbar.set_description(f"encode r{process_rank}/{world_size} (flushing pool)")
        _drain_to(0)
    finally:
        pbar.close()
        if pool is not None:
            pool.shutdown(wait=True)
        if direct_writer is not None:
            direct_writer.close()

    if save_cfg.get("archive") == "tar":
        if archive_mode == "class" and archive_on_class_change and last_wnid is not None:
            archive_tmp = save_cfg.get("archive_tmp", "_pt_tmp")
            keep_pt = bool(save_cfg.get("archive_keep_pt", False))
            tmp_root = split_root / archive_tmp
            class_dir = tmp_root / last_wnid
            out_path = split_root / f"{last_wnid}.tar"
            if out_path.exists():
                raise RuntimeError(
                    f"Archive already exists for wnid={last_wnid}. "
                    "Classes may not be contiguous in the loader."
                )
            tar_path = _archive_pt_dir_to_tar(class_dir=class_dir, out_path=out_path, keep_pt=keep_pt)
            if tar_path is not None:
                created_tar_paths.append(tar_path)
            if not keep_pt:
                try:
                    tmp_root.rmdir()
                except OSError:
                    pass

        if direct_writer is not None:
            created_tar_paths.extend(direct_writer.created_tar_paths)

        _finalize_tar_export(
            split_root,
            save_cfg,
            wnid_to_label=wnid_to_label,
            precreated_tar_paths=created_tar_paths,
            streamed=direct_writer is not None,
        )
    else:
        if bool(save_cfg.get("write_class_map_json", True)):
            _write_class_map_json(split_root, wnid_to_label)


def load_cfg(cfg_path: str) -> dict:
    """Load yaml config and flatten fields used by this script."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "encoder_type" not in cfg:
        raise ValueError("`encoder_type` must be specified in config.yaml")

    etype = cfg["encoder_type"]
    encoders = cfg.get("encoders", {})
    if etype not in encoders:
        raise ValueError(f"encoder_type '{etype}' not found under `encoders` in {cfg_path}")
    encoder_cfg = encoders[etype]

    dl = cfg.get("dataloader", {}) or {}
    if dl.get("image_size") is None:
        dl["image_size"] = encoder_cfg["input_size"]

    save = cfg.get("save", {}) or {}
    flat = {
        "base_dir": cfg["base_dir"],
        "out_root": cfg["out_root"],
        "split": cfg.get("split", "train"),
        "max_batches": cfg.get("max_batches", -1),
        "dataloader": {
            "batch_size": dl.get("batch_size", 64),
            "workers": dl.get("workers", 8),
            "shuffle": dl.get("shuffle", False),
            "image_size": dl["image_size"],
            "imagenetsmall": dl.get("imagenetsmall", None),
        },
        "encoder_cfg": {
            "cls": encoder_cfg["cls"],
            "config_path": encoder_cfg["config_path"],
            "input_size": encoder_cfg["input_size"],
            "params": encoder_cfg["params"],
        },
        "save_cfg": {
            # Minimal pMF-consumable export by default.
            "archive": save.get("archive", "tar"),
            "archive_mode": save.get("archive_mode", "sharded"),
            "archive_tmp": save.get("archive_tmp", "_pt_tmp"),
            "archive_keep_pt": save.get("archive_keep_pt", False),
            "archive_on_class_change": save.get("archive_on_class_change", False),
            "samples_per_tar": int(save.get("samples_per_tar", 1000)),
            "shard_prefix": save.get("shard_prefix", None),
            "z_dtype": save.get("z_dtype", "float16"),
            # Runtime/performance knobs (kept here to avoid another config section).
            "device": save.get("device", ""),
            "encoder_amp_dtype": save.get("encoder_amp_dtype", "none"),
            "tf32": save.get("tf32", True),
            "cudnn_benchmark": save.get("cudnn_benchmark", True),
            "process_rank": int(save.get("process_rank", 0)),
            "world_size": int(save.get("world_size", 1)),
            # cluster_workers: -1 = auto (cpu_count // world_size, capped),
            # 0 = serial (no pool), >0 = explicit worker count.
            "cluster_workers": int(save.get("cluster_workers", -1)),
            "write_manifest_pkl": save.get("write_manifest_pkl", True),
            "write_class_map_json": save.get("write_class_map_json", True),
        },
    }
    return flat


def run_single_image(
    image_path: str,
    out_root: str,
    encoder_cfg: dict,
    save_cfg: dict | None = None,
) -> None:
    """Encode one image and export a single sample payload."""
    save_cfg = dict(save_cfg or {})
    device = _resolve_device(save_cfg)
    _configure_torch_runtime(save_cfg, device)
    amp_dtype = _get_encoder_amp_dtype(save_cfg, device)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    enc = build_encoder_from_cfg(encoder_cfg).to(device).eval()
    imgs = load_single_image(image_path, encoder_cfg["input_size"]).to(device)

    with torch.inference_mode():
        with _autocast_ctx_for(device, amp_dtype):
            z = enc(imgs)

    img_path = Path(image_path)
    save_triplet(
        root=out_root_path,
        wnid="single",
        filename=img_path.name,
        z_bchw=z.cpu(),
        save_cfg=save_cfg,
        label=0,
    )

    if save_cfg.get("archive") == "tar":
        _finalize_tar_export(out_root_path, save_cfg, wnid_to_label={"single": 0})
    elif bool(save_cfg.get("write_class_map_json", True)):
        _write_class_map_json(out_root_path, {"single": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help="Rebuild .latent_sample_index.pkl (+ class_map.json) from existing .tar files and exit.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional: path to a single image to encode instead of a dataset split.",
    )
    parser.add_argument(
        "--single_out",
        type=str,
        default=None,
        help="Optional output directory for single-image mode (defaults to cfg['out_root']).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override torch device, e.g. cuda:0")
    parser.add_argument("--rank", type=int, default=None, help="Process rank for multi-process export")
    parser.add_argument("--world_size", type=int, default=None, help="Total processes for multi-process export")
    parser.add_argument(
        "--cluster_workers",
        type=int,
        default=None,
        help=(
            "Parallelize the per-image hierarchy compute + .pt write across N CPU workers. "
            "Default -1 = auto (cpu_count // world_size, capped at 16); 0 = serial; >0 = explicit."
        ),
    )
    parser.add_argument(
        "--encoder_amp_dtype",
        type=str,
        default=None,
        choices=["none", "float16", "bfloat16"],
        help="Use autocast for encoder inference on CUDA.",
    )
    parser.add_argument(
        "--shard_prefix",
        type=str,
        default=None,
        help="Optional shard prefix override (useful for multi-process exports).",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    save_cfg = dict(cfg.get("save_cfg", {}))

    if args.device is not None:
        save_cfg["device"] = args.device
    if args.rank is not None:
        save_cfg["process_rank"] = int(args.rank)
    if args.world_size is not None:
        save_cfg["world_size"] = int(args.world_size)
    if args.cluster_workers is not None:
        save_cfg["cluster_workers"] = int(args.cluster_workers)
    if args.encoder_amp_dtype is not None:
        save_cfg["encoder_amp_dtype"] = args.encoder_amp_dtype
    if args.shard_prefix is not None:
        save_cfg["shard_prefix"] = args.shard_prefix

    if args.finalize_only:
        split_root = Path(cfg["out_root"]) / cfg["split"]
        finalize_existing_export(split_root, save_cfg=save_cfg)
        return

    if args.image is not None:
        run_single_image(
            image_path=args.image,
            out_root=args.single_out or cfg["out_root"],
            encoder_cfg=cfg["encoder_cfg"],
            save_cfg=save_cfg,
        )
        return

    run(
        split=cfg["split"],
        base_dir=cfg["base_dir"],
        out_root=cfg["out_root"],
        encoder_cfg=cfg["encoder_cfg"],
        dataloader_cfg=cfg["dataloader"],
        save_cfg=save_cfg,
        max_batches=cfg["max_batches"],
    )


if __name__ == "__main__":
    main()
