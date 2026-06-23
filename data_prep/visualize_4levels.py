"""Encode images with DINOv2; render the 4 pMF levels (objbg/parts/subparts/finest) PCA-RGB side-by-side.

Usage:
    python visualize_4levels.py --image_dir <DIR> [--output <PNG>] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from encoder import EncoderOnly
from imagenet1k_encoder import (
    build_coords,
    load_single_image,
    parts_labels_with_normalized_cuts,
    split_object_bg,
)


ENCODER_KW = dict(
    encoder_cls="Dinov2withNorm",
    encoder_config_path="facebook/dinov2-with-registers-base",
    encoder_input_size=224,
    encoder_params={
        "dinov2_path": "facebook/dinov2-with-registers-base",
        "normalize": True,
    },
    reshape_to_2d=True,
)

# Hierarchy hyperparameters — match `imagenet1k_encoder.parts_labels_with_normalized_cuts` defaults.
METRIC = "cosine"
LAMBDA_SPATIAL = 0.2
LINKAGE_METHOD = "average"
H_PART = 0.65
H_SUB = 0.35
ALPHA_OBJ_KMEANS = 0.0


def fit_pca_3(feats_nc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = feats_nc.mean(axis=0, keepdims=True)
    x = feats_nc - mean
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = min(3, vt.shape[0])
    comps = vt[:k].T
    if k < 3:
        pad = np.zeros((comps.shape[0], 3 - k), dtype=comps.dtype)
        comps = np.concatenate([comps, pad], axis=1)
    return mean, comps


def compute_pca_color_range(z_flat: np.ndarray, mean: np.ndarray, comps: np.ndarray):
    """Per-channel [1%, 99%] range from the raw token projection — reused for all
    levels so colors are comparable and binary-region maps don't collapse to monochrome."""
    proj = (z_flat - mean) @ comps
    los = np.empty(3, dtype=np.float32)
    his = np.empty(3, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(proj[:, c].min()), float(proj[:, c].max())
        if hi <= lo:
            lo, hi = 0.0, 1.0
        los[c] = lo
        his[c] = hi
    return los, his


def project_rgb_with_range(feats_nc, mean, comps, los, his) -> np.ndarray:
    proj = (feats_nc - mean) @ comps
    rgb = np.clip((proj - los) / (his - los), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def avg_per_region(z_hwc: np.ndarray, ids_hw: np.ndarray) -> np.ndarray:
    h, w, c = z_hwc.shape
    flat_z = z_hwc.reshape(h * w, c)
    flat_ids = ids_hw.reshape(h * w)
    out = np.zeros_like(flat_z, dtype=np.float32)
    for rid in np.unique(flat_ids):
        mask = flat_ids == rid
        out[mask] = flat_z[mask].mean(axis=0, keepdims=True)
    return out.reshape(h, w, c)


def four_level_ids(z_hwc: np.ndarray) -> dict[str, np.ndarray]:
    """objbg / parts / subparts / finest id maps following pMF convention."""
    h, w, c = z_hwc.shape
    z_flat = z_hwc.reshape(h * w, c)
    if METRIC == "cosine":
        feats = z_flat / (np.linalg.norm(z_flat, axis=1, keepdims=True) + 1e-8)
    else:
        feats = z_flat
    coords = build_coords(h, w)

    obj_mask = split_object_bg(feats, coords, alpha=ALPHA_OBJ_KMEANS, seed=0)
    objbg_hw = obj_mask.reshape(h, w).astype(np.int64)

    parts_flat = np.zeros((h * w,), dtype=np.int64)
    sub_flat = np.zeros((h * w,), dtype=np.int64)
    idx_obj = np.where(obj_mask)[0]
    if idx_obj.size >= 2:
        labs_part, labs_sub = parts_labels_with_normalized_cuts(
            feats[idx_obj],
            coords[idx_obj],
            metric=METRIC,
            lam_spatial=LAMBDA_SPATIAL,
            linkage_method=LINKAGE_METHOD,
            cut_mode="height",
            h_sub=H_SUB,
            h_part=H_PART,
        )
        parts_flat[idx_obj] = labs_part.astype(np.int64) + 1
        sub_flat[idx_obj] = labs_sub.astype(np.int64) + 1
    elif idx_obj.size == 1:
        parts_flat[idx_obj] = 1
        sub_flat[idx_obj] = 1

    finest_hw = np.arange(h * w, dtype=np.int64).reshape(h, w)

    return {
        "objbg": objbg_hw,
        "parts": parts_flat.reshape(h, w),
        "subparts": sub_flat.reshape(h, w),
        "finest": finest_hw,
    }


def resize_shorter_then_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Match `load_single_image`: Resize shorter side to `size`, then CenterCrop `size`."""
    iw, ih = img.size
    short = min(iw, ih)
    nw, nh = round(iw * size / short), round(ih * size / short)
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def build_image_row(img_pil: Image.Image, z_chw: torch.Tensor, tile_px: int = 224):
    z = z_chw.float().cpu().numpy()
    z_hwc = z.transpose(1, 2, 0)
    h, w, c = z_hwc.shape
    z_flat = z_hwc.reshape(h * w, c)
    pca_mean, pca_comps = fit_pca_3(z_flat)
    los, his = compute_pca_color_range(z_flat, pca_mean, pca_comps)

    levels = four_level_ids(z_hwc)

    orig = np.array(resize_shorter_then_center_crop(img_pil.convert("RGB"), tile_px))
    tiles = [orig]
    for name in ("objbg", "parts", "subparts", "finest"):
        avg = avg_per_region(z_hwc, levels[name])
        rgb_small = project_rgb_with_range(avg.reshape(h * w, c), pca_mean, pca_comps, los, his).reshape(h, w, 3)
        tiles.append(np.array(Image.fromarray(rgb_small).resize((tile_px, tile_px), Image.NEAREST)))
    return tiles


def render_panel(rows: list[list[np.ndarray]], gap: int = 10) -> Image.Image:
    n_cols = max(len(t) for t in rows)
    h, w, _ = rows[0][0].shape
    panel_w = gap + n_cols * w + (n_cols - 1) * gap + gap
    panel_h = gap + (h + gap) * len(rows)
    canvas = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    for ri, tiles in enumerate(rows):
        y = gap + ri * (h + gap)
        for ci, tile in enumerate(tiles):
            x = gap + ci * (w + gap)
            canvas.paste(Image.fromarray(tile, mode="RGB").convert("RGBA"), (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True, type=Path, help="Directory containing input images")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output PNG path (default: <image_dir>/levels_pca.png)")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu | cuda:0")
    args = parser.parse_args()

    image_dir: Path = args.image_dir
    output: Path = args.output if args.output is not None else image_dir / "levels_pca.png"

    enc = EncoderOnly(**ENCODER_KW).to(args.device).eval()

    img_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.resolve() != output.resolve()
    )
    if not img_paths:
        raise SystemExit(f"No images under {image_dir}")
    print(f"Found {len(img_paths)} images: {[p.name for p in img_paths]}")

    rows = []
    for p in img_paths:
        print(f"-> encoding {p.name}")
        x = load_single_image(str(p), ENCODER_KW["encoder_input_size"]).to(args.device)
        with torch.inference_mode():
            z = enc(x)
        rows.append(build_image_row(Image.open(p).convert("RGB"), z[0].detach().cpu()))

    panel = render_panel(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    print(f"Saved: {output}  ({panel.size[0]}x{panel.size[1]})")


if __name__ == "__main__":
    main()
