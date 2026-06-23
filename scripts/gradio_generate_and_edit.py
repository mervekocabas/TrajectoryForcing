#!/usr/bin/env python3
"""Gradio app: generate .pt latents from class numbers, then edit & visualize.

Combines image_generation.py (to create .pt files from class IDs like 288, 305)
with the token-exchange editing pipeline in a single UI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import gradio as gr
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_GEN_SCRIPT = REPO_ROOT / "scripts" / "image_generation.py"
TOKEN_EXCHANGE_SCRIPT = REPO_ROOT / "scripts" / "token_exchange_specific_points.py"
GRADIO_RUNS_DIR = REPO_ROOT / "gradio_runs"
DEFAULT_CONFIG = "configs/eval_config.yml"
DEFAULT_GRID_SIZE = 16
PREVIEW_SCALE = 32
COS_REGION_THRESHOLD = 0.9
FIXED_SEED = 666
FIXED_TILE_PX = 256


# ---------------------------------------------------------------------------
# Utility helpers (shared with gradio_latent_editor.py)
# ---------------------------------------------------------------------------

def _empty_preview(size: int = DEFAULT_GRID_SIZE, scale: int = PREVIEW_SCALE) -> np.ndarray:
    return np.zeros((int(size) * int(scale), int(size) * int(scale), 3), dtype=np.uint8)


def _overlay_points(
    base_img: np.ndarray | None,
    points: Sequence[Tuple[int, int]],
    size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    img = np.asarray(base_img if base_img is not None else _empty_preview(size=size), dtype=np.uint8).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        img = _empty_preview(size=size)

    h, w = int(img.shape[0]), int(img.shape[1])
    cell_h = max(1, h // int(size))
    cell_w = max(1, w // int(size))

    for r in range(int(size) + 1):
        y = min(h - 1, r * cell_h)
        img[y : min(h, y + 1), :, :] = np.array([90, 90, 90], dtype=np.uint8)
    for c in range(int(size) + 1):
        x = min(w - 1, c * cell_w)
        img[:, x : min(w, x + 1), :] = np.array([90, 90, 90], dtype=np.uint8)

    for idx, (r, c) in enumerate(points, start=1):
        rr, cc = int(r), int(c)
        if rr < 0 or rr >= int(size) or cc < 0 or cc >= int(size):
            continue
        y0 = rr * cell_h
        y1 = min(h, (rr + 1) * cell_h)
        x0 = cc * cell_w
        x1 = min(w, (cc + 1) * cell_w)
        border = 2
        color = np.array([255, 80, 80], dtype=np.uint8)
        img[y0 : min(y1, y0 + border), x0:x1] = color
        img[max(y0, y1 - border) : y1, x0:x1] = color
        img[y0:y1, x0 : min(x1, x0 + border)] = color
        img[y0:y1, max(x0, x1 - border) : x1] = color

    return img


def _toggle_point_from_coords(
    points: Sequence[Tuple[int, int]],
    r: int,
    c: int,
    base_img: np.ndarray | None,
    size: int = DEFAULT_GRID_SIZE,
):
    current = [(int(pr), int(pc)) for (pr, pc) in points]
    rr, cc = int(r), int(c)
    if rr < 0 or rr >= int(size) or cc < 0 or cc >= int(size):
        return (
            current,
            _points_to_text(current),
            len(current),
            _overlay_points(base_img, current, size=size),
        )
    pt = (rr, cc)
    if pt in current:
        current.remove(pt)
    else:
        current.append(pt)
    return (
        current,
        _points_to_text(current),
        len(current),
        _overlay_points(base_img, current, size=size),
    )


def _toggle_region_from_coords(
    points: Sequence[Tuple[int, int]],
    r: int,
    c: int,
    base_img: np.ndarray | None,
    region_ids: np.ndarray | None,
    size: int = DEFAULT_GRID_SIZE,
):
    current = [(int(pr), int(pc)) for (pr, pc) in points]
    rr, cc = int(r), int(c)
    if rr < 0 or rr >= int(size) or cc < 0 or cc >= int(size):
        return (
            current,
            _points_to_text(current),
            len(current),
            _overlay_points(base_img, current, size=size),
        )
    ids = None if region_ids is None else np.asarray(region_ids)
    if ids is None or ids.ndim != 2 or ids.shape[0] != int(size) or ids.shape[1] != int(size):
        return _toggle_point_from_coords(current, rr, cc, base_img, size=size)
    target_region = int(ids[rr, cc])
    region_coords = np.argwhere(ids == target_region)
    if region_coords.size == 0:
        return _toggle_point_from_coords(current, rr, cc, base_img, size=size)
    region_points = sorted([(int(y), int(x)) for y, x in region_coords.tolist()])
    current_set = set(current)
    region_set = set(region_points)
    if region_set.issubset(current_set):
        current = [pt for pt in current if pt not in region_set]
    else:
        current.extend([pt for pt in region_points if pt not in current_set])
    return (
        current,
        _points_to_text(current),
        len(current),
        _overlay_points(base_img, current, size=size),
    )


def _points_to_text(points: Sequence[Tuple[int, int]]) -> str:
    return " ".join(f"{int(r)},{int(c)}" for r, c in points)


def _toggle_point_or_region_from_image(
    points: Sequence[Tuple[int, int]],
    base_img: np.ndarray | None,
    region_select_enabled: bool,
    region_ids: np.ndarray | None,
    evt: gr.SelectData,
    size: int = DEFAULT_GRID_SIZE,
):
    idx = getattr(evt, "index", None)
    if not isinstance(idx, (list, tuple)) or len(idx) < 2:
        current = [(int(r), int(c)) for (r, c) in points]
        return (
            current,
            _points_to_text(current),
            len(current),
            _overlay_points(base_img, current, size=size),
        )
    x, y = int(idx[0]), int(idx[1])
    img = np.asarray(base_img if base_img is not None else _empty_preview(size=size), dtype=np.uint8)
    h, w = int(img.shape[0]), int(img.shape[1])
    cell_h = max(1, h / float(size))
    cell_w = max(1, w / float(size))
    r = int(np.clip(np.floor(y / cell_h), 0, int(size) - 1))
    c = int(np.clip(np.floor(x / cell_w), 0, int(size) - 1))
    if bool(region_select_enabled):
        return _toggle_region_from_coords(points, r, c, base_img, region_ids, size=size)
    return _toggle_point_from_coords(points, r, c, base_img, size=size)


def _clear_points(base_img: np.ndarray | None, size: int = DEFAULT_GRID_SIZE):
    empty: list[Tuple[int, int]] = []
    return empty, "", 0, _overlay_points(base_img, empty, size=size)


def _normalize_points_input(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Point list cannot be empty.")
    cleaned = raw.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")
    cleaned = cleaned.replace(";", " ").replace("|", " ").replace("\n", " ")
    tuple_matches = re.findall(r"\(\s*\d+\s*,\s*\d+(?:\s*:\s*\d+)?\s*\)", cleaned)
    if tuple_matches:
        tokens = [m.strip().strip("()").replace(" ", "") for m in tuple_matches]
    else:
        tokens = re.findall(r"\d+\s*,\s*\d+(?:\s*:\s*\d+)?", cleaned)
        tokens = [t.replace(" ", "") for t in tokens]
    if not tokens:
        raise ValueError(
            "No valid points found. Use formats like: '1,1 1,2 1,3' or '(1,1),(1,2)' or '1,10:12'."
        )
    return tokens


def _safe_name(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


def _resize_img(img: np.ndarray, size_px: int, nearest: bool) -> np.ndarray:
    try:
        from PIL import Image
        resample = Image.NEAREST if nearest else Image.LANCZOS
        return np.asarray(
            Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((int(size_px), int(size_px)), resample),
            dtype=np.uint8,
        )
    except Exception:
        return np.asarray(img, dtype=np.uint8)


def _hstack(images: Sequence[np.ndarray], pad: int = 3, bg: int = 255) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    h = int(images[0].shape[0])
    c = int(images[0].shape[2])
    w_total = sum(int(im.shape[1]) for im in images) + int(pad) * (len(images) - 1)
    out = np.full((h, w_total, c), int(bg), dtype=np.uint8)
    x = 0
    for i, img in enumerate(images):
        hh, ww = int(img.shape[0]), int(img.shape[1])
        out[:hh, x : x + ww] = img
        x += ww
        if i + 1 < len(images):
            x += int(pad)
    return out


def _vstack(images: Sequence[np.ndarray], pad: int = 3, bg: int = 255) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    w = max(int(im.shape[1]) for im in images)
    c = int(images[0].shape[2])
    total_h = sum(int(im.shape[0]) for im in images) + int(pad) * (len(images) - 1)
    out = np.full((total_h, w, c), int(bg), dtype=np.uint8)
    y = 0
    for i, img in enumerate(images):
        hh, ww = int(img.shape[0]), int(img.shape[1])
        out[y : y + hh, :ww] = img
        y += hh
        if i + 1 < len(images):
            y += int(pad)
    return out


def _resolve_config_path(config_value: str) -> str:
    raw = str(config_value or "").strip()
    if not raw:
        raw = DEFAULT_CONFIG
    if ":" in raw:
        return raw
    p = Path(raw)
    if p.suffix.lower() in {".yml", ".yaml"}:
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return str(p)
    return raw


def _load_yaml_defaults(config_spec: str) -> dict:
    p = Path(config_spec)
    if p.suffix.lower() not in {".yml", ".yaml"} or not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    sampling = data.get("sampling", {}) if isinstance(data, dict) else {}
    decoder = data.get("rae_decoder", {}) if isinstance(data, dict) else {}

    omega_default = None
    omegas = sampling.get("omegas", None)
    if isinstance(omegas, list) and omegas:
        omega_default = float(omegas[0])
    elif sampling.get("omega", None) is not None:
        omega_default = float(sampling["omega"])

    t_min_default = None
    t_max_default = None
    interval = sampling.get("interval", None)
    if isinstance(interval, list) and interval and isinstance(interval[0], (list, tuple)) and len(interval[0]) >= 2:
        t_min_default = float(interval[0][0])
        t_max_default = float(interval[0][1])
    elif sampling.get("t_min", None) is not None and sampling.get("t_max", None) is not None:
        t_min_default = float(sampling["t_min"])
        t_max_default = float(sampling["t_max"])

    ema_default = None
    emas = sampling.get("emas", None)
    if isinstance(emas, list) and emas:
        ema_default = float(emas[0])

    load_from = ""
    if isinstance(data, dict):
        load_from = str(data.get("load_from", "") or "").strip()

    out = {
        "num_steps": sampling.get("num_steps", None),
        "omega": omega_default,
        "t_min": t_min_default,
        "t_max": t_max_default,
        "ema": ema_default,
        "decoder_stat": str(decoder.get("normalization_stat_path", "") or "").strip(),
        "decoder_code_dir": str(decoder.get("code_dir", "") or "").strip(),
        "decoder_config_path": str(decoder.get("decoder_config_path", "") or "").strip(),
        "decoder_ckpt": str(decoder.get("pretrained_decoder_path", "") or "").strip(),
        "decoder_device": str(decoder.get("device", "") or "").strip(),
        "strict_stats": bool(decoder.get("strict_stats", False)),
        "load_from": load_from,
    }
    return out


# ---------------------------------------------------------------------------
# Latent preview / region helpers
# ---------------------------------------------------------------------------

def _extract_level_and_preview(
    pt_path: Path,
    source_level_index: int,
    size: int = DEFAULT_GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import token_exchange_specific_points as tx

    levels, _ = tx._load_latent_levels(pt_path, latent_dim_hint=None)
    levels = np.asarray(levels, dtype=np.float32)
    if levels.ndim != 4:
        raise ValueError(f"Expected [L,H,W,C], got {levels.shape}")
    src_idx = int(source_level_index)
    if src_idx < 0 or src_idx >= int(levels.shape[0]):
        raise ValueError(f"source_level_index {src_idx} out of range [0, {int(levels.shape[0]) - 1}]")

    level = np.asarray(levels[src_idx], dtype=np.float32)
    h, w = int(level.shape[0]), int(level.shape[1])
    if h != int(size) or w != int(size):
        raise ValueError(f"This UI expects {size}x{size} latent grid, got {h}x{w} in {pt_path.name}")

    tiles = tx._joint_soft_rgb_level_tiles(levels)
    tile = np.asarray(tiles[src_idx], dtype=np.uint8)
    try:
        from PIL import Image
        tile_preview = np.asarray(
            Image.fromarray(tile).resize((int(size) * PREVIEW_SCALE, int(size) * PREVIEW_SCALE), Image.NEAREST),
            dtype=np.uint8,
        )
    except Exception:
        tile_preview = np.kron(tile, np.ones((PREVIEW_SCALE, PREVIEW_SCALE, 1), dtype=np.uint8))
    return level, tile_preview


def _region_id_map_from_local_cosine(level_hwc: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(level_hwc, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected latent level [H,W,C], got {arr.shape}")
    h, w, c = arr.shape
    flat = arr.reshape(h * w, c)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat_n = flat / np.clip(norms, 1e-8, None)
    flat_n = flat_n.reshape(h, w, c)
    right_sim = np.sum(flat_n[:, :-1, :] * flat_n[:, 1:, :], axis=-1) >= float(threshold)
    down_sim = np.sum(flat_n[:-1, :, :] * flat_n[1:, :, :], axis=-1) >= float(threshold)
    n = h * w
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for r in range(h):
        for c0 in range(w - 1):
            if right_sim[r, c0]:
                union(r * w + c0, r * w + (c0 + 1))
    for r in range(h - 1):
        for c0 in range(w):
            if down_sim[r, c0]:
                union(r * w + c0, (r + 1) * w + c0)

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    _, inv = np.unique(roots, return_inverse=True)
    return inv.reshape(h, w).astype(np.int32)


# ---------------------------------------------------------------------------
# 2x4 panel builder (identical to gradio_latent_editor)
# ---------------------------------------------------------------------------

def _build_levels_panel_2x4(
    pt_path: Path,
    out_dir: Path,
    panel_name: str,
    config_spec: str,
    defaults: dict,
    use_decoder_stats_override: bool,
    decoder_stat: str,
    strict_stats: bool,
) -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import token_exchange_specific_points as tx
    from utils.rae_decoder import get_decoder

    levels, _ = tx._load_latent_levels(pt_path, latent_dim_hint=None)
    levels = np.asarray(levels, dtype=np.float32)
    if levels.ndim != 4:
        raise ValueError(f"Expected levels [L,H,W,C], got {levels.shape}")

    use_levels = min(4, int(levels.shape[0]))
    level_ids = list(range(use_levels))
    if not level_ids:
        raise ValueError(f"No levels found in {pt_path}")

    cfg = tx._load_config_from_spec(config_spec)
    if not hasattr(cfg, "rae_decoder") or cfg.rae_decoder is None:
        raise ValueError("Config must contain rae_decoder section.")
    cfg.rae_decoder.enabled = True

    if defaults.get("decoder_code_dir"):
        cfg.rae_decoder.code_dir = str(defaults["decoder_code_dir"])
    if defaults.get("decoder_config_path"):
        cfg.rae_decoder.decoder_config_path = str(defaults["decoder_config_path"])
    if defaults.get("decoder_ckpt"):
        cfg.rae_decoder.pretrained_decoder_path = str(defaults["decoder_ckpt"])
    if defaults.get("decoder_device"):
        cfg.rae_decoder.device = str(defaults["decoder_device"])

    decoder_stat_eff = str(decoder_stat or "").strip() or str(defaults.get("decoder_stat", "")).strip()
    strict_stats_eff = bool(strict_stats) if bool(use_decoder_stats_override) else bool(defaults.get("strict_stats", False))
    enable_decoder_stats = bool(use_decoder_stats_override) or bool(decoder_stat_eff) or bool(strict_stats_eff)
    if enable_decoder_stats:
        cfg.rae_decoder.normalization_stat_path = decoder_stat_eff
        cfg.rae_decoder.strict_stats = bool(strict_stats_eff)
    else:
        cfg.rae_decoder.normalization_stat_path = ""
        cfg.rae_decoder.strict_stats = False

    h, c = int(levels.shape[1]), int(levels.shape[3])
    decoder = get_decoder(cfg, latent_hw=h, latent_dim=c)
    if decoder is None:
        raise RuntimeError("Failed to initialize RAE decoder for panel generation.")

    latent_tiles_all = tx._joint_soft_rgb_level_tiles(levels)
    decoded_tiles: list[np.ndarray] = []
    latent_tiles: list[np.ndarray] = []
    for li in level_ids:
        lvl = np.asarray(levels[li], dtype=np.float32)
        bchw = np.transpose(lvl[None, ...], (0, 3, 1, 2))
        dec = np.asarray(decoder.decode_uint8(bchw)[0], dtype=np.uint8)
        decoded_tiles.append(dec)
        latent_tiles.append(np.asarray(latent_tiles_all[li], dtype=np.uint8))

    tile_px = int(decoded_tiles[0].shape[0])
    decoded_tiles = [_resize_img(im, tile_px, nearest=False) for im in decoded_tiles]
    latent_tiles = [_resize_img(im, tile_px, nearest=True) for im in latent_tiles]

    if len(decoded_tiles) < 4:
        blank = np.full_like(decoded_tiles[0], 255, dtype=np.uint8)
        for _ in range(4 - len(decoded_tiles)):
            decoded_tiles.append(blank.copy())
            latent_tiles.append(blank.copy())

    row_decoded = _hstack(decoded_tiles, pad=3, bg=255)
    row_latent = _hstack(latent_tiles, pad=3, bg=255)
    panel = _vstack([row_decoded, row_latent], pad=3, bg=255)

    out_path = out_dir / panel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(panel).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# .pt generation from class numbers
# ---------------------------------------------------------------------------

def _generate_pt(
    class_id: int,
    config_spec: str,
    defaults: dict,
    out_dir: Path,
    sample_index: int = 0,
) -> Path:
    """Call image_generation.py to produce a single .pt for the given class."""
    cmd = [
        sys.executable,
        str(IMAGE_GEN_SCRIPT),
        "--config",
        config_spec,
        "--load_from",
        str(defaults.get("load_from", "")),
        "--out_dir",
        str(out_dir),
        "--classes",
        str(int(class_id)),
        "--num_per_class",
        "1",
        "--class_sample_offset",
        str(int(sample_index)),
        "--seed",
        str(FIXED_SEED),
        "--tile_px",
        str(FIXED_TILE_PX),
        "--no_individual_level_images",
    ]

    # Apply sampling overrides from config defaults.
    if defaults.get("omega") is not None:
        cmd.extend(["--omega", str(float(defaults["omega"]))])
    if defaults.get("t_min") is not None and defaults.get("t_max") is not None:
        cmd.extend(["--t_min", str(float(defaults["t_min"])), "--t_max", str(float(defaults["t_max"]))])
    if defaults.get("ema") is not None:
        cmd.extend(["--ema", str(float(defaults["ema"]))])
    if defaults.get("num_steps") is not None:
        cmd.extend(["--num_steps", str(int(defaults["num_steps"]))])

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        raise RuntimeError(f"Generation failed for class {class_id} (exit {proc.returncode}).\n{logs}")

    pt_path = out_dir / "latents_pt" / f"class_{int(class_id)}_sample_{int(sample_index)}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Expected .pt not found: {pt_path}")
    return pt_path


def _run_generate(
    target_class: int,
    reference_class: int,
    num_samples: int,
    source_level_index: int,
):
    """Generate .pt files for both classes and return paths + preview panels."""
    if target_class < 0 or target_class >= 1000:
        raise gr.Error("Target class must be in [0, 999].")
    if reference_class < 0 or reference_class >= 1000:
        raise gr.Error("Reference class must be in [0, 999].")

    config_spec = _resolve_config_path(DEFAULT_CONFIG)
    defaults = _load_yaml_defaults(config_spec)

    if not defaults.get("load_from"):
        raise gr.Error("No load_from checkpoint path found in config. Cannot generate.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = GRADIO_RUNS_DIR / f"gen_{stamp}_cls{int(target_class)}_cls{int(reference_class)}"

    target_gen_dir = gen_dir / "target"
    ref_gen_dir = gen_dir / "reference"

    num_samples = max(1, int(num_samples))
    target_pts: list[Path] = []
    ref_pts: list[Path] = []

    # Generate target class samples.
    for i in range(num_samples):
        pt = _generate_pt(int(target_class), config_spec, defaults, target_gen_dir, sample_index=i)
        target_pts.append(pt)

    # Generate reference class samples.
    for i in range(num_samples):
        pt = _generate_pt(int(reference_class), config_spec, defaults, ref_gen_dir, sample_index=i)
        ref_pts.append(pt)

    # Build preview panels for first sample of each.
    target_panel = _build_levels_panel_2x4(
        target_pts[0],
        out_dir=gen_dir,
        panel_name="target_preview_panel.png",
        config_spec=config_spec,
        defaults=defaults,
        use_decoder_stats_override=False,
        decoder_stat="",
        strict_stats=False,
    )
    ref_panel = _build_levels_panel_2x4(
        ref_pts[0],
        out_dir=gen_dir,
        panel_name="reference_preview_panel.png",
        config_spec=config_spec,
        defaults=defaults,
        use_decoder_stats_override=False,
        decoder_stat="",
        strict_stats=False,
    )

    # Load latent previews for point selection.
    _, tgt_base = _extract_level_and_preview(target_pts[0], int(source_level_index), size=DEFAULT_GRID_SIZE)
    _, ref_base = _extract_level_and_preview(ref_pts[0], int(source_level_index), size=DEFAULT_GRID_SIZE)

    empty: list[Tuple[int, int]] = []

    return (
        str(target_pts[0]),
        str(ref_pts[0]),
        str(target_panel) if target_panel.exists() else None,
        str(ref_panel) if ref_panel.exists() else None,
        ref_base,
        tgt_base,
        _overlay_points(ref_base, empty, size=DEFAULT_GRID_SIZE),
        _overlay_points(tgt_base, empty, size=DEFAULT_GRID_SIZE),
        empty,
        empty,
        "",
        "",
        0,
        0,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Load previews (for re-loading after generation or manual file swap)
# ---------------------------------------------------------------------------

def _load_previews(target_pt_path: str, ref_pt_path: str, source_level_index: int):
    if not target_pt_path:
        raise gr.Error("No target .pt available. Generate first.")
    tgt = Path(target_pt_path)
    ref = Path(ref_pt_path) if ref_pt_path else tgt
    if not tgt.exists():
        raise gr.Error(f"Target .pt not found: {tgt}")
    if not ref.exists():
        raise gr.Error(f"Reference .pt not found: {ref}")

    _, ref_base = _extract_level_and_preview(ref, int(source_level_index), size=DEFAULT_GRID_SIZE)
    _, tgt_base = _extract_level_and_preview(tgt, int(source_level_index), size=DEFAULT_GRID_SIZE)
    empty: list[Tuple[int, int]] = []
    return (
        ref_base,
        tgt_base,
        _overlay_points(ref_base, empty, size=DEFAULT_GRID_SIZE),
        _overlay_points(tgt_base, empty, size=DEFAULT_GRID_SIZE),
        empty,
        empty,
        "",
        "",
        0,
        0,
        None,
        None,
    )


def _apply_cosine_regions(
    target_pt_path: str,
    ref_pt_path: str,
    source_level_index: int,
    ref_base_img: np.ndarray | None,
    tgt_base_img: np.ndarray | None,
    ref_points: Sequence[Tuple[int, int]],
    tgt_points: Sequence[Tuple[int, int]],
):
    if not target_pt_path:
        raise gr.Error("No target .pt available. Generate first.")
    tgt = Path(target_pt_path)
    ref = Path(ref_pt_path) if ref_pt_path else tgt
    if not tgt.exists():
        raise gr.Error(f"Target .pt not found: {tgt}")
    if not ref.exists():
        raise gr.Error(f"Reference .pt not found: {ref}")

    ref_level, _ = _extract_level_and_preview(ref, int(source_level_index), size=DEFAULT_GRID_SIZE)
    tgt_level, _ = _extract_level_and_preview(tgt, int(source_level_index), size=DEFAULT_GRID_SIZE)

    ref_region_ids = _region_id_map_from_local_cosine(ref_level, threshold=COS_REGION_THRESHOLD)
    tgt_region_ids = _region_id_map_from_local_cosine(tgt_level, threshold=COS_REGION_THRESHOLD)

    ref_overlay = _overlay_points(ref_base_img, ref_points, size=DEFAULT_GRID_SIZE)
    tgt_overlay = _overlay_points(tgt_base_img, tgt_points, size=DEFAULT_GRID_SIZE)

    return ref_region_ids, tgt_region_ids, ref_overlay, tgt_overlay


# ---------------------------------------------------------------------------
# Run edit + generate
# ---------------------------------------------------------------------------

def _run_edit(
    target_pt_path: str,
    ref_pt_path: str,
    source_level_index: int,
    reference_points: str,
    target_points: str,
    num_steps,
    omega,
    t_min,
    t_max,
    ema,
    use_online: bool,
    use_decoder_stats_override: bool,
    decoder_stat: str,
    strict_stats: bool,
):
    if not target_pt_path:
        raise gr.Error("No target .pt available. Generate first.")
    target_path = Path(target_pt_path)
    if not target_path.exists():
        raise gr.Error(f"Target .pt not found: {target_path}")

    ref_path = Path(ref_pt_path) if ref_pt_path else target_path
    if not ref_path.exists():
        raise gr.Error(f"Reference .pt not found: {ref_path}")

    try:
        ref_tokens = _normalize_points_input(reference_points)
        tgt_tokens = _normalize_points_input(target_points)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    config_spec = _resolve_config_path(DEFAULT_CONFIG)
    defaults = _load_yaml_defaults(config_spec)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"edit_{run_stamp}_{_safe_name(str(target_path))}"
    out_dir = GRADIO_RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build original panel.
    original_panel_path = _build_levels_panel_2x4(
        target_path,
        out_dir=out_dir,
        panel_name="original_uploaded_panel_2x4.png",
        config_spec=config_spec,
        defaults=defaults,
        use_decoder_stats_override=bool(use_decoder_stats_override),
        decoder_stat=str(decoder_stat),
        strict_stats=bool(strict_stats),
    )

    cmd = [
        sys.executable,
        str(TOKEN_EXCHANGE_SCRIPT),
        "--target_pt",
        str(target_path),
        "--reference_pt",
        str(ref_path),
        "--out_dir",
        str(out_dir),
        "--source_level_index",
        str(int(source_level_index)),
        "--config",
        str(config_spec),
        "--seed",
        str(FIXED_SEED),
        "--tile_px",
        str(FIXED_TILE_PX),
        "--reference_points",
        *ref_tokens,
        "--target_points",
        *tgt_tokens,
    ]

    load_from = str(defaults.get("load_from", "")).strip()
    if load_from:
        cmd.extend(["--load_from", load_from])

    num_steps_eff = num_steps if num_steps is not None else defaults.get("num_steps", None)
    omega_eff = omega if omega is not None else defaults.get("omega", None)
    t_min_eff = t_min if t_min is not None else defaults.get("t_min", None)
    t_max_eff = t_max if t_max is not None else defaults.get("t_max", None)
    ema_eff = ema if ema is not None else defaults.get("ema", None)

    if num_steps_eff is not None:
        cmd.extend(["--num_steps", str(int(num_steps_eff))])
    if omega_eff is not None:
        cmd.extend(["--omega", str(float(omega_eff))])
    if t_min_eff is not None:
        cmd.extend(["--t_min", str(float(t_min_eff))])
    if t_max_eff is not None:
        cmd.extend(["--t_max", str(float(t_max_eff))])
    if bool(use_online):
        cmd.append("--use_online")
    elif ema_eff is not None:
        cmd.extend(["--ema", str(float(ema_eff))])

    decoder_stat_eff = str(decoder_stat or "").strip() or str(defaults.get("decoder_stat", "")).strip()
    strict_stats_eff = bool(strict_stats) if bool(use_decoder_stats_override) else bool(defaults.get("strict_stats", False))
    enable_decoder_stats = bool(use_decoder_stats_override) or bool(decoder_stat_eff) or bool(strict_stats_eff)
    if enable_decoder_stats:
        cmd.append("--use_decoder_stats")
        if decoder_stat_eff:
            cmd.extend(["--decoder_stat", decoder_stat_eff])
        if strict_stats_eff:
            cmd.append("--strict_stats")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise gr.Error(f"Edit run failed (exit {proc.returncode}).\n\n{logs}")

    panel_path = out_dir / "panel_3x3.png"
    edited_last_decoded_path = out_dir / "bottom_right_edited_last_level_decoded.png"
    unedited_last_decoded_path = out_dir / "top_right_target_last_level_decoded.png"
    generated_pt = out_dir / "generated_from_specific_token_exchange.pt"

    edited_panel_path = _build_levels_panel_2x4(
        generated_pt,
        out_dir=out_dir,
        panel_name="edited_generated_panel_2x4.png",
        config_spec=config_spec,
        defaults=defaults,
        use_decoder_stats_override=bool(use_decoder_stats_override),
        decoder_stat=str(decoder_stat),
        strict_stats=bool(strict_stats),
    )

    if not panel_path.exists() or not generated_pt.exists():
        raise gr.Error(f"Run finished but expected outputs are missing in {out_dir}")

    return (
        str(original_panel_path) if original_panel_path.exists() else None,
        str(edited_panel_path) if edited_panel_path.exists() else None,
        str(panel_path),
        str(unedited_last_decoded_path) if unedited_last_decoded_path.exists() else None,
        str(edited_last_decoded_path) if edited_last_decoded_path.exists() else None,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="pMF Generate & Edit") as demo:
        gr.Markdown(
            "# pMF Generate & Edit\n"
            "Enter ImageNet class numbers to generate `.pt` latents, then edit tokens and regenerate."
        )

        # Hidden state.
        ref_points_state = gr.State([])
        tgt_points_state = gr.State([])
        ref_base_img_state = gr.State(_empty_preview())
        tgt_base_img_state = gr.State(_empty_preview())
        ref_region_ids_state = gr.State(None)
        tgt_region_ids_state = gr.State(None)
        target_pt_state = gr.State("")
        ref_pt_state = gr.State("")

        # --- Step 1: Generate ---
        gr.Markdown("## Step 1: Generate latents from class IDs")
        with gr.Row():
            target_class = gr.Number(value=288, precision=0, label="Target class ID (0-999)")
            reference_class = gr.Number(value=305, precision=0, label="Reference class ID (0-999)")
            num_samples = gr.Number(value=1, precision=0, label="Samples per class")
            source_level_index = gr.Number(value=2, precision=0, label="Source level index")

        generate_btn = gr.Button("Generate .pt files", variant="primary")

        with gr.Row():
            target_gen_panel = gr.Image(type="filepath", label="Target class panel (2x4)")
            ref_gen_panel = gr.Image(type="filepath", label="Reference class panel (2x4)")

        # --- Step 2: Select points ---
        gr.Markdown("## Step 2: Select tokens to exchange")
        gr.Markdown("Click directly on latent images to select/unselect cells. Click order defines token mapping.")

        load_previews_btn = gr.Button("Reload latent previews", variant="secondary")
        with gr.Row():
            apply_regions_btn = gr.Button("Apply cosine regions (threshold 0.90)", variant="secondary")
            region_select_enabled = gr.Checkbox(value=False, label="Activate region click selection")

        with gr.Row():
            with gr.Column():
                ref_count = gr.Number(value=0, precision=0, label="Reference selected points")
                clear_ref = gr.Button("Clear reference selection")
                reference_points = gr.Textbox(
                    label="Reference/source points (auto-filled, editable)",
                    value="",
                    lines=2,
                )
                reference_preview_img = gr.Image(
                    value=_overlay_points(_empty_preview(), [], size=DEFAULT_GRID_SIZE),
                    type="numpy",
                    label="Reference latent image (click to select)",
                    interactive=True,
                )
            with gr.Column():
                tgt_count = gr.Number(value=0, precision=0, label="Target selected points")
                clear_tgt = gr.Button("Clear target selection")
                target_points = gr.Textbox(
                    label="Target/destination points (auto-filled, editable)",
                    value="",
                    lines=2,
                )
                target_preview_img = gr.Image(
                    value=_overlay_points(_empty_preview(), [], size=DEFAULT_GRID_SIZE),
                    type="numpy",
                    label="Target latent image (click to select)",
                    interactive=True,
                )

        # --- Advanced options ---
        with gr.Accordion("Advanced sampling/decoder options", open=False):
            with gr.Row():
                num_steps = gr.Number(value=None, precision=0, label="num_steps override")
                omega = gr.Number(value=None, label="omega override")
                t_min = gr.Number(value=None, label="t_min override")
                t_max = gr.Number(value=None, label="t_max override")
                ema = gr.Number(value=None, label="ema override")
            with gr.Row():
                use_online = gr.Checkbox(value=False, label="Use online params (not EMA)")
                use_decoder_stats_override = gr.Checkbox(
                    value=False,
                    label="Override decoder stats (otherwise use config defaults)",
                )
                strict_stats = gr.Checkbox(value=False, label="Strict decoder stats")
            decoder_stat = gr.Textbox(value="", label="decoder_stat path")

        # --- Step 3: Edit ---
        gr.Markdown("## Step 3: Run edit + generate")
        run_btn = gr.Button("Run edit + generate", variant="primary")

        with gr.Row():
            uploaded_panel_img = gr.Image(type="filepath", label="Original .pt panel (2x4)")
            edited_panel_img = gr.Image(type="filepath", label="Edited/generated panel (2x4)")
        with gr.Row():
            panel_img = gr.Image(type="filepath", label="3x3 panel")
            unedited_last_img = gr.Image(type="filepath", label="Unedited last level decoded")
            edited_last_img = gr.Image(type="filepath", label="Edited last level decoded")

        # --- Wiring ---

        generate_btn.click(
            fn=_run_generate,
            inputs=[target_class, reference_class, num_samples, source_level_index],
            outputs=[
                target_pt_state,
                ref_pt_state,
                target_gen_panel,
                ref_gen_panel,
                ref_base_img_state,
                tgt_base_img_state,
                reference_preview_img,
                target_preview_img,
                ref_points_state,
                tgt_points_state,
                reference_points,
                target_points,
                ref_count,
                tgt_count,
                ref_region_ids_state,
                tgt_region_ids_state,
            ],
        )

        load_previews_btn.click(
            fn=_load_previews,
            inputs=[target_pt_state, ref_pt_state, source_level_index],
            outputs=[
                ref_base_img_state,
                tgt_base_img_state,
                reference_preview_img,
                target_preview_img,
                ref_points_state,
                tgt_points_state,
                reference_points,
                target_points,
                ref_count,
                tgt_count,
                ref_region_ids_state,
                tgt_region_ids_state,
            ],
        )

        run_btn.click(
            fn=_run_edit,
            inputs=[
                target_pt_state,
                ref_pt_state,
                source_level_index,
                reference_points,
                target_points,
                num_steps,
                omega,
                t_min,
                t_max,
                ema,
                use_online,
                use_decoder_stats_override,
                decoder_stat,
                strict_stats,
            ],
            outputs=[
                uploaded_panel_img,
                edited_panel_img,
                panel_img,
                unedited_last_img,
                edited_last_img,
            ],
        )

        clear_ref.click(
            fn=_clear_points,
            inputs=[ref_base_img_state],
            outputs=[ref_points_state, reference_points, ref_count, reference_preview_img],
        )
        clear_tgt.click(
            fn=_clear_points,
            inputs=[tgt_base_img_state],
            outputs=[tgt_points_state, target_points, tgt_count, target_preview_img],
        )

        reference_preview_img.select(
            fn=_toggle_point_or_region_from_image,
            inputs=[ref_points_state, ref_base_img_state, region_select_enabled, ref_region_ids_state],
            outputs=[ref_points_state, reference_points, ref_count, reference_preview_img],
        )
        target_preview_img.select(
            fn=_toggle_point_or_region_from_image,
            inputs=[tgt_points_state, tgt_base_img_state, region_select_enabled, tgt_region_ids_state],
            outputs=[tgt_points_state, target_points, tgt_count, target_preview_img],
        )

        apply_regions_btn.click(
            fn=_apply_cosine_regions,
            inputs=[
                target_pt_state,
                ref_pt_state,
                source_level_index,
                ref_base_img_state,
                tgt_base_img_state,
                ref_points_state,
                tgt_points_state,
            ],
            outputs=[
                ref_region_ids_state,
                tgt_region_ids_state,
                reference_preview_img,
                target_preview_img,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=str, default="127.0.0.1", help="Host for Gradio server")
    p.add_argument("--port", type=int, default=7861, help="Port for Gradio server")
    p.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not IMAGE_GEN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {IMAGE_GEN_SCRIPT}")
    if not TOKEN_EXCHANGE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {TOKEN_EXCHANGE_SCRIPT}")
    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=int(args.port), share=bool(args.share))


if __name__ == "__main__":
    main()
