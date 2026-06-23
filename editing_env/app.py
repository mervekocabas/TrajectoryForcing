#!/usr/bin/env python3
"""Trajectory Forcing — interactive latent token-exchange editing (local GPU).

Load-once design:
  * tf_pipeline is imported at module top (no JAX/GPU touched there).
  * The model is built/loaded lazily on the first Generate (get_pipeline()), so JAX
    initializes its XLA GPU backend only when inference actually runs.
  * The pipeline is cached, so subsequent calls reuse the loaded model.

Launch with ./run.sh [PORT], which points CUDA at the GPU assigned node.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import gradio as gr

# editing_env/ is a thin launcher; the model/config/checkpoint machinery lives in the
# repo root one level up. Point imports there (tf_pipeline, scripts/ helpers, classes).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the grid/point UI helpers from the existing app (pure functions, no model load).
from gradio_generate_and_edit import (  # type: ignore
    _overlay_points,
    _toggle_point_or_region_from_image,
    _clear_points,
    _points_to_text,
    _normalize_points_input,
    _region_id_map_from_local_cosine,
    DEFAULT_GRID_SIZE,
    PREVIEW_SCALE,
    FIXED_SEED,
)

# ImageNet-1k id -> class name (0–999), loaded from the JSON beside this file.
# (JSON object keys are strings; convert back to int to match IMAGENET_CLASSES.get(int(v)).)
import json
with open(Path(__file__).resolve().parent / "imagenet_classes.json", encoding="utf-8") as _f:
    IMAGENET_CLASSES = {int(k): v for k, v in json.load(_f).items()}

GRID = int(DEFAULT_GRID_SIZE)
PREVIEW_PX = GRID * int(PREVIEW_SCALE)  # 16 * 32 = 512


def _grid_base(tile_hwc: np.ndarray) -> np.ndarray:
    """Upscale a small latent PCA tile to the clickable preview size (nearest)."""
    from PIL import Image
    im = Image.fromarray(np.asarray(tile_hwc, dtype=np.uint8)).resize((PREVIEW_PX, PREVIEW_PX), Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


# --------------------------- helpers ---------------------------
DEFAULT_EDIT_LEVEL = 2  # level pre-selected after Generate (clamped to model num_levels)
DEFAULT_COS_THRESHOLD = 0.9  # cosine-similarity threshold for token clustering


def _region_ids(levels, idx, threshold):
    """Connected-component cosine clusters of the latent tokens at a given level."""
    if levels is None:
        return None
    arr = np.asarray(levels)
    i = int(np.clip(idx, 0, arr.shape[0] - 1))
    return _region_id_map_from_local_cosine(arr[i], float(threshold))


def _is_blank(v) -> bool:
    """True if a gr.Number field is empty/None/NaN (used for the optional target class)."""
    if v is None:
        return True
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False


def _caption_font(px: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            pass
    try:
        return ImageFont.load_default(px)
    except Exception:
        return ImageFont.load_default()


def _with_caption_strip(img: np.ndarray, text: str) -> np.ndarray:
    """Append a label strip BELOW the image (drawn into the pixels) so the caption sits
    directly under each tile and never overlaps the content."""
    from PIL import Image, ImageDraw
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    strip_h = max(24, w // 14)
    out = np.empty((h + strip_h, w, 3), dtype=np.uint8)
    out[:h] = arr
    out[h:] = np.array([17, 26, 43], dtype=np.uint8)  # panel bg (#111A2B)
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    font = _caption_font(max(13, strip_h - 10))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textlength(text, font=font), strip_h - 8
    tx = max(2, (w - int(tw)) // 2)
    ty = h + max(1, (strip_h - int(th)) // 2 - 2)
    draw.text((tx, ty), text, fill=(203, 213, 225), font=font)
    return np.asarray(im, dtype=np.uint8)


def _level_caption(i: int, num_levels: int) -> str:
    if i == num_levels - 1:
        note = "fine"
    else:
        note = {0: "coarse", 1: "parts", 2: "subparts"}.get(i)
    return f"Level {i} ({note})" if note else f"Level {i}"


def _level_galleries(pipe, levels, palette=None):
    """Build (latent_tiles, latent_gallery, decoded_gallery) for every level.

    latent_tiles: list of upscaled PCA tiles (for re-use as clickable grid bases).
    *_gallery: list of (image, caption) suitable for gr.Gallery.
    palette: optional shared PCA palette so colors are comparable across images.
    """
    lat_tiles = [_grid_base(t) for t in pipe.pca_tiles(levels, palette=palette)]  # clean (grid base)
    dec_imgs = pipe.decode_all(levels)
    n = pipe.num_levels
    # gallery items have the level label baked in as a strip beneath each tile (no overlay text)
    lat_gallery = [_with_caption_strip(img, _level_caption(i, n)) for i, img in enumerate(lat_tiles)]
    dec_gallery = [_with_caption_strip(img, _level_caption(i, n)) for i, img in enumerate(dec_imgs)]
    return lat_tiles, lat_gallery, dec_gallery


def _class_name(v):
    try:
        n = IMAGENET_CLASSES.get(int(v))
    except (TypeError, ValueError):
        return None
    return n.lower() if n else None


def _ref_label(v) -> str:
    base = "reference class id (0–999)"
    n = _class_name(v)
    return f"{base} — {n}" if n else base


def _tgt_label(v) -> str:
    base = "target class id (0–999)"
    if _is_blank(v):
        return f"{base} — blank reuses reference"
    n = _class_name(v)
    return f"{base} — {n}" if n else base


def _level_status(ref_level: int, tgt_level: int) -> str:
    return f"**Selected — reference level:** {int(ref_level)}  |  **target level:** {int(tgt_level)}"


# --------------------------- inference callbacks ---------------------------
def do_generate(reference_class: int, target_class, seed: int, cos_threshold: float = DEFAULT_COS_THRESHOLD):
    from tf_pipeline import get_pipeline
    pipe = get_pipeline()

    reuse = _is_blank(target_class)  # blank target -> use one image as both reference and target
    ref_levels = pipe.generate(int(reference_class), seed=int(seed))
    if reuse:
        tgt_levels = ref_levels
    else:
        tgt_levels = pipe.generate(int(target_class), seed=int(seed))

    # Shared PCA palette so reference / target / edited latents use the SAME colors.
    palette = pipe.fit_palette([ref_levels] if reuse else [ref_levels, tgt_levels])

    ref_lat, ref_lat_g, ref_dec_g = _level_galleries(pipe, ref_levels, palette=palette)
    if reuse:
        tgt_lat, tgt_lat_g, tgt_dec_g = ref_lat, ref_lat_g, ref_dec_g
    else:
        tgt_lat, tgt_lat_g, tgt_dec_g = _level_galleries(pipe, tgt_levels, palette=palette)

    s = int(np.clip(DEFAULT_EDIT_LEVEL, 0, pipe.num_levels - 1))
    ref_tile, tgt_tile = ref_lat[s], tgt_lat[s]
    ref_reg = _region_ids(ref_levels, s, cos_threshold)
    tgt_reg = _region_ids(tgt_levels, s, cos_threshold)

    empty: list = []
    return (
        ref_levels, tgt_levels,                                  # state (np arrays)
        ref_lat_g, tgt_lat_g, ref_dec_g, tgt_dec_g,             # per-level galleries
        ref_tile, tgt_tile,                                      # clickable grid bases
        _overlay_points(ref_tile, empty, size=GRID),            # ref_grid
        _overlay_points(tgt_tile, empty, size=GRID),            # tgt_grid
        empty, empty, "", "", 0, 0,                             # reset points / text / counts
        s, s,                                                    # ref_level, tgt_level states
        _level_status(s, s),                                    # status line
        palette,                                                 # shared PCA palette state
        s, s,                                                    # ref_pending, tgt_pending
        _ref_btn_committed(s), _tgt_btn_committed(s),            # green: level s is in the edit panel
        ref_reg, tgt_reg,                                        # cosine cluster region ids
    )


def _ref_btn_committed(idx):
    return gr.update(value=f"✅ Level {idx} is in the edit panel (REFERENCE)", variant="secondary")


def _ref_btn_pending(idx):
    return gr.update(value=f"Use level {idx} for REFERENCE editing", variant="secondary")


def _tgt_btn_committed(idx):
    return gr.update(value=f"✅ Level {idx} is in the edit panel (TARGET)", variant="secondary")


def _tgt_btn_pending(idx):
    return gr.update(value=f"Use level {idx} for TARGET editing", variant="secondary")


def preview_ref_level(evt: gr.SelectData):
    """Single click a Reference latent tile -> just record which level is being previewed
    (the gallery itself opens it full-size). Button goes neutral until committed."""
    idx = int(evt.index)
    return idx, _ref_btn_pending(idx)


def preview_tgt_level(evt: gr.SelectData):
    idx = int(evt.index)
    return idx, _tgt_btn_pending(idx)


def choose_ref_level(ref_levels, pending_idx, tgt_level, palette, cos_threshold):
    """Commit the previewed reference level: load its 16x16 grid; turn the tick green."""
    if ref_levels is None:
        raise gr.Error("Generate first.")
    from tf_pipeline import get_pipeline
    pipe = get_pipeline()  # cached; CPU-side viz only
    idx = int(pending_idx)
    base = _grid_base(pipe.pca_tiles(np.asarray(ref_levels), palette=palette)[idx])
    empty: list = []
    return (idx, base, _overlay_points(base, empty, size=GRID), empty, "", 0,
            _level_status(idx, tgt_level), _ref_btn_committed(idx),
            _region_ids(ref_levels, idx, cos_threshold))


def choose_tgt_level(tgt_levels, pending_idx, ref_level, palette, cos_threshold):
    """Commit the previewed target level: load its 16x16 grid; turn the tick green."""
    if tgt_levels is None:
        raise gr.Error("Generate first.")
    from tf_pipeline import get_pipeline
    pipe = get_pipeline()
    idx = int(pending_idx)
    base = _grid_base(pipe.pca_tiles(np.asarray(tgt_levels), palette=palette)[idx])
    empty: list = []
    return (idx, base, _overlay_points(base, empty, size=GRID), empty, "", 0,
            _level_status(ref_level, idx), _tgt_btn_committed(idx),
            _region_ids(tgt_levels, idx, cos_threshold))


def recompute_regions(ref_levels, tgt_levels, ref_level, tgt_level, cos_threshold):
    """Re-cluster both committed levels when the threshold slider changes."""
    return (_region_ids(ref_levels, ref_level, cos_threshold),
            _region_ids(tgt_levels, tgt_level, cos_threshold))


def do_edit(tgt_levels, ref_levels, ref_level, tgt_level, ref_points, tgt_points,
            reference_class, target_class, seed, palette):
    if tgt_levels is None or ref_levels is None:
        raise gr.Error("Generate first.")
    rp = [(int(r), int(c)) for (r, c) in (ref_points or [])]
    tp = [(int(r), int(c)) for (r, c) in (tgt_points or [])]
    if not rp or not tp:
        raise gr.Error("Select at least one reference token and one target token.")

    from tf_pipeline import get_pipeline
    pipe = get_pipeline()
    # downstream re-sampling keeps the target image's class (reference class if target was blank)
    eff_class = int(reference_class) if _is_blank(target_class) else int(target_class)
    edited = pipe.edit(np.asarray(tgt_levels), np.asarray(ref_levels),
                       int(ref_level), int(tgt_level), rp, tp, class_id=eff_class, seed=int(seed))
    _, edit_lat_g, edit_dec_g = _level_galleries(pipe, edited, palette=palette)
    return edit_lat_g, edit_dec_g


# ------------------------------- UI -------------------------------
def _theme():
    # Soft light pastel: airy lavender/white surfaces, gentle shadows, soft rounded.
    # System fonts (no external Google Fonts fetch — avoids proxy stalls at load).
    # Tokens set for BOTH light and dark so it always reads as the light pastel palette.
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
        font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    ).set(
        # one dark base, distinct panels, neutral text/labels; accent reserved for CTAs/focus
        body_background_fill="#0B1220",
        body_background_fill_dark="#0B1220",
        body_text_color="#E5E7EB",
        body_text_color_dark="#E5E7EB",
        body_text_color_subdued="#94A3B8",
        body_text_color_subdued_dark="#94A3B8",
        background_fill_primary="#111A2B",
        background_fill_primary_dark="#111A2B",
        background_fill_secondary="#162238",
        background_fill_secondary_dark="#162238",
        block_background_fill="#111A2B",
        block_background_fill_dark="#111A2B",
        block_border_color="#23324A",
        block_border_color_dark="#23324A",
        block_label_background_fill="#283466",
        block_label_background_fill_dark="#283466",
        block_label_text_color="#dbe4f3",
        block_label_text_color_dark="#dbe4f3",
        block_title_text_color="#F8FAFC",
        block_title_text_color_dark="#F8FAFC",
        input_background_fill="#0E1830",
        input_background_fill_dark="#0E1830",
        input_border_color="#23324A",
        input_border_color_dark="#23324A",
        border_color_primary="#23324A",
        border_color_primary_dark="#23324A",
        button_primary_background_fill="linear-gradient(90deg, #243A66, #2B5270)",
        button_primary_background_fill_dark="linear-gradient(90deg, #243A66, #2B5270)",
        button_primary_background_fill_hover="linear-gradient(90deg, #2D4778, #356283)",
        button_primary_background_fill_hover_dark="linear-gradient(90deg, #2D4778, #356283)",
        button_primary_text_color="#e6eefa",
        button_primary_text_color_dark="#e6eefa",
        button_primary_border_color="#243A66",
        button_primary_border_color_dark="#243A66",
        button_secondary_background_fill="#162238",
        button_secondary_background_fill_dark="#162238",
        button_secondary_text_color="#CBD5E1",
        button_secondary_text_color_dark="#CBD5E1",
    )


CUSTOM_CSS = """
.gradio-container {max-width: 96vw !important; width: 96vw !important; margin: 0 auto !important;
  background: #0B1220 !important; color: #E5E7EB !important;}
body {background: #0B1220 !important;}

/* header — panel with a faint accent glow */
#tf-banner {
  background: #111A2B; border: 1px solid #23324A; border-radius: 14px;
  padding: 20px 24px; margin: 6px 0 18px;
  box-shadow: 0 0 0 1px rgba(79,116,163,.10), 0 0 24px rgba(79,116,163,.04);
}
#tf-banner h1 {margin: 0; font-size: 1.45rem; font-weight: 750; letter-spacing: .2px; color: #F8FAFC;}
#tf-banner p  {margin: .45rem 0 0; font-size: .95rem; line-height: 1.5; max-width: 80ch; color: #94A3B8;}

/* section cards — distinct panels, neutral borders */
.tf-card {
  border-radius: 14px !important;
  padding: 16px 20px 20px !important;
  background: #111A2B !important;
  border: 1px solid #23324A !important;
  box-shadow: none !important;
  margin-bottom: 16px !important;
}
.tf-section h3 {
  margin: 2px 0 12px; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; letter-spacing: .2px;
}

/* soft badges (subtle, not bright) */
.tf-pill {display:inline-block; padding:3px 12px; border-radius:999px;
          background:rgba(79,116,163,.13); color:#93b1d2; border:1px solid rgba(79,116,163,.28);
          font-size:.77rem; font-weight:600; letter-spacing:.5px; margin-bottom:8px;}

/* galleries — subtle accent ring on hover */
.gradio-container .grid-wrap, .gradio-container .gallery, .gradio-container .thumbnails {border-radius: 12px !important;}
.gradio-container .gallery .thumbnail-item {border-radius: 8px !important; transition: box-shadow .12s ease;}
.gradio-container .gallery .thumbnail-item:hover {box-shadow: 0 0 0 1px rgba(79,116,163,.42) inset;}
/* gallery title sits ABOVE the images */
.tf-gtitle p {margin: 8px 0 4px; font-size: .9rem; font-weight: 600; color: #cbd5e1;}

/* primary CTA — deep navy-blue gradient, soft glow */
.gradio-container button.primary {
  font-weight: 600; letter-spacing:.2px; color:#e6eefa !important;
  background: linear-gradient(90deg, #243A66, #2B5270) !important; border: none !important;
  box-shadow: 0 0 0 1px rgba(79,116,163,.16), 0 5px 16px rgba(20,30,58,.30) !important;
}
.gradio-container button.primary:hover {background: linear-gradient(90deg, #2D4778, #356283) !important;}

/* secondary — neutral panel buttons */
.gradio-container button.secondary {
  background: #162238 !important; border: 1px solid #23324A !important; box-shadow: none !important;
  color: #CBD5E1 !important; font-weight: 500;
}
.gradio-container button.secondary:hover {background: #1B2A44 !important; border-color: #2F415F !important;}

/* accent focus ring on inputs */
.gradio-container input:focus, .gradio-container textarea:focus {
  border-color: #4F74A3 !important; box-shadow: 0 0 0 2px rgba(79,116,163,.25) !important;
}

/* class-id / seed number boxes — lighter midnight-blue fill, clearer text */
.gradio-container input[type=number] {
  background: #283466 !important; color: #eaf1fb !important;
  border: 1px solid #3a4a82 !important; border-radius: 8px !important;
}
/* white up/down stepper arrows on the number boxes (dark theme) */
.gradio-container input[type=number]::-webkit-inner-spin-button,
.gradio-container input[type=number]::-webkit-outer-spin-button {
  filter: invert(1) brightness(1.8); opacity: 1 !important;
}

.tf-card p {font-size:.93rem; color:#CBD5E1;}
footer {visibility: hidden;}
"""

HOWTO_MD = (
    "1. Enter a **Reference class id** (0–999); the matching ImageNet **class name is shown in the "
    "field's label** as you type. Optionally enter a **Target class id** — leave it **blank** to use "
    "one image as both reference and target.\n"
    "2. Click **Generate**. Each image is shown across all levels (coarse → fine) as both "
    "the **latent** (PCA) and the **decoded RGB**.\n"
    "3. **View full size:** single-click any tile in a **latent** or **decoded RGB** gallery to "
    "open it enlarged; use the ◀ ▶ arrows to step through levels and click ✕ (or **Esc**) to return.\n"
    "4. **Pick the edit level:** click a tile in a **latent** gallery, then press "
    "**“Use level N for reference/target editing”** below it (they can be the same level).\n"
    "5. On the 16×16 grids, **click tokens** to copy FROM (reference) and paste INTO (target). "
    "Counts need not match — each target token receives a reference token, cycling through your "
    "reference picks (1 reference + many targets pastes that one token everywhere). Tip: enable "
    "**Cluster select (cosine)** to grab a whole similar-token cluster in one click (tune the "
    "threshold slider).\n"
    "6. Click **Run edit + generate** to see the edited result across all levels."
)


def launch_kwargs() -> dict:
    """Theme + CSS to pass to demo.launch(). Gradio 6 moved these off
    gr.Blocks(), so every launch site (the __main__ block AND the Colab
    notebook) should splat this into launch()."""
    return {"theme": _theme(), "css": CUSTOM_CSS}


def build_demo() -> gr.Blocks:
    # NOTE: In Gradio 6, `theme` and `css` must be passed to demo.launch(),
    # NOT to gr.Blocks() (doing so is ignored and emits a UserWarning).
    # Every launch site passes them via launch_kwargs() below.
    with gr.Blocks(title="Trajectory Forcing — Interactive Editing") as demo:
        gr.HTML(
            "<div id='tf-banner'>"
            "<h1>Trajectory Forcing — Interactive Latent Token Exchange</h1>"
            "<p>Generate a reference &amp; a target from ImageNet classes, copy latent tokens "
            "between them across hierarchy levels, and regenerate — all in a few clicks.</p>"
            "</div>"
        )
        with gr.Accordion("📖 How to use", open=False):
            gr.Markdown(HOWTO_MD)

        ref_levels_st = gr.State(None)
        tgt_levels_st = gr.State(None)
        ref_pts_st = gr.State([])
        tgt_pts_st = gr.State([])
        ref_base_st = gr.State(None)
        tgt_base_st = gr.State(None)
        ref_level_st = gr.State(DEFAULT_EDIT_LEVEL)
        tgt_level_st = gr.State(DEFAULT_EDIT_LEVEL)
        ref_pending_st = gr.State(DEFAULT_EDIT_LEVEL)  # last latent tile clicked (previewed)
        tgt_pending_st = gr.State(DEFAULT_EDIT_LEVEL)
        palette_st = gr.State(None)  # shared PCA color palette across ref/target/edited
        ref_region_ids_st = gr.State(None)  # cosine token clusters for the committed ref level
        tgt_region_ids_st = gr.State(None)

        with gr.Accordion("① Generate", open=True, elem_classes="tf-card"):
            with gr.Row():
                reference_class = gr.Number(value=213, precision=0, label=_ref_label(213))
                target_class = gr.Number(value=207, precision=0, label=_tgt_label(207))
                seed = gr.Number(value=592, precision=0, label="seed")
            generate_btn = gr.Button("✨ Generate", variant="primary")

        with gr.Accordion("② All levels (coarse → fine) — latent & decoded RGB", open=True, elem_classes="tf-card"):
            gr.Markdown("_Single-click a **latent** tile to view it full size; then press the button "
                        "below it to use that level for editing._")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("<span class='tf-pill'>REFERENCE</span>")
                    gr.Markdown("Reference latent (per level) — click to enlarge", elem_classes="tf-gtitle")
                    ref_lat_gallery = gr.Gallery(show_label=False,
                                                 columns=4, height="auto", object_fit="contain", allow_preview=True)
                    ref_choose_btn = gr.Button(f"Use level {DEFAULT_EDIT_LEVEL} for REFERENCE editing",
                                               variant="secondary")
                    gr.Markdown("Reference decoded RGB (per level) — click to enlarge", elem_classes="tf-gtitle")
                    ref_dec_gallery = gr.Gallery(show_label=False,
                                                 columns=4, height="auto", object_fit="contain", allow_preview=True)
                with gr.Column():
                    gr.Markdown("<span class='tf-pill'>TARGET</span>")
                    gr.Markdown("Target latent (per level) — click to enlarge", elem_classes="tf-gtitle")
                    tgt_lat_gallery = gr.Gallery(show_label=False,
                                                 columns=4, height="auto", object_fit="contain", allow_preview=True)
                    tgt_choose_btn = gr.Button(f"Use level {DEFAULT_EDIT_LEVEL} for TARGET editing",
                                               variant="secondary")
                    gr.Markdown("Target decoded RGB (per level) — click to enlarge", elem_classes="tf-gtitle")
                    tgt_dec_gallery = gr.Gallery(show_label=False,
                                                 columns=4, height="auto", object_fit="contain", allow_preview=True)

        with gr.Accordion("③ Token exchange", open=True, elem_classes="tf-card"):
            level_status = gr.Markdown(_level_status(DEFAULT_EDIT_LEVEL, DEFAULT_EDIT_LEVEL))
            with gr.Row():
                cluster_select = gr.Checkbox(
                    value=False,
                    label="Cluster select (cosine) — one click grabs the whole similar-token cluster")
                cos_threshold = gr.Slider(0.5, 0.99, value=DEFAULT_COS_THRESHOLD, step=0.01,
                                          label="cluster similarity threshold")
            with gr.Row():
                with gr.Column():
                    ref_count = gr.Number(value=0, precision=0, label="Reference tokens selected", visible=False)
                    ref_text = gr.Textbox(label="Reference points (row,col)", interactive=False, visible=False)
                    gr.Markdown("Reference latent — click tokens to copy FROM", elem_classes="tf-gtitle")
                    ref_grid = gr.Image(show_label=False, interactive=False)
                    clear_ref = gr.Button("Clear reference selection")
                with gr.Column():
                    tgt_count = gr.Number(value=0, precision=0, label="Target tokens selected", visible=False)
                    tgt_text = gr.Textbox(label="Target points (row,col)", interactive=False, visible=False)
                    gr.Markdown("Target latent — click tokens to paste INTO", elem_classes="tf-gtitle")
                    tgt_grid = gr.Image(show_label=False, interactive=False)
                    clear_tgt = gr.Button("Clear target selection")
            run_btn = gr.Button("🚀 Run edit + generate", variant="primary")

        with gr.Accordion("④ Edited result — all levels (coarse → fine)", open=True, elem_classes="tf-card"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Edited latent (per level)", elem_classes="tf-gtitle")
                    edit_lat_gallery = gr.Gallery(show_label=False, columns=4,
                                                  height="auto", object_fit="contain", allow_preview=True)
                with gr.Column():
                    gr.Markdown("Edited decoded RGB (per level) — click to enlarge", elem_classes="tf-gtitle")
                    edit_dec_gallery = gr.Gallery(show_label=False, columns=4,
                                                  height="auto", object_fit="contain", allow_preview=True)

        # ---- wiring ----
        generate_btn.click(
            do_generate,
            inputs=[reference_class, target_class, seed, cos_threshold],
            outputs=[ref_levels_st, tgt_levels_st,
                     ref_lat_gallery, tgt_lat_gallery, ref_dec_gallery, tgt_dec_gallery,
                     ref_base_st, tgt_base_st, ref_grid, tgt_grid,
                     ref_pts_st, tgt_pts_st, ref_text, tgt_text, ref_count, tgt_count,
                     ref_level_st, tgt_level_st, level_status, palette_st,
                     ref_pending_st, tgt_pending_st, ref_choose_btn, tgt_choose_btn,
                     ref_region_ids_st, tgt_region_ids_st],
        )

        # live class name shown in each class-id field's label
        reference_class.change(lambda v: gr.update(label=_ref_label(v)), inputs=[reference_class],
                               outputs=[reference_class], queue=False, show_progress="hidden")
        target_class.change(lambda v: gr.update(label=_tgt_label(v)), inputs=[target_class],
                            outputs=[target_class], queue=False, show_progress="hidden")

        # single click a latent tile -> enlarge (native preview) + record which level is previewed
        ref_lat_gallery.select(preview_ref_level, inputs=None, outputs=[ref_pending_st, ref_choose_btn])
        tgt_lat_gallery.select(preview_tgt_level, inputs=None, outputs=[tgt_pending_st, tgt_choose_btn])

        # button commits the previewed level for editing (loads its 16x16 token grid)
        ref_choose_btn.click(
            choose_ref_level, inputs=[ref_levels_st, ref_pending_st, tgt_level_st, palette_st, cos_threshold],
            outputs=[ref_level_st, ref_base_st, ref_grid, ref_pts_st, ref_text, ref_count, level_status,
                     ref_choose_btn, ref_region_ids_st],
        )
        tgt_choose_btn.click(
            choose_tgt_level, inputs=[tgt_levels_st, tgt_pending_st, ref_level_st, palette_st, cos_threshold],
            outputs=[tgt_level_st, tgt_base_st, tgt_grid, tgt_pts_st, tgt_text, tgt_count, level_status,
                     tgt_choose_btn, tgt_region_ids_st],
        )

        # re-cluster both committed levels when the threshold changes
        cos_threshold.change(
            recompute_regions, inputs=[ref_levels_st, tgt_levels_st, ref_level_st, tgt_level_st, cos_threshold],
            outputs=[ref_region_ids_st, tgt_region_ids_st], queue=False, show_progress="hidden")

        def _sel(points, base, region_on, region_ids, evt: gr.SelectData):
            pts, text, n, overlay = _toggle_point_or_region_from_image(
                points, base, bool(region_on), region_ids, evt, size=GRID)
            return pts, text, n, overlay

        # token toggles are tiny CPU ops — run un-queued with no progress UI so they feel instant
        ref_grid.select(_sel, inputs=[ref_pts_st, ref_base_st, cluster_select, ref_region_ids_st],
                        outputs=[ref_pts_st, ref_text, ref_count, ref_grid], queue=False, show_progress="hidden")
        tgt_grid.select(_sel, inputs=[tgt_pts_st, tgt_base_st, cluster_select, tgt_region_ids_st],
                        outputs=[tgt_pts_st, tgt_text, tgt_count, tgt_grid], queue=False, show_progress="hidden")

        clear_ref.click(lambda b: _clear_points(b, size=GRID), inputs=[ref_base_st],
                        outputs=[ref_pts_st, ref_text, ref_count, ref_grid],
                        queue=False, show_progress="hidden")
        clear_tgt.click(lambda b: _clear_points(b, size=GRID), inputs=[tgt_base_st],
                        outputs=[tgt_pts_st, tgt_text, tgt_count, tgt_grid],
                        queue=False, show_progress="hidden")

        run_btn.click(
            do_edit,
            inputs=[tgt_levels_st, ref_levels_st, ref_level_st, tgt_level_st,
                    ref_pts_st, tgt_pts_st, reference_class, target_class, seed, palette_st],
            outputs=[edit_lat_gallery, edit_dec_gallery],
        )
    return demo


def _warmup():
    """Build the pipeline + trigger JIT compile and decoder load once at startup,
    so the first user click is as fast as subsequent ones (instead of paying the
    full model-build + XLA-compile cost on the first Generate). Returns True once
    the model is built, compiled, and the decoder is warm — i.e. Generate is ready."""
    import time
    t0 = time.perf_counter()
    print("[warmup] starting...", flush=True)
    try:
        from tf_pipeline import get_pipeline
        pipe = get_pipeline()
        t1 = time.perf_counter(); print(f"[warmup] build+load: {t1 - t0:.1f}s", flush=True)
        levels = pipe.generate(0, seed=int(FIXED_SEED))  # compiles gen_fn
        t2 = time.perf_counter(); print(f"[warmup] generate/compile: {t2 - t1:.1f}s", flush=True)
        pipe.decode_last(levels)                          # builds + warms decoder
        t3 = time.perf_counter(); print(f"[warmup] decoder build+warm: {t3 - t2:.1f}s", flush=True)
        print(f"[warmup] READY in {t3 - t0:.1f}s total", flush=True)
        return True
    except Exception as e:
        print(f"[warmup] failed ({e!r})", flush=True)
        return False


if __name__ == "__main__":
    demo = build_demo().queue()
    # Warm up BEFORE launching so the server only starts serving once the model is
    # built, JIT-compiled, and the decoder is loaded — the "Running on ..." line and
    # the page becoming reachable then signal that Generate is ready (no hang on the
    # first click). Set TF_WARMUP=0 to skip and launch immediately (first click pays
    # the full compile cost; useful for quickly checking the UI without a GPU).
    if os.environ.get("TF_WARMUP", "1") != "0":
        _warmup()
    demo.launch(**launch_kwargs())  # theme + css go on launch() in Gradio 6
