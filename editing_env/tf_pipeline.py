#!/usr/bin/env python3
"""Load-once Trajectory Forcing inference pipeline (interactive editing env).

Wraps the generation + token-exchange math into a single object that loads the
JAX flow model + RAE/DINOv2 decoder ONCE and reuses them across requests. All JAX
imports happen lazily (inside the Pipeline constructor), so importing this module
does NOT touch the GPU until inference actually runs.

Public API
----------
    pipe = get_pipeline()                       # cached, builds on first call
    levels = pipe.generate(class_id, seed)      # np [L,H,W,C]
    edited = pipe.edit(target_levels, ref_levels, ref_idx, tgt_idx, ref_pts, tgt_pts, class_id, seed)
    rgb    = pipe.decode_last(levels)           # uint8 [Hd,Wd,3] final image
    rgbs   = pipe.decode_all(levels)            # list[uint8] per level
    tiles  = pipe.pca_tiles(levels)             # list[uint8 HxWx3] per-level PCA grid

Env / config
------------
    TF_CONFIG       config spec        (default: configs/edit_env_config.yml)
    TF_LOAD_FROM    local checkpoint path  (overrides the HuggingFace download)
    TF_CKPT_REPO    HF model repo to pull the ckpt from
                    (default: mervekocabas/TrajectoryForcing)
    TF_CKPT_FILE    file in that repo  (default: TF_L_edit)
    HF_TOKEN        read token (only needed if the repo is private)
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# tf_pipeline.py lives in editing_env/; the model/config/checkpoint machinery
# (pmf, utils/, configs/, third_party/) lives in the repo root one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = os.environ.get("TF_CONFIG", "configs/edit_env_config.yml")

# Where the editing checkpoint lives on HuggingFace (single file at repo root).
DEFAULT_CKPT_REPO = "mervekocabas/TrajectoryForcing"
DEFAULT_CKPT_FILE = "TF_L_edit"

_PIPELINE = None
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# checkpoint resolution (pull from the public HF model repo by default)
# ---------------------------------------------------------------------------
def _resolve_checkpoint() -> str:
    explicit = os.environ.get("TF_LOAD_FROM", "").strip()
    if explicit:
        return explicit
    repo = os.environ.get("TF_CKPT_REPO", DEFAULT_CKPT_REPO).strip()
    fname = os.environ.get("TF_CKPT_FILE", DEFAULT_CKPT_FILE).strip()
    if repo and fname:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo,
            filename=fname,
            repo_type="model",
            token=os.environ.get("HF_TOKEN"),
        )
    return ""  # fall back to config.load_from


# ---------------------------------------------------------------------------
# config loading (mirrors the yaml-merge done by the eval scripts)
# ---------------------------------------------------------------------------
def _load_config(spec: str):
    spec = str(spec).strip()
    if ":" in spec:
        lhs, rhs = (s.strip() for s in spec.split(":", 1))
        if lhs.endswith("load_config.py"):
            from configs.load_config import get_config
            return get_config(rhs)
        raise ValueError(f"Unsupported config spec {spec!r}")
    if spec.endswith((".yml", ".yaml")):
        import yaml
        from configs.default import get_config as get_default_config
        cfg = get_default_config()
        with open(spec, "r") as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        return cfg
    from configs.load_config import get_config
    return get_config(spec)


def _prepare_ema_keys_for_restore(config, load_from: str) -> None:
    try:
        sampling = getattr(config, "sampling", None)
        if sampling is not None and hasattr(sampling, "get") and "emas" in sampling:
            emas = sampling.get("emas", None)
            if emas is not None:
                config.training.ema_val = emas
                return
    except Exception:
        pass
    try:
        from flax.training import checkpoints
        raw = checkpoints.restore_checkpoint(str(load_from), target=None)
        raw_ema = raw.get("ema_params", None) if isinstance(raw, dict) else None
        if isinstance(raw_ema, dict) and raw_ema:
            out = []
            for k in raw_ema.keys():
                try:
                    v = float(k)
                    out.append(int(v) if float(v).is_integer() else float(v))
                except Exception:
                    continue
            if out:
                config.training.ema_val = out
    except Exception:
        pass


def _load_eval_params_fast(load_from: str):
    """Read the checkpoint once (target=None) and return the params we use for eval:
    the highest-EMA params if present, else the raw params. Avoids model.init and the
    per-EMA deepcopies done by create_train_state. Returns None if nothing usable."""
    from flax.training import checkpoints
    raw = checkpoints.restore_checkpoint(str(load_from), target=None)
    if not isinstance(raw, dict):
        return None
    ema = raw.get("ema_params", None)
    if isinstance(ema, dict) and ema:
        numeric = {}
        for k in ema.keys():
            try:
                numeric[float(k)] = k
            except (TypeError, ValueError):
                continue
        if numeric:
            chosen = numeric[max(numeric)]  # mirror _choose_params(ema=None)
            return ema[chosen]
    return raw.get("params", None)


def _choose_params(state, ema: float | None = None):
    ema_keys = list(getattr(state, "ema_params", {}).keys())
    if not ema_keys:
        return state.params
    numeric = {float(k): k for k in ema_keys}
    chosen_f = max(numeric) if ema is None else min(numeric, key=lambda kf: abs(kf - float(ema)))
    return state.ema_params[numeric[chosen_f]]


def _resolve_sampling_hparams(config):
    sampling = config.sampling
    omega = float(sampling.get("omega", 1.0)) if hasattr(sampling, "get") else 1.0
    if hasattr(sampling, "get") and "t_min" in sampling and "t_max" in sampling:
        t_min, t_max = float(sampling.get("t_min")), float(sampling.get("t_max"))
    else:
        interval = sampling.get("interval", None) if hasattr(sampling, "get") else None
        if interval:
            t_min, t_max = float(interval[0][0]), float(interval[0][1])
        else:
            t_min, t_max = 0.0, 1.0
    return omega, t_min, t_max


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, config_spec: str = DEFAULT_CONFIG, load_from: str = ""):
        import jax
        from jax import random
        from flax import jax_utils
        from functools import partial

        from pmf import pixelMeanFlow
        from utils.ckpt_util import restore_checkpoint
        from utils.lr_utils import lr_schedules
        from utils.rae_decoder import get_decoder
        from utils.trainstate_util import create_train_state

        self._jax = jax
        self._jnp = __import__("jax.numpy", fromlist=["numpy"])
        self._random = random
        self._jax_utils = jax_utils

        config = _load_config(config_spec)
        load_from = str(load_from or getattr(config, "load_from", "")).strip()
        if not load_from:
            raise ValueError("No checkpoint path. Set TF_LOAD_FROM / TF_CKPT_REPO or config.load_from.")
        config.load_from = str(Path(load_from).expanduser())
        config.eval_only = True
        _prepare_ema_keys_for_restore(config, config.load_from)
        if not getattr(config, "rae_decoder", None):
            raise ValueError("Config must contain rae_decoder settings for latent decoding.")
        config.rae_decoder.enabled = True

        self.config = config
        self.num_levels = int(config.dataset.get("num_levels", 4))
        self.num_classes = int(config.dataset.num_classes)
        self.num_steps = int(config.sampling.num_steps)
        self.omega, self.t_min, self.t_max = _resolve_sampling_hparams(config)

        # --- build model (mirrors train.py's eval model-config assembly) ---
        rng = random.key(0)
        image_size = int(config.dataset.image_size)
        lr_fn = lr_schedules(config, 1000)
        model_config = config.model.to_dict()
        model_config["num_classes"] = int(config.dataset.num_classes)
        model_config["input_size"] = int(config.dataset.image_size)
        model_config["in_channels"] = int(config.dataset.image_channels)
        model_config["num_levels"] = self.num_levels
        if str(config.dataset.get("kind", "imagenet")).lower() in {"latent_hier", "latent"}:
            model_config["use_token_embed"] = True
            model_config["use_prev_cond"] = True
            model_config["use_level_cond"] = True
        # release default.py carries a training-only key that pixelMeanFlow does not accept
        model_config.pop("data_proportion_schedule", None)
        model = pixelMeanFlow(**model_config, eval=True)

        params = None
        if os.environ.get("TF_FAST_LOAD", "1") != "0":
            # Fast path: read the checkpoint once and pull the single EMA we actually use,
            # skipping model.init (a full forward + JIT compile) and the per-EMA deepcopies
            # that create_train_state does (the source of the slow/heavy warmup).
            try:
                params = _load_eval_params_fast(config.load_from)
            except Exception as e:  # fall back to the original, known-good path
                print(f"[tf_pipeline] fast load failed ({e!r}); using full restore path", flush=True)
                params = None
        if params is None:
            state = create_train_state(rng, config, model, image_size, lr_fn)
            state = restore_checkpoint(state, config.load_from, params_only=True)
            params = _choose_params(state, ema=None)
        self.model = model
        self.variable_repl = jax_utils.replicate({"params": params})

        # pmapped generate/edit closures are attached by _install_steps() (see _bind()).
        self._gen_fn = None
        self._edit_fn = None

        # --- RAE decoder (PyTorch); lazily (re)built per latent shape ---
        self._get_decoder = get_decoder
        self._decoder = None

        # PCA viz helpers (per-image fit, plus shared-palette fit/apply)
        from utils.vis_util import latent_levels_to_pca_column, fit_pca_palette, apply_pca_palette
        self._pca_col = latent_levels_to_pca_column
        self._fit_palette = fit_pca_palette
        self._apply_palette = apply_pca_palette

    # ----- decoder -----
    def _ensure_decoder(self, h: int, c: int):
        if self._decoder is None or int(self._decoder.latent_hw) != int(h) or int(self._decoder.latent_dim) != int(c):
            self.config.rae_decoder.latent_hw = int(h)
            self.config.rae_decoder.latent_dim = int(c)
            self._decoder = self._get_decoder(self.config, latent_hw=int(h), latent_dim=int(c))
            if self._decoder is None:
                raise RuntimeError("RAE decoder not configured (config.rae_decoder).")
        return self._decoder

    def decode_all(self, levels_hwc: np.ndarray) -> List[np.ndarray]:
        arr = np.asarray(levels_hwc, dtype=np.float32)  # [L,H,W,C]
        _, h, w, c = arr.shape
        dec = self._ensure_decoder(h, c)
        levels_bchw = np.transpose(arr, (0, 3, 1, 2))
        return list(dec.decode_uint8(levels_bchw))  # [L][Hd,Wd,3]

    def decode_last(self, levels_hwc: np.ndarray) -> np.ndarray:
        # Only the final level is displayed; decode just that one (avoids decoding
        # all L levels through the ViT-XL decoder and throwing L-1 of them away).
        arr = np.asarray(levels_hwc, dtype=np.float32)  # [L,H,W,C]
        last = arr[-1:]                                  # [1,H,W,C]
        _, h, w, c = last.shape
        dec = self._ensure_decoder(h, c)
        last_bchw = np.transpose(last, (0, 3, 1, 2))     # [1,C,H,W]
        return list(dec.decode_uint8(last_bchw))[0]

    def fit_palette(self, levels_list):
        """Fit a shared PCA color palette from one or more [L,H,W,C] latent stacks."""
        return self._fit_palette(levels_list)

    def pca_tiles(self, levels_hwc: np.ndarray, palette=None) -> List[np.ndarray]:
        arr = np.asarray(levels_hwc, dtype=np.float32)
        L, h, w, c = arr.shape
        if palette is None:
            col = self._pca_col(arr, gap=0, bg_value=0)            # per-image fit
        else:
            col = self._apply_palette(arr, palette, gap=0, bg_value=0)  # shared palette
        return [np.asarray(col[i * h:(i + 1) * h], dtype=np.uint8) for i in range(L)]


# ---------------------------------------------------------------------------
# step closures (defined at module scope to mirror the scripts exactly)
# ---------------------------------------------------------------------------
def _install_steps(pipe: "Pipeline"):
    """Attach the concrete pmapped generate/edit closures."""
    import jax
    import jax.numpy as jnp
    from jax import random
    from flax import jax_utils
    from functools import partial

    model = pipe.model
    config = pipe.config
    num_steps = pipe.num_steps
    num_classes = pipe.num_classes

    def gen_step(variable, sample_idx, rng_init, omega, t_min, t_max):
        rng_sample = random.fold_in(rng_init, sample_idx)
        img = int(config.dataset.image_size)
        ch = int(config.dataset.image_channels)
        x_shape = (1, img, img, ch)
        rng_work, _ = random.split(rng_sample)
        y = (jnp.arange(1, dtype=jnp.int32) + sample_idx) % num_classes
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

        def sample_one(level_ids, cond, rng_in):
            rng_out, rng_l = random.split(rng_in)
            z_t = jax.random.normal(rng_l, x_shape, dtype=model.dtype) * model.noise_scale
            def step_fn(i, x_i):
                return model.apply(variable, x_i, y, level_ids, i, t_steps, omega, t_min, t_max,
                                   cond, method=model.sample_one_step)
            return jax.lax.fori_loop(0, num_steps, step_fn, z_t), rng_out

        nlev = int(getattr(model, "num_levels", 4))
        prev = jnp.zeros(x_shape, dtype=model.dtype)
        outs = []
        for lid in range(nlev):
            level_ids = jnp.full((1,), lid, dtype=jnp.int32)
            lvl, rng_work = sample_one(level_ids, prev, rng_work)
            outs.append(lvl); prev = lvl
        return jnp.stack(outs, axis=1)  # [1, L, H, W, C]

    def edit_step(variable, sample_idx, start_level, rng_init, start_idx, omega, t_min, t_max):
        rng_sample = random.fold_in(rng_init, sample_idx)
        h0, w0, c0 = start_level.shape
        x_shape = (1, h0, w0, c0)
        rng_work, _ = random.split(rng_sample)
        y = (jnp.arange(1, dtype=jnp.int32) + sample_idx) % num_classes
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

        def sample_one(level_id, cond, rng_in):
            rng_out, rng_l = random.split(rng_in)
            z_t = jax.random.normal(rng_l, x_shape, dtype=model.dtype) * model.noise_scale
            level_ids = jnp.full((1,), int(level_id), dtype=jnp.int32)
            def step_fn(i, x_i):
                return model.apply(variable, x_i, y, level_ids, i, t_steps, omega, t_min, t_max,
                                   cond, method=model.sample_one_step)
            return jax.lax.fori_loop(0, num_steps, step_fn, z_t), rng_out

        nlev = int(getattr(model, "num_levels", 4))
        zero = jnp.zeros(x_shape, dtype=model.dtype)
        outs = []
        prev = None
        for lid in range(nlev):
            if lid < int(start_idx):
                outs.append(zero); continue
            if lid == int(start_idx):
                lvl = jnp.asarray(start_level, dtype=model.dtype)[None, ...]
                outs.append(lvl); prev = lvl; continue
            lvl, rng_work = sample_one(lid, prev, rng_work)
            outs.append(lvl); prev = lvl
        return jnp.stack(outs, axis=1)  # [1, L, H, W, C]

    pipe._gen_fn = jax.pmap(gen_step, axis_name="batch")
    pipe._edit_fn = jax.pmap(edit_step, axis_name="batch", static_broadcasted_argnums=(4,))
    pipe._jax_utils = jax_utils
    pipe._random = random
    pipe._jnp = jnp
    pipe._jax = jax


def _generate(pipe: "Pipeline", class_id: int, seed: int) -> np.ndarray:
    jax, jnp, random, jax_utils = pipe._jax, pipe._jnp, pipe._random, pipe._jax_utils
    sample_idx = int(class_id)  # label = sample_idx % num_classes
    kw = jax_utils.replicate({"omega": float(pipe.omega), "t_min": float(pipe.t_min), "t_max": float(pipe.t_max)})
    rng_init = jax_utils.replicate(random.PRNGKey(int(seed)))
    sidx = jnp.asarray([sample_idx], dtype=jnp.int32)  # one device
    out = pipe._gen_fn(pipe.variable_repl, sample_idx=sidx, rng_init=rng_init, **kw)
    out = np.asarray(jax.device_get(out), dtype=np.float32)  # [D,1,L,H,W,C]
    out = out.reshape(-1, *out.shape[2:])[0]  # [L,H,W,C]
    return out


def _edit(pipe, target_levels, ref_levels, ref_idx, tgt_idx, ref_points, tgt_points, class_id, seed) -> np.ndarray:
    """Copy tokens from ref level `ref_idx` into target level `tgt_idx`, then re-sample
    the target levels below `tgt_idx`. ref_idx and tgt_idx may differ (all levels share
    the same H,W,C). Levels above `tgt_idx` keep the original target latents so the full
    stack is coherent for display."""
    jax, jnp, random, jax_utils = pipe._jax, pipe._jnp, pipe._random, pipe._jax_utils
    tgt = np.asarray(target_levels, dtype=np.float32)
    ref = np.asarray(ref_levels, dtype=np.float32)
    ri = int(ref_idx)
    ti = int(tgt_idx)
    edited_source = np.array(tgt[ti], dtype=np.float32, copy=True)
    ref_source = ref[ri]
    rps = list(ref_points)
    tps = list(tgt_points)
    # Each target token receives a reference token, cycling through the reference
    # selections (counts need not match: 1 ref + N targets broadcasts to all N).
    for j, (dr, dc) in enumerate(tps):
        sr, sc = rps[j % len(rps)]
        edited_source[int(dr), int(dc)] = ref_source[int(sr), int(sc)]

    last = pipe.num_levels - 1
    if ti >= last:
        generated = np.array(tgt, dtype=np.float32, copy=True)
        generated[ti] = edited_source
        return generated

    kw = jax_utils.replicate({"omega": float(pipe.omega), "t_min": float(pipe.t_min), "t_max": float(pipe.t_max)})
    rng_init = jax_utils.replicate(random.PRNGKey(int(seed)))
    # downstream re-sampling must keep the TARGET class label (label = sample_idx % num_classes)
    sample_idx = int(class_id)
    sidx = jnp.asarray([sample_idx], dtype=jnp.int32)
    start = jnp.asarray(edited_source[None], dtype=jnp.float32)  # one device -> [1,H,W,C]
    out = pipe._edit_fn(pipe.variable_repl, sidx, start, rng_init, ti,
                        kw["omega"], kw["t_min"], kw["t_max"])
    out = np.array(jax.device_get(out), dtype=np.float32)    # [D,1,L,H,W,C] (writable copy)
    out = np.array(out.reshape(-1, *out.shape[2:])[0])       # [L,H,W,C], writable
    # levels above the edited one were zero-filled by edit_step; restore originals for display
    for i in range(ti):
        out[i] = tgt[i]
    return out


# bind instance methods
def _bind(pipe: Pipeline):
    _install_steps(pipe)
    pipe.generate = lambda class_id, seed=666: _generate(pipe, class_id, seed)
    pipe.edit = lambda target_levels, ref_levels, ref_idx, tgt_idx, ref_points, tgt_points, class_id, seed=666: \
        _edit(pipe, target_levels, ref_levels, ref_idx, tgt_idx, ref_points, tgt_points, class_id, seed)
    return pipe


def build_pipeline(config_spec: str | None = None, load_from: str | None = None) -> Pipeline:
    cfg = config_spec or DEFAULT_CONFIG
    lf = load_from if load_from is not None else _resolve_checkpoint()
    pipe = Pipeline(config_spec=cfg, load_from=lf)
    return _bind(pipe)


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                _PIPELINE = build_pipeline()
    return _PIPELINE
