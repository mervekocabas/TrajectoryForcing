import importlib.util
import logging
import os
import sys
import types
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# Decoder source loader (importlib, no package install required)
# ---------------------------------------------------------------------------

def _load_decoder_classes(code_dir: str):
    """Load GeneralDecoder/ViTMAEConfig from `decoder.py` + `utils.py`."""
    code_dir = os.path.abspath(os.path.expanduser(code_dir))
    if not os.path.isdir(code_dir):
        raise FileNotFoundError(f"RAE decoder code_dir not found: {code_dir!r}")

    utils_py = os.path.join(code_dir, "utils.py")
    decoder_py = os.path.join(code_dir, "decoder.py")
    for p in (utils_py, decoder_py):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing RAE decoder source file: {p}")

    pkg_name = f"_fdloss_rae_decoder_{abs(hash(code_dir))}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [code_dir]
        sys.modules[pkg_name] = pkg

    def _load(mod_name: str, path: str):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    utils_mod = _load(f"{pkg_name}.utils", utils_py)
    decoder_mod = _load(f"{pkg_name}.decoder", decoder_py)
    return decoder_mod.GeneralDecoder, utils_mod.ViTMAEConfig


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    state = obj
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    if any(str(k).startswith("decoder.") for k in state.keys()):
        return {k[len("decoder."):]: v for k, v in state.items() if str(k).startswith("decoder.")}
    return state


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Default path resolution (env vars + sensible fallbacks under user's tree)
# ---------------------------------------------------------------------------

# Repo-relative defaults (resolved from third_party/fd_loss, where the launch
# scripts cd). Reuse the TF repo's RAE decoder source/config and the decoder
# weights it auto-downloads under checkpoints/rae/. Post-training always runs
# with --rae_stats_path none (no latent de-normalization), so no stats path.
DEFAULTS = {
    "decoder_path":  "../../checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
    "config_path":   "../rae_decoder/configs/ViTXL",
    "code_dir":      "../rae_decoder",
}


def _resolve(arg_value: Optional[str], env_key: str, default_key: str) -> str:
    if arg_value:
        return os.path.abspath(os.path.expanduser(arg_value))
    env = os.environ.get(env_key)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return DEFAULTS[default_key]


# ---------------------------------------------------------------------------
# RAEDecoder tokenizer
# ---------------------------------------------------------------------------

class RAEDecoder(nn.Module):
    """DINOv2-RAE ViTXL decoder, exposed via the FD-Loss tokenizer interface.

    Latent layout: (B, latent_dim, latent_hw, latent_hw), torch NCHW.
    Output layout: (B, 3, image_size, image_size).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        decoder_path: Optional[str] = None,
        stats_path: Optional[str] = None,
        config_path: Optional[str] = None,
        code_dir: Optional[str] = None,
        latent_dim: int = 768,
        latent_hw: int = 16,
        decoder_patch_size: int = 16,
        eps: float = 1e-5,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.latent_hw = int(latent_hw)
        self.decoder_patch_size = int(decoder_patch_size)
        self.image_size = self.decoder_patch_size * self.latent_hw
        self.eps = float(eps)

        decoder_path = _resolve(decoder_path, "FDL_RAE_DECODER_PATH", "decoder_path")
        config_path = _resolve(config_path, "FDL_RAE_CONFIG_PATH", "config_path")
        code_dir = _resolve(code_dir, "FDL_RAE_CODE_DIR", "code_dir")

        # Allow disabling latent (de)normalization entirely: pass --rae_stats_path
        # to a sentinel ("none"/"null"/"identity"/"off") or set FDL_RAE_STATS_PATH
        # to one. Mirrors the JAX eval's normalization_stat_path="" (identity
        # denormalize, no stat.pt) -- the single biggest lever on the score.
        _DISABLE = {"none", "null", "identity", "off"}
        _stats_arg = stats_path.strip().lower() if isinstance(stats_path, str) else ""
        _stats_env = os.environ.get("FDL_RAE_STATS_PATH", "").strip().lower()
        self.normalize_latents = not (_stats_arg in _DISABLE or _stats_env in _DISABLE)
        if self.normalize_latents:
            stats_path = _resolve(stats_path, "FDL_RAE_STATS_PATH", "stats_path")
        else:
            stats_path = None

        _check = [("decoder_path", decoder_path), ("config_path", config_path),
                  ("code_dir", code_dir)]
        if self.normalize_latents:
            _check.append(("stats_path", stats_path))
        for label, p in _check:
            if not os.path.exists(p):
                raise FileNotFoundError(f"RAE {label} not found: {p}")

        logger.info(f"[RAE] code_dir={code_dir}")
        logger.info(f"[RAE] config={config_path}")
        logger.info(f"[RAE] decoder weights={decoder_path}")
        logger.info(f"[RAE] stats={stats_path if self.normalize_latents else 'DISABLED (identity denormalize, no stat.pt)'}")

        # Load decoder source classes & config
        GeneralDecoder, ViTMAEConfig = _load_decoder_classes(code_dir)
        cfg = ViTMAEConfig.from_pretrained(config_path)
        cfg.hidden_size = self.latent_dim
        cfg.patch_size = self.decoder_patch_size
        cfg.image_size = self.image_size
        num_patches = self.latent_hw * self.latent_hw

        decoder = GeneralDecoder(cfg, num_patches=num_patches)
        state = _extract_state_dict(_torch_load(decoder_path))
        msg = decoder.load_state_dict(state, strict=False)
        if msg.missing_keys:
            logger.warning(f"[RAE] missing decoder keys: {len(msg.missing_keys)} "
                           f"(first 5: {msg.missing_keys[:5]})")
        if msg.unexpected_keys:
            logger.warning(f"[RAE] unexpected decoder keys: {len(msg.unexpected_keys)} "
                           f"(first 5: {msg.unexpected_keys[:5]})")

        # Freeze parameters: we never update the decoder weights, but we DO
        # let gradients flow through it to the latent input during FD-Loss training.
        for p in decoder.parameters():
            p.requires_grad_(False)
        decoder = decoder.to(dtype=torch_dtype).eval()
        self.decoder = decoder
        self.torch_dtype = torch_dtype

        # Load latent normalization stats (mean/var over DINOv2 latents).
        if self.normalize_latents:
            stats = _torch_load(stats_path)
            if not isinstance(stats, dict) or "var" not in stats:
                raise ValueError(f"RAE stats {stats_path} missing 'var' key. Contents: {list(stats.keys()) if hasattr(stats, 'keys') else type(stats)}")
            var = torch.as_tensor(stats["var"], dtype=torch.float32)
            mean = stats.get("mean", None)
            if mean is not None:
                mean = torch.as_tensor(mean, dtype=torch.float32)
            else:
                mean = torch.zeros(self.latent_dim, dtype=torch.float32)
                logger.warning(f"[RAE] stats has mean=None; using zero-mean denormalization")
            if var.ndim == 1:
                var = var.view(1, -1, 1, 1)
            if mean.ndim == 1:
                mean = mean.view(1, -1, 1, 1)
            # Precompute std = sqrt(var + eps) so denormalize is one mul + one add.
            std = torch.sqrt(var + self.eps)
        else:
            # Identity denormalize: denormalize_z(z) = z * 1 + 0 = z.
            mean = torch.zeros(1, self.latent_dim, 1, 1, dtype=torch.float32)
            std = torch.ones(1, self.latent_dim, 1, 1, dtype=torch.float32)
        self.register_buffer("lat_mean", mean, persistent=False)
        self.register_buffer("lat_std", std, persistent=False)

        # ImageNet (de)normalization for the decoder's pixel output.
        im_mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        im_std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", im_mean, persistent=False)
        self.register_buffer("image_std", im_std, persistent=False)

        n_params = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        logger.info(f"[RAE] decoder params: {n_params:.1f}M, "
                    f"latent={self.latent_dim}x{self.latent_hw}x{self.latent_hw} "
                    f"-> image={self.image_size}x{self.image_size}, "
                    f"dtype={torch_dtype}")

    # FD-Loss tokenizer interface ------------------------------------------------

    def denormalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Undo training-time latent normalization: z * std + mean."""
        return z * self.lat_std.to(z) + self.lat_mean.to(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixels in [-1, 1] (FD-Loss convention).

        z: (B, latent_dim, H, W) — in the decoder's native (denormalized) space.
        Differentiable: gradients flow through the decoder back to z.
        """
        if z.ndim != 4 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected (B, {self.latent_dim}, H, W) latent, got {tuple(z.shape)}"
            )
        b, c, h, w = z.shape
        if h != self.latent_hw or w != self.latent_hw:
            raise ValueError(
                f"expected spatial {self.latent_hw}x{self.latent_hw}, got {h}x{w}"
            )

        z_in = z.to(dtype=self.torch_dtype)
        # (B, C, H, W) -> (B, H*W, C) tokens
        z_seq = z_in.flatten(2).transpose(1, 2).contiguous()
        out = self.decoder(z_seq, drop_cls_token=False).logits
        x = self.decoder.unpatchify(out)              # (B, 3, image_size, image_size)
        # decoder output is in ImageNet-normalized space; map to [0, 1]
        x = x * self.image_std.to(x) + self.image_mean.to(x)
        # FD-Loss expects [-1, 1] (it does *0.5+0.5 to get [0,1] for judges)
        x = x * 2.0 - 1.0
        return x

    @torch.inference_mode()
    def detokenize(self, z: torch.Tensor, decode_bsz: Optional[int] = None) -> torch.Tensor:
        """Decode (chunked, no-grad) to pixels in [0, 1] for vis/eval."""
        if decode_bsz is None:
            pixels_per_sample = self.image_size * self.image_size
            decode_bsz = max(1, 64 * (32 * 32) // pixels_per_sample)
        z_bsz = z.shape[0]
        if z_bsz <= decode_bsz:
            return torch.clamp(self.decode(self.denormalize_z(z)) * 0.5 + 0.5, 0.0, 1.0)
        out_shape = self.decode(self.denormalize_z(z[:1])).shape
        out = torch.empty(z_bsz, *out_shape[1:], device=z.device)
        for i in range(0, z_bsz, decode_bsz):
            chunk = self.decode(self.denormalize_z(z[i:i+decode_bsz]))
            out[i:i+decode_bsz] = torch.clamp(chunk * 0.5 + 0.5, 0.0, 1.0)
        return out

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use decode() / detokenize() / denormalize_z(); RAEDecoder.forward is unused")


RAE_models = ["rae_dinov2_b_vitxl"]
