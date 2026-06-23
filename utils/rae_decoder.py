import importlib.util
import math
import os
import sys
import types
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


_DECODER_CACHE: Dict[Tuple[Any, ...], "RAEDecoderOnly"] = {}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            return cfg.get(key) if key in cfg else default
    return getattr(cfg, key, default)


# Public HuggingFace collection holding the RAE decoders + latent stats.
_RAE_HF_REPO = "nyu-visionx/RAE-collections"
_RAE_LOCAL_MARKER = "checkpoints/rae/"


def _maybe_hf_download(local_path, repo_id=_RAE_HF_REPO, auto_download=True):
    """Fetch a missing RAE file from HuggingFace, mirroring DINOv2's auto-download.

    If ``local_path`` is missing and lives under a ``checkpoints/rae/`` root, the
    portion after that marker is treated as the repo-relative filename and pulled
    from ``repo_id`` into the same root (so eval/inference works without running
    scripts/download_models.sh first). Paths that already exist, are empty, or do
    not contain the marker are returned unchanged (the caller then errors clearly).
    """
    if not local_path or os.path.exists(local_path) or not auto_download:
        return local_path
    norm = str(local_path).replace(os.sep, "/")
    if _RAE_LOCAL_MARKER not in norm:
        return local_path  # cannot infer the repo-relative path; leave as-is
    head, rel = norm.split(_RAE_LOCAL_MARKER, 1)
    local_root = head + _RAE_LOCAL_MARKER.rstrip("/")
    from huggingface_hub import hf_hub_download

    try:
        from utils.logging_util import log_for_0

        log_for_0("RAE decoder: %r missing locally; downloading from %s ...", rel, repo_id)
    except Exception:
        print(f"RAE decoder: {rel} missing locally; downloading from {repo_id} ...")
    hf_hub_download(repo_id=repo_id, filename=rel, local_dir=local_root)
    return local_path


def _load_decoder_classes_from_files(code_dir: Optional[str]):
    """Load GeneralDecoder/ViTMAEConfig from decoder.py + utils.py directly."""
    code_dir = _resolve_path(code_dir or "third_party/rae_decoder")
    if not code_dir or not os.path.isdir(code_dir):
        raise FileNotFoundError(f"RAE decoder code_dir not found: {code_dir!r}")

    utils_py = os.path.join(code_dir, "utils.py")
    decoder_py = os.path.join(code_dir, "decoder.py")
    if not os.path.exists(utils_py) or not os.path.exists(decoder_py):
        raise FileNotFoundError(
            f"Expected decoder source files at {code_dir} (need utils.py and decoder.py)"
        )

    # Create a temporary package so decoder.py can resolve `from .utils import ...`.
    pkg_name = f"_pmf_rae_decoder_{abs(hash(code_dir))}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [code_dir]  # namespace package path
        sys.modules[pkg_name] = pkg

    def _load_submodule(mod_name: str, path: str):
        if mod_name in sys.modules:
            return sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    utils_mod = _load_submodule(f"{pkg_name}.utils", utils_py)
    decoder_mod = _load_submodule(f"{pkg_name}.decoder", decoder_py)
    return decoder_mod.GeneralDecoder, utils_mod.ViTMAEConfig


def _is_enabled(config) -> bool:
    dec_cfg = getattr(config, "rae_decoder", None)
    return bool(dec_cfg is not None and _cfg_get(dec_cfg, "enabled", False))


def _torch_load_compat(torch, path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    else:
        state = obj

    # Support checkpoints saved with a "decoder." prefix.
    if any(str(k).startswith("decoder.") for k in state.keys()):
        stripped = {}
        for k, v in state.items():
            k = str(k)
            if k.startswith("decoder."):
                stripped[k[len("decoder."):]] = v
        if len(stripped) > 0:
            return stripped
    return state


class RAEDecoderOnly:
    """Minimal RAE decoder-only wrapper for DINOv2 latent -> RGB decoding."""

    def __init__(
        self,
        *,
        decoder_config_path: str,
        pretrained_decoder_path: str,
        latent_dim: int = 768,
        latent_hw: int = 16,
        decoder_patch_size: int = 16,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        device: Optional[str] = None,
        decoder_code_dir: Optional[str] = None,
        auto_download: bool = True,
        hf_repo_id: str = _RAE_HF_REPO,
    ):
        import_err = None
        try:
            import torch
            from third_party.rae_decoder import GeneralDecoder, ViTMAEConfig
        except Exception as e:  # pragma: no cover - runtime dependency path
            import_err = e
            try:
                import torch
                GeneralDecoder, ViTMAEConfig = _load_decoder_classes_from_files(
                    decoder_code_dir
                )
            except Exception as e2:
                raise RuntimeError(
                    "RAE decoder support requires torch + transformers and decoder source files. "
                    "Failed imports: package import error="
                    f"{type(import_err).__name__}: {import_err}; "
                    "direct-file import error="
                    f"{type(e2).__name__}: {e2}. "
                    "Set `rae_decoder.code_dir` to a folder containing decoder.py and utils.py "
                    "(e.g. <RAE>/src/stage1/decoders)."
                ) from e2

        self._torch = torch
        self._GeneralDecoder = GeneralDecoder
        self._ViTMAEConfig = ViTMAEConfig

        self.decoder_config_path = _resolve_path(decoder_config_path)
        self.pretrained_decoder_path = _resolve_path(pretrained_decoder_path)

        # Auto-fetch the decoder weights from HuggingFace if they are not present
        # locally (so eval/inference works without a separate download step).
        self.pretrained_decoder_path = _maybe_hf_download(
            self.pretrained_decoder_path, hf_repo_id, auto_download
        )
        self.latent_dim = int(latent_dim)
        self.latent_hw = int(latent_hw)
        self.decoder_patch_size = int(decoder_patch_size)
        self.decoder_code_dir = _resolve_path(decoder_code_dir) if decoder_code_dir else None

        if not self.decoder_config_path or not os.path.exists(self.decoder_config_path):
            raise FileNotFoundError(
                f"RAE decoder config not found: {self.decoder_config_path!r}"
            )
        if not self.pretrained_decoder_path or not os.path.exists(self.pretrained_decoder_path):
            raise FileNotFoundError(
                f"RAE decoder checkpoint not found: {self.pretrained_decoder_path!r}. "
                "Run `bash scripts/download_models.sh`, or set a valid "
                "rae_decoder.pretrained_decoder_path (auto-download only applies to "
                "paths under checkpoints/rae/)."
            )

        self.device = torch.device(
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)

        self.decoder = self._build_decoder().to(self.device).eval()
        self.image_mean = self.image_mean.to(self.device)
        self.image_std = self.image_std.to(self.device)

    def _build_decoder(self):
        torch = self._torch

        if self.latent_hw <= 0:
            raise ValueError(f"latent_hw must be > 0, got {self.latent_hw}")
        num_patches = self.latent_hw * self.latent_hw

        cfg = self._ViTMAEConfig.from_pretrained(self.decoder_config_path)
        cfg.hidden_size = int(self.latent_dim)
        cfg.patch_size = int(self.decoder_patch_size)
        cfg.image_size = int(self.decoder_patch_size * self.latent_hw)

        decoder = self._GeneralDecoder(cfg, num_patches=num_patches)
        state = _extract_state_dict(_torch_load_compat(torch, self.pretrained_decoder_path))
        keys = decoder.load_state_dict(state, strict=False)
        if getattr(keys, "missing_keys", None):
            warnings.warn(
                f"Missing decoder keys when loading {self.pretrained_decoder_path}: "
                f"{keys.missing_keys[:8]}{'...' if len(keys.missing_keys) > 8 else ''}"
            )
        if getattr(keys, "unexpected_keys", None):
            warnings.warn(
                f"Unexpected decoder keys when loading {self.pretrained_decoder_path}: "
                f"{keys.unexpected_keys[:8]}{'...' if len(keys.unexpected_keys) > 8 else ''}"
            )
        return decoder

    @property
    def image_size(self) -> int:
        return int(self.decoder_patch_size * self.latent_hw)

    def _validate_latent(self, z):
        if z.ndim != 4:
            raise ValueError(f"Expected latent BCHW tensor, got shape {tuple(z.shape)}")
        b, c, h, w = z.shape
        if c != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {c} (shape={tuple(z.shape)})"
            )
        if h != self.latent_hw or w != self.latent_hw:
            raise ValueError(
                f"Expected latent spatial size {self.latent_hw}x{self.latent_hw}, "
                f"got {h}x{w}"
            )

    def decode_tensor(self, z):
        torch = self._torch
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, dtype=torch.float32)
        else:
            z = z.to(dtype=torch.float32)
        self._validate_latent(z)
        z = z.to(self.device, non_blocking=True)

        with torch.no_grad():
            b, c, h, w = z.shape
            z_seq = z.view(b, c, h * w).transpose(1, 2).contiguous()
            out = self.decoder(z_seq, drop_cls_token=False).logits
            x = self.decoder.unpatchify(out)
            x = x * self.image_std + self.image_mean
        return x

    def decode_uint8(self, z) -> np.ndarray:
        torch = self._torch
        x = self.decode_tensor(z)
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.detach().cpu().numpy()


def _cache_key(config, latent_hw: int, latent_dim: int) -> Tuple[Any, ...]:
    dec_cfg = getattr(config, "rae_decoder", None)
    return (
        _resolve_path(_cfg_get(dec_cfg, "code_dir", "")),
        _resolve_path(_cfg_get(dec_cfg, "decoder_config_path", "")),
        _resolve_path(_cfg_get(dec_cfg, "pretrained_decoder_path", "")),
        int(_cfg_get(dec_cfg, "decoder_patch_size", 16)),
        int(_cfg_get(dec_cfg, "latent_dim", latent_dim)),
        int(latent_hw),
        str(_cfg_get(dec_cfg, "device", "")),
        tuple(_cfg_get(dec_cfg, "image_mean", (0.485, 0.456, 0.406))),
        tuple(_cfg_get(dec_cfg, "image_std", (0.229, 0.224, 0.225))),
    )


def get_decoder(config, latent_hw: int, latent_dim: int) -> Optional[RAEDecoderOnly]:
    if not _is_enabled(config):
        return None

    key = _cache_key(config, latent_hw=latent_hw, latent_dim=latent_dim)
    dec = _DECODER_CACHE.get(key, None)
    if dec is not None:
        return dec

    dec_cfg = getattr(config, "rae_decoder", None)
    dec = RAEDecoderOnly(
        decoder_code_dir=_cfg_get(dec_cfg, "code_dir", "third_party/rae_decoder"),
        decoder_config_path=_cfg_get(dec_cfg, "decoder_config_path", "third_party/rae_decoder/configs/ViTXL"),
        pretrained_decoder_path=_cfg_get(dec_cfg, "pretrained_decoder_path", ""),
        decoder_patch_size=int(_cfg_get(dec_cfg, "decoder_patch_size", 16)),
        latent_dim=int(_cfg_get(dec_cfg, "latent_dim", latent_dim)),
        latent_hw=int(_cfg_get(dec_cfg, "latent_hw", latent_hw)),
        image_mean=tuple(_cfg_get(dec_cfg, "image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(_cfg_get(dec_cfg, "image_std", (0.229, 0.224, 0.225))),
        device=_cfg_get(dec_cfg, "device", None),
        auto_download=bool(_cfg_get(dec_cfg, "auto_download", True)),
        hf_repo_id=_cfg_get(dec_cfg, "hf_repo_id", _RAE_HF_REPO),
    )
    _DECODER_CACHE[key] = dec
    return dec


def decode_bchw_to_uint8(latents_bchw, config, batch_size: int = 32) -> np.ndarray:
    latents = np.asarray(latents_bchw, dtype=np.float32)
    if latents.ndim != 4:
        raise ValueError(f"Expected latents in BCHW format, got {latents.shape}")
    b, c, h, w = latents.shape
    dec = get_decoder(config, latent_hw=h, latent_dim=c)
    if dec is None:
        raise RuntimeError(
            "Latent outputs detected but `config.rae_decoder.enabled` is False. "
            "Enable the RAE decoder and set checkpoint/config paths."
        )

    outs = []
    bs = max(1, int(batch_size))
    for i in range(0, b, bs):
        outs.append(dec.decode_uint8(latents[i : i + bs]))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, dec.image_size, dec.image_size, 3), dtype=np.uint8)
