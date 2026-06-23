import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Optional, Protocol
from transformers import AutoImageProcessor, AutoConfig
from models import ARCHS  
from sklearn.decomposition import PCA
from torchvision.utils import save_image

class Stage1Protocol(Protocol):
    patch_size: int
    hidden_size: int
    def __call__(self, x: torch.Tensor) -> torch.Tensor: 
        ...

class EncoderOnly(nn.Module):
    def __init__(
        self,
        encoder_cls: str = "Dinov3withNorm",
        encoder_config_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        encoder_input_size: int = 256,
        encoder_params: dict = {},
        reshape_to_2d: bool = True,         # True -> [B,C,Hp,Wp]; False -> [B,N,C]
        normalization_stat_path: Optional[str] = None,  # (optional) latent z-norm
        eps: float = 1e-5,
    ):
        super().__init__()
        # ---- encoder ----
        enc_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocol = enc_cls(**encoder_params)
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size

        assert encoder_input_size % self.encoder_patch_size == 0, \
            f"encoder_input_size {encoder_input_size} must be divisible by patch_size {self.encoder_patch_size}"

        # ---- input normalization (mean/std from HF processor) ----
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std  = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        # ---- output shaping ----
        self.reshape_to_2d = reshape_to_2d

        # ---- optional latent normalization (for stable caching across runs) ----
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var  = stats.get("var", None)
            self.do_normalization = True
            self.eps = eps
        else:
            self.do_normalization = False

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] in [0,1]
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic", align_corners=False
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

        tokens = self.encoder(x)  # [B, N, C] (CLS/regs already removed in DINO wrapper)
        if self.reshape_to_2d:
            b, n, c = tokens.shape
            hp = wp = int(sqrt(n))
            assert hp * wp == n, f"Token count {n} is not a square, got hp*wp={hp*wp}"
            z = tokens.transpose(1, 2).reshape(b, c, hp, wp)  # [B,C,Hp,Wp]
        else:
            z = tokens  # [B,N,C]

        if self.do_normalization:
            mean = (self.latent_mean.to(z.device) if self.latent_mean is not None else 0)
            var  = (self.latent_var.to(z.device)  if self.latent_var  is not None else 1)
            z = (z - mean) / torch.sqrt(var + self.eps)

        return z

    # alias for forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ---------- convenience helpers ----------

@torch.no_grad()
def save_latents_pt(z: torch.Tensor, path: str, meta: dict):
    torch.save({"z": z.cpu(), "meta": meta}, path)

@torch.no_grad()
def to_seq(z_bchw: torch.Tensor) -> torch.Tensor:
    # [B,C,Hp,Wp] -> [B,N,C]
    B, C, Hp, Wp = z_bchw.shape
    return z_bchw.view(B, C, Hp*Wp).transpose(1, 2).contiguous()
