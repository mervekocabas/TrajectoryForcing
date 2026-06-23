from torch import nn
import torch
from transformers import AutoModel
from . import register_encoder

@register_encoder()
class Dinov3wNorm(nn.Module):
    """
    DINOv3 visual encoder wrapper that:
      - loads the vision tower from a HF checkpoint (handles vision-only or vision+text containers)
      - disables any projection head (we don't need alignment heads for your use case)
      - optionally removes affine params of the final LayerNorm (normalize=True)
      - returns ONLY patch tokens (drops CLS + register tokens if present)
    """
    def __init__(self, model_name: str, normalize: bool = True, drop_cls: bool = True):
        super().__init__()
        self.model_name = model_name
        self.drop_cls = drop_cls

        # The visual encoder in dinov3 includes a projection head we dont need it 
        base = AutoModel.from_pretrained(self.model_name)
        self.encoder = getattr(base, "vision_model", base)
        self.encoder.requires_grad_(False)

        def strip_proj(mod: nn.Module):
            for attr in ["image_projection", "visual_projection", "projection", "proj",
                         "projector", "mlp_head", "head"]:
                if hasattr(mod, attr) and isinstance(getattr(mod, attr), nn.Module):
                    setattr(mod, attr, nn.Identity())

        strip_proj(base)
        strip_proj(self.encoder)

        if normalize:
            for name in ["post_layernorm", "layernorm", "ln_post", "norm",
                         "final_layernorm", "last_layernorm"]:
                ln = getattr(self.encoder, name, None)
                if isinstance(ln, nn.LayerNorm):
                    ln.elementwise_affine = False
                    ln.weight = None
                    ln.bias = None
                    break

        cfg = getattr(self.encoder, "config", None)
        self.patch_size  = getattr(cfg, "patch_size", 16)  # rae encoders patch size 14
        self.hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
            or 1024
        )
        self.num_register_tokens = getattr(cfg, "num_register_tokens", 0)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W] in [0,1] (your outer wrapper applies mean/std normalization)
        returns: [B, N_patches, C]  (CLS + registers removed)
        """
        try:
            out = self.encoder(images, output_hidden_states=True, interpolate_pos_encoding=True)
        except TypeError:
            out = self.encoder(images, output_hidden_states=True)

        tokens = out.last_hidden_state  # [B, (CLS)+(registers)+patches, C]

        drop = (1 if self.drop_cls else 0) + int(self.num_register_tokens)
        if drop > 0:
            tokens = tokens[:, drop:, :]

        return tokens
