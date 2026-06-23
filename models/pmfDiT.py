import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.embedder import (
    TimestepEmbedder,
    LabelEmbedder,
    BottleneckPatchEmbedder,
    TokenEmbed,
)
from models.torch_models import TorchLinear, RMSNorm, SwiGLUMlp


def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return jnp.expand_dims(t, axis=dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    hidden_size: int
    num_heads: int

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0
    dtype: Any = None

    def setup(self):
        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def __call__(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        attn = nn.dot_product_attention(q, k, v, dtype=jnp.float32)
        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 8 / 3

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0
    dtype: Any = None

    def setup(self):
        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )

        self.attn_scale = self.param(
            "attn_scale", nn.initializers.zeros, (self.hidden_size,)
        )
        self.mlp_scale = self.param(
            "mlp_scale", nn.initializers.zeros, (self.hidden_size,)
        )

    def __call__(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    hidden_size: int
    out_channels: int
    patch_size: int = 1
    token_mode: bool = False
    dtype: Any = None

    def setup(self):
        out_dim = (
            self.out_channels
            if self.token_mode
            else self.patch_size * self.patch_size * self.out_channels
        )
        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            out_dim,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
            dtype=self.dtype,
        )

    def __call__(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class pmfDiT(nn.Module):
    """
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    input_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    depth: int = 16
    num_heads: int = 12
    mlp_ratio: float = 8 / 3
    num_classes: int = 1000

    aux_head_depth: int = 8
    head_wide_layers: int = 0       # number of final head layers at wider size
    head_wide_size: int = 2048      # hidden size for the wide layers
    head_wide_num_heads: int = 16   # num attention heads for wide layers (must divide head_wide_size)

    # v-side head overrides. Set to -1 (default) to follow the u-side.
    # Enables asymmetric heads, e.g. u = wide 2x2048 while v = narrow 2x768.
    # shared_depth is computed as depth - max(u_total, v_total).
    v_aux_head_depth: int = -1
    v_head_wide_layers: int = -1
    v_head_wide_size: int = -1
    v_head_wide_num_heads: int = -1

    # Legacy wide-head forward path (for evaluating older checkpoints trained
    # before the DDT-style time re-injection / RoPE fix). When True:
    #   - No `wide_time_proj_u` / `wide_time_proj_v` module is created.
    #   - Forward computes `u_ctx = u_ctx_proj(u_seq)` with NO silu and NO
    #     time re-injection (matches pre-refactor behavior).
    #   - Wide RoPE is doubled via `concat([rope, rope], axis=0)` so the
    #     last 2*image_tokens positions (ctx_patches + data) all get rotated,
    #     matching the pre-refactor collision behavior.
    # Keep False for new training; set True only to load old checkpoints.
    legacy_wide_head: bool = False

    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    num_level_tokens: int = 2

    token_init_constant: float = 1.0
    embedding_init_constant: float = 1.0
    weight_init_constant: float = 0.32

    num_levels: int = 4
    use_level_cond: bool = False
    use_prev_cond: bool = False
    use_token_embed: bool = False
    cond_token_mode: str = "full_concat"  # full_concat | fused
    use_cfg: bool = True
    gradient_checkpointing: bool = False
    dtype: Any = None  # computation dtype (None = float32)

    v_loss_weight: float = 1.0
    eval: bool = False

    def setup(self):
        """
        Set up the pmfDiT model components.
         - Patch embedder for input images.
         - Embedders for time, omega, cfg intervals, and class labels.
         - Learnable tokens for conditioning.
         - Transformer blocks with shared backbone and dual heads.
         - Final projection layers for u and v outputs.
        """

        self.out_channels = self.in_channels
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )

        if self.use_token_embed:
            self.x_embedder = TokenEmbed(
                img_size=self.input_size,
                in_chans=self.in_channels,
                embed_dim=self.hidden_size,
                bias=True,
                dtype=self.dtype,
            )
            if self.use_prev_cond:
                self.prev_embedder = TokenEmbed(
                    img_size=self.input_size,
                    in_chans=self.in_channels,
                    embed_dim=self.hidden_size,
                    bias=True,
                    dtype=self.dtype,
                )
        else:
            self.x_embedder = BottleneckPatchEmbedder(
                self.input_size,
                self.patch_size,
                128 if self.hidden_size <= 1024 else 256,  # pca channels. 256 for H/G
                self.in_channels,
                self.hidden_size,
                bias=True,
                dtype=self.dtype,
            )
            if self.use_prev_cond:
                self.prev_embedder = BottleneckPatchEmbedder(
                    self.input_size,
                    self.patch_size,
                    128 if self.hidden_size <= 1024 else 256,
                    self.in_channels,
                    self.hidden_size,
                    bias=True,
                    dtype=self.dtype,
                )
        if self.use_prev_cond:
            if self.cond_token_mode not in {"full_concat", "fused"}:
                raise ValueError(
                    f"Invalid cond_token_mode={self.cond_token_mode}. "
                    "Expected one of: full_concat, fused"
                )
            if self.cond_token_mode == "fused":
                self.cond_fuse_proj = TorchLinear(
                    in_features=2 * self.hidden_size,
                    out_features=self.hidden_size,
                    bias=True,
                    weight_init="scaled_variance",
                    init_constant=self.embedding_init_constant,
                    bias_init="zeros",
                    dtype=self.dtype,
                )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
            dtype=self.dtype,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        if self.use_cfg:
            self.omega_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
            self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(self.num_classes, **embed_kwargs)
        if self.use_level_cond:
            self.level_embedder = LabelEmbedder(self.num_levels, **embed_kwargs)

        token_initializer = nn.initializers.normal(
            stddev=self.token_init_constant / math.sqrt(self.hidden_size)
        )

        self.time_tokens = self.param(
            "time_tokens",
            token_initializer,
            (1, self.num_time_tokens, self.hidden_size),
        )
        self.class_tokens = self.param(
            "class_tokens",
            token_initializer,
            (1, self.num_class_tokens, self.hidden_size),
        )
        if self.use_cfg:
            self.omega_tokens = self.param(
                "omega_tokens",
                token_initializer,
                (1, self.num_cfg_tokens, self.hidden_size),
            )
            self.t_min_tokens = self.param(
                "t_min_tokens",
                token_initializer,
                (1, self.num_interval_tokens, self.hidden_size),
            )
            self.t_max_tokens = self.param(
                "t_max_tokens",
                token_initializer,
                (1, self.num_interval_tokens, self.hidden_size),
            )
        if self.use_level_cond:
            self.level_tokens = self.param(
                "level_tokens",
                token_initializer,
                (1, self.num_level_tokens, self.hidden_size),
            )

        self.image_tokens = self.x_embedder.num_patches
        self.prev_tokens = (
            self.image_tokens
            if (self.use_prev_cond and self.cond_token_mode == "full_concat")
            else 0
        )
        cfg_token_count = (
            self.num_cfg_tokens + 2 * self.num_interval_tokens
        ) if self.use_cfg else 0
        self.cond_tokens = (
            self.num_class_tokens
            + cfg_token_count
            + self.num_time_tokens
            + (self.num_level_tokens if self.use_level_cond else 0)
        )
        self.prefix_tokens = self.cond_tokens + self.prev_tokens
        total_tokens = self.prefix_tokens + self.image_tokens
        self.head_dim = self.hidden_size // self.num_heads

        patch_rope = precompute_rope_freqs_2d(self.head_dim, self.image_tokens)
        self.rope_freqs = (
            jnp.concatenate([patch_rope, patch_rope], axis=0)
            if (self.use_prev_cond and self.cond_token_mode == "full_concat")
            else patch_rope
        )
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, total_tokens, self.hidden_size),
        )

        # Resolve per-side head shapes. v-side fields with value -1 fall back to u-side.
        u_total_head    = self.aux_head_depth
        u_wide_layers   = self.head_wide_layers
        u_wide_size     = self.head_wide_size
        u_wide_nh       = self.head_wide_num_heads

        v_total_head    = self.v_aux_head_depth      if self.v_aux_head_depth      >= 0 else u_total_head
        v_wide_layers   = self.v_head_wide_layers    if self.v_head_wide_layers    >= 0 else u_wide_layers
        v_wide_size     = self.v_head_wide_size      if self.v_head_wide_size      >= 0 else u_wide_size
        v_wide_nh       = self.v_head_wide_num_heads if self.v_head_wide_num_heads >= 0 else u_wide_nh

        u_narrow_layers = u_total_head - u_wide_layers
        v_narrow_layers = v_total_head - v_wide_layers
        assert u_narrow_layers >= 0, (
            f"head_wide_layers ({u_wide_layers}) > aux_head_depth ({u_total_head})"
        )
        assert v_narrow_layers >= 0, (
            f"v_head_wide_layers ({v_wide_layers}) > v_aux_head_depth ({v_total_head})"
        )
        assert u_wide_size % u_wide_nh == 0, (
            f"head_wide_size ({u_wide_size}) must be divisible by head_wide_num_heads ({u_wide_nh})"
        )
        assert v_wide_size % v_wide_nh == 0, (
            f"v_head_wide_size ({v_wide_size}) must be divisible by "
            f"v_head_wide_num_heads ({v_wide_nh})"
        )

        shared_depth = self.depth - max(u_total_head, v_total_head)
        assert shared_depth >= 0, (
            f"depth ({self.depth}) < max head depth ({max(u_total_head, v_total_head)})"
        )

        # Save derived counts for __call__ (int attributes, not params).
        self.u_wide_layers_count = u_wide_layers
        self.v_wide_layers_count = v_wide_layers

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )

        block_cls = (
            nn.remat(TransformerBlock)
            if self.gradient_checkpointing and not self.eval
            else TransformerBlock
        )

        self.shared_blocks = [
            block_cls(**block_kwargs) for _ in range(shared_depth)
        ]

        u_wide_block_kwargs = dict(
            hidden_size=u_wide_size,
            num_heads=u_wide_nh,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )
        v_wide_block_kwargs = dict(
            hidden_size=v_wide_size,
            num_heads=v_wide_nh,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )

        # RoPE tables for wide heads (may differ per side if sizes differ).
        if u_wide_layers > 0:
            u_wide_head_dim = u_wide_size // u_wide_nh
            u_wide_patch_rope = precompute_rope_freqs_2d(u_wide_head_dim, self.image_tokens)
            self.rope_freqs_u_wide = (
                jnp.concatenate([u_wide_patch_rope, u_wide_patch_rope], axis=0)
                if (self.use_prev_cond and self.cond_token_mode == "full_concat")
                else u_wide_patch_rope
            )
        if v_wide_layers > 0:
            v_wide_head_dim = v_wide_size // v_wide_nh
            v_wide_patch_rope = precompute_rope_freqs_2d(v_wide_head_dim, self.image_tokens)
            self.rope_freqs_v_wide = (
                jnp.concatenate([v_wide_patch_rope, v_wide_patch_rope], axis=0)
                if (self.use_prev_cond and self.cond_token_mode == "full_concat")
                else v_wide_patch_rope
            )

        self.u_heads      = [block_cls(**block_kwargs)        for _ in range(u_narrow_layers)]
        self.u_heads_wide = [block_cls(**u_wide_block_kwargs) for _ in range(u_wide_layers)]

        _need_v_heads = (not self.eval) and (self.v_loss_weight > 0)
        self.v_heads = [
            block_cls(**block_kwargs)
            for _ in range(v_narrow_layers if _need_v_heads else 0)
        ]
        self.v_heads_wide = [
            block_cls(**v_wide_block_kwargs)
            for _ in range(v_wide_layers if _need_v_heads else 0)
        ]

        # DDT-style wide head components: u-side.
        if u_wide_layers > 0:
            # Time re-injection into context, mirroring RAE DDT's `s = silu(t + s)`.
            # Skipped in legacy mode so old checkpoints don't have a mismatched key.
            if not self.legacy_wide_head:
                self.wide_time_proj_u = TorchLinear(
                    self.hidden_size, u_wide_size, bias=True,
                    weight_init="scaled_variance",
                    init_constant=self.weight_init_constant,
                    bias_init="zeros",
                    dtype=self.dtype,
                )
            self.u_ctx_proj = TorchLinear(
                self.hidden_size, u_wide_size, bias=False,
                weight_init="scaled_variance",
                init_constant=self.weight_init_constant,
                dtype=self.dtype,
            )
            if self.use_token_embed:
                self.u_wide_x_embed = TokenEmbed(
                    img_size=self.input_size,
                    in_chans=self.in_channels,
                    embed_dim=u_wide_size,
                    bias=True,
                    dtype=self.dtype,
                )
            else:
                self.u_wide_x_embed = BottleneckPatchEmbedder(
                    self.input_size, self.patch_size,
                    128 if u_wide_size <= 1024 else 256,
                    self.in_channels, u_wide_size,
                    bias=True, dtype=self.dtype,
                )
            if self.use_prev_cond and self.cond_token_mode == "fused":
                if self.use_token_embed:
                    self.u_wide_prev_embed = TokenEmbed(
                        img_size=self.input_size,
                        in_chans=self.in_channels,
                        embed_dim=u_wide_size,
                        bias=True,
                        dtype=self.dtype,
                    )
                else:
                    self.u_wide_prev_embed = BottleneckPatchEmbedder(
                        self.input_size, self.patch_size,
                        128 if u_wide_size <= 1024 else 256,
                        self.in_channels, u_wide_size,
                        bias=True, dtype=self.dtype,
                    )
                self.u_wide_cond_fuse = TorchLinear(
                    in_features=2 * u_wide_size,
                    out_features=u_wide_size,
                    bias=True,
                    weight_init="scaled_variance",
                    init_constant=self.embedding_init_constant,
                    bias_init="zeros",
                    dtype=self.dtype,
                )

        # DDT-style wide head components: v-side.
        if _need_v_heads and v_wide_layers > 0:
            if not self.legacy_wide_head:
                self.wide_time_proj_v = TorchLinear(
                    self.hidden_size, v_wide_size, bias=True,
                    weight_init="scaled_variance",
                    init_constant=self.weight_init_constant,
                    bias_init="zeros",
                    dtype=self.dtype,
                )
            self.v_ctx_proj = TorchLinear(
                self.hidden_size, v_wide_size, bias=False,
                weight_init="scaled_variance",
                init_constant=self.weight_init_constant,
                dtype=self.dtype,
            )
            if self.use_token_embed:
                self.v_wide_x_embed = TokenEmbed(
                    img_size=self.input_size,
                    in_chans=self.in_channels,
                    embed_dim=v_wide_size,
                    bias=True,
                    dtype=self.dtype,
                )
            else:
                self.v_wide_x_embed = BottleneckPatchEmbedder(
                    self.input_size, self.patch_size,
                    128 if v_wide_size <= 1024 else 256,
                    self.in_channels, v_wide_size,
                    bias=True, dtype=self.dtype,
                )
            if self.use_prev_cond and self.cond_token_mode == "fused":
                if self.use_token_embed:
                    self.v_wide_prev_embed = TokenEmbed(
                        img_size=self.input_size,
                        in_chans=self.in_channels,
                        embed_dim=v_wide_size,
                        bias=True,
                        dtype=self.dtype,
                    )
                else:
                    self.v_wide_prev_embed = BottleneckPatchEmbedder(
                        self.input_size, self.patch_size,
                        128 if v_wide_size <= 1024 else 256,
                        self.in_channels, v_wide_size,
                        bias=True, dtype=self.dtype,
                    )
                self.v_wide_cond_fuse = TorchLinear(
                    in_features=2 * v_wide_size,
                    out_features=v_wide_size,
                    bias=True,
                    weight_init="scaled_variance",
                    init_constant=self.embedding_init_constant,
                    bias_init="zeros",
                    dtype=self.dtype,
                )

        # Per-side final layer hidden sizes.
        u_final_hidden = u_wide_size if u_wide_layers > 0 else self.hidden_size
        v_final_hidden = v_wide_size if v_wide_layers > 0 else self.hidden_size

        self.u_final_layer = FinalLayer(
            u_final_hidden,
            self.out_channels,
            patch_size=self.patch_size,
            token_mode=self.use_token_embed,
            dtype=self.dtype,
        )
        if _need_v_heads:
            self.v_final_layer = FinalLayer(
                v_final_hidden,
                self.out_channels,
                patch_size=self.patch_size,
                token_mode=self.use_token_embed,
                dtype=self.dtype,
            )
        else:
            v_out_dim = (
                self.out_channels
                if self.use_token_embed
                else self.patch_size * self.patch_size * self.out_channels
            )
            self.v_final_layer = lambda x: jnp.zeros(
                (x.shape[0], x.shape[1], v_out_dim),
                dtype=x.dtype,
            )

    def unpatchify(self, x, h=None, w=None):
        if self.use_token_embed:
            if h is None or w is None:
                hw = x.shape[1]
                h = int(hw**0.5)
                w = h
            return x.reshape((x.shape[0], h, w, self.out_channels))

        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        images = x.reshape((x.shape[0], h * p, w * p, c))
        return images

    def _build_sequence(self, x, t, h, w, t_min, t_max, y, level=None, cond=None):
        """
        Build the input token sequence for the transformer.
        1. Embed the input image patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input images
            h: timestep
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels
        
        Returns:
            seq: Token sequence for the transformer
        """

        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + unsqueeze(h_embed, 1)
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1)

        if self.use_cfg:
            omega_embed = self.omega_embedder(1 - 1 / w)
            t_min_embed = self.cfg_t_start_embedder(t_min)
            t_max_embed = self.cfg_t_end_embedder(t_max)
            omega_tokens = self.omega_tokens + unsqueeze(omega_embed, 1)
            t_min_tokens = self.t_min_tokens + unsqueeze(t_min_embed, 1)
            t_max_tokens = self.t_max_tokens + unsqueeze(t_max_embed, 1)
        if self.use_level_cond:
            if level is None:
                level = jnp.zeros((x.shape[0],), dtype=jnp.int32)
            level_embed = self.level_embedder(level.astype(jnp.int32))
            level_tokens = self.level_tokens + unsqueeze(level_embed, 1)
        else:
            level_tokens = None

        cond_tokens = None
        if self.use_prev_cond:
            if cond is None:
                cond = jnp.zeros_like(x)
            cond_tokens = self.prev_embedder(cond)
            if self.cond_token_mode == "fused":
                x_embed = self.cond_fuse_proj(jnp.concatenate([x_embed, cond_tokens], axis=-1))
                cond_tokens = None

        seq_chunks = [class_tokens]
        if self.use_cfg:
            seq_chunks.extend([omega_tokens, t_min_tokens, t_max_tokens])
        seq_chunks.append(time_tokens)
        if level_tokens is not None:
            seq_chunks.append(level_tokens)
        if cond_tokens is not None:
            seq_chunks.append(cond_tokens)
        seq_chunks.append(x_embed)
        seq = jnp.concatenate(seq_chunks, axis=1)

        seq = seq + self.pos_embed

        # Cast to computation dtype (e.g., bfloat16 for mixed precision)
        if self.dtype is not None:
            seq = seq.astype(self.dtype)

        return seq

    def __call__(self, x, t, h, w, t_min, t_max, y, level=None, cond=None):
        """
        Forward pass of the pmfDiT model.
        Returns the predicted u and v components.

        Args:
            x: Input images
            t, h: time steps
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            u: Average velocity field
            v: Instantaneous velocity field
        """

        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        seq = self._build_sequence(x, t, h, w, t_min, t_max, y, level=level, cond=cond)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)

        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)

        # Precompute the time embedding once for potential re-injection into
        # wide contexts. This reuses the shared h_embedder params.
        # Not needed in legacy mode (no time re-injection).
        h_embed_raw = None
        if not self.legacy_wide_head and (
            self.u_wide_layers_count > 0
            or (self.v_wide_layers_count > 0 and self.v_heads_wide)
        ):
            h_embed_raw = self.h_embedder(h)  # (B, hidden_size)

        # u-side: DDT-style wide head if configured, else narrow tokens.
        if self.u_wide_layers_count > 0:
            u_ctx = self.u_ctx_proj(u_seq)                        # (B, prefix+patches, u_wide_size)
            if not self.legacy_wide_head:
                # DDT-style: inject time into context and apply silu.
                h_wide_u = self.wide_time_proj_u(h_embed_raw)     # (B, u_wide_size)
                h_wide_u = jnp.expand_dims(h_wide_u, axis=1)      # (B, 1, u_wide_size)
                if self.dtype is not None:
                    h_wide_u = h_wide_u.astype(self.dtype)
                u_ctx = nn.silu(u_ctx + h_wide_u)
            # else: legacy path leaves u_ctx as the raw linear projection.
            u_data = self.u_wide_x_embed(x)                       # (B, patches, u_wide_size)
            if self.use_prev_cond and self.cond_token_mode == "fused" and cond is not None:
                u_cond_data = self.u_wide_prev_embed(cond)
                u_data = self.u_wide_cond_fuse(jnp.concatenate([u_data, u_cond_data], axis=-1))
            u_seq = jnp.concatenate([u_ctx, u_data], axis=1)
            if self.legacy_wide_head:
                # Legacy: rotate last 2*image_tokens (ctx_patches + data), the
                # pre-refactor collision behavior that old checkpoints expect.
                u_wide_rope = jnp.concatenate(
                    [self.rope_freqs_u_wide, self.rope_freqs_u_wide], axis=0
                )
            else:
                # New: rotate only the last image_tokens positions (= u_data).
                # Context acts as positionless conditioning.
                u_wide_rope = self.rope_freqs_u_wide              # (image_tokens, u_head_dim)
            for block in self.u_heads_wide:
                u_seq = block(u_seq, u_wide_rope)
            u_tokens = u_seq[:, -self.image_tokens:]
        else:
            u_tokens = u_seq[:, self.prefix_tokens:]

        # v-side: independent. DDT-style wide head if configured and needed, else narrow.
        if self.v_wide_layers_count > 0 and self.v_heads_wide:
            v_ctx = self.v_ctx_proj(v_seq)
            if not self.legacy_wide_head:
                h_wide_v = self.wide_time_proj_v(h_embed_raw)     # (B, v_wide_size)
                h_wide_v = jnp.expand_dims(h_wide_v, axis=1)      # (B, 1, v_wide_size)
                if self.dtype is not None:
                    h_wide_v = h_wide_v.astype(self.dtype)
                v_ctx = nn.silu(v_ctx + h_wide_v)
            v_data = self.v_wide_x_embed(x)
            if self.use_prev_cond and self.cond_token_mode == "fused" and cond is not None:
                v_cond_data = self.v_wide_prev_embed(cond)
                v_data = self.v_wide_cond_fuse(jnp.concatenate([v_data, v_cond_data], axis=-1))
            v_seq = jnp.concatenate([v_ctx, v_data], axis=1)
            if self.legacy_wide_head:
                v_wide_rope = jnp.concatenate(
                    [self.rope_freqs_v_wide, self.rope_freqs_v_wide], axis=0
                )
            else:
                v_wide_rope = self.rope_freqs_v_wide
            for block in self.v_heads_wide:
                v_seq = block(v_seq, v_wide_rope)
            v_tokens = v_seq[:, -self.image_tokens:]
        else:
            v_tokens = v_seq[:, self.prefix_tokens:]

        u_pred = self.unpatchify(self.u_final_layer(u_tokens), x.shape[1], x.shape[2])
        v_pred = self.unpatchify(self.v_final_layer(v_tokens), x.shape[1], x.shape[2])
        x_u_pred = u_pred

        t = t.reshape((-1, 1, 1, 1))

        u = (x - u_pred) / jnp.clip(t, 0.05, 1.0)
        v = (x - v_pred) / jnp.clip(t, 0.05, 1.0)

        return u, v, x_u_pred


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs_2d(dim: int, seq_len: int, theta: float = 10000.0):
    dim = dim // 2  # for 2d rotary embeddings
    T = int(seq_len ** 0.5)
    if T * T != seq_len:
        raise ValueError(f"seq_len must be a square number for 2D RoPE, got {seq_len}")
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(T, dtype=jnp.float32)
    freqs_h = jnp.einsum('i,j->ij', positions, freqs)
    freqs_w = jnp.einsum('i,j->ij', positions, freqs)
    freqs = jnp.concatenate([jnp.tile(freqs_h[:, None, :], (1, T, 1)), jnp.tile(freqs_w[None, :, :], (T, 1, 1))], axis=-1)  # (T, T, 2D)
    real = jnp.cos(freqs).reshape(seq_len, dim)
    imag = jnp.sin(freqs).reshape(seq_len, dim)
    return jax.lax.complex(real, imag)


def apply_rotary_pos_emb(x, freqs_cis):
    orig_dtype = x.dtype
    x_complex = x.astype(jnp.float32).view(jnp.complex64)
    x_complex = x_complex.reshape(x.shape[:-1] + (-1,))
    freqs_cis = unsqueeze(unsqueeze(freqs_cis, 0), 2)
    T = freqs_cis.shape[1]
    x_rotated = x_complex.at[:, -T:, :].multiply(freqs_cis)
    x_out = x_rotated.astype(x_complex.dtype).view(jnp.float32)
    return x_out.reshape(x.shape).astype(orig_dtype)


#################################################################################
#                                   pMF Configs                                 #
#################################################################################


pmfDiT_B_16 = partial(
    pmfDiT,
    input_size=256,
    depth=16,
    hidden_size=768,
    patch_size=16,
    num_heads=12,
    aux_head_depth=8,
)


pmfDiT_B_32 = partial(
    pmfDiT,
    input_size=512,
    depth=16,
    hidden_size=768,
    patch_size=32,
    num_heads=12,
    aux_head_depth=8,
)

pmfDiT_L_16 = partial(
    pmfDiT,
    input_size=256,
    depth=32,
    hidden_size=1024,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_L_32 = partial(
    pmfDiT,
    input_size=512,
    depth=32,
    hidden_size=1024,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_16 = partial(
    pmfDiT,
    input_size=256,
    depth=48,
    hidden_size=1280,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_32 = partial(
    pmfDiT,
    input_size=512,
    depth=48,
    hidden_size=1280,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H = partial(
    pmfDiT,
    input_size=256,
    depth=36,
    hidden_size=1280,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)
