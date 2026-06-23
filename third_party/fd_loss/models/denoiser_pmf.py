import logging
import math
import re
from functools import partial

import torch
import torch.nn as nn
from tqdm import trange

from .mit import MiT_models

logger = logging.getLogger("FD_loss")


class pMFDenoiser(nn.Module):
    """pixel meanflow denoiser with cfg-aware training and perceptual loss."""

    def __init__(
        self,
        backbone="MiT_B",
        img_size=256,
        patch_size=16,
        in_channels=3,
        tokenizer_patch_size=1,
        bottleneck_dim=128,
        num_classes=1000,
        label_drop_prob=0.1,
        P_mean=0.8,
        P_std=0.8,
        ratio_r_neq_t=0.5,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        aux_head_depth=8,
        class_tokens=8,
        time_tokens=4,
        guidance_tokens=4,
        interval_tokens=2,
        token_init_constant=1.0,
        embedding_init_constant=1.0,
        weight_init_constant=0.32,
        tr_uniform=False,
        norm_eps=1e-4,
        norm_p=1.0,
        t_eps=0.05,
        noise_scale=None,
        perceptual_threshold=0.8,
        perceptual_loss_on_aux=False,
        rope_2d=False,
        learned_pe=False,
        disable_v_head=False,
        # latent-space + hierarchical extensions (pmfDiT compatibility):
        use_token_embed=False,
        num_levels=0,
        use_level_cond=False,
        num_level_tokens=8,
        use_prev_cond=False,
        cond_token_mode="fused",
        mlp_round_to_8=True,
    ):
        super().__init__()
        assert tokenizer_patch_size == 1, "tokenizer_patch_size must be 1 for pMF"

        self.input_size = self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.label_drop_prob = label_drop_prob
        self.P_mean = P_mean
        self.P_std = P_std
        self.ratio_r_neq_t = ratio_r_neq_t
        self.t_eps = t_eps
        self.cfg_beta = cfg_beta
        self.cfg_omega_max = cfg_omega_max
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.tr_uniform = tr_uniform
        self.perceptual_threshold = perceptual_threshold
        self.perceptual_loss_on_aux = perceptual_loss_on_aux
        self.use_level_cond = use_level_cond
        self.use_prev_cond = use_prev_cond
        # default noise_scale heuristic only makes sense for pixel-space pMF;
        # fall back to 1.0 if no scale is given and we're in latent mode.
        if noise_scale is not None:
            self.noise_scale = noise_scale
        elif use_token_embed:
            self.noise_scale = 1.0
        else:
            self.noise_scale = img_size / 256.0
        if backbone not in MiT_models:
            raise ValueError(f"unknown backbone: {backbone}. available: {list(MiT_models.keys())}")
        self.net = MiT_models[backbone](
            input_size=self.input_size,
            in_channels=in_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            aux_head_depth=aux_head_depth,
            num_class_tokens=class_tokens,
            num_time_tokens=time_tokens,
            num_cfg_tokens=guidance_tokens,
            num_interval_tokens=interval_tokens,
            token_init_constant=token_init_constant,
            embedding_init_constant=embedding_init_constant,
            weight_init_constant=weight_init_constant,
            bottleneck_dim=bottleneck_dim,
            output_type="x",
            rope_2d=rope_2d,
            learned_pe=learned_pe,
            disable_v_head=disable_v_head,
            t_eps=t_eps,
            use_token_embed=use_token_embed,
            num_levels=num_levels,
            use_level_cond=use_level_cond,
            num_level_tokens=num_level_tokens,
            use_prev_cond=use_prev_cond,
            cond_token_mode=cond_token_mode,
            mlp_round_to_8=mlp_round_to_8,
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[pMF Denoiser] params: {n_params:.2f}M, backbone: {backbone}, rope_2d: {rope_2d}, learned_pe: {learned_pe}")
        logger.info(f"[pMF Denoiser] noise_scale: {self.noise_scale:.3f}")

    def sample_t(self, n, device):
        return torch.sigmoid(torch.randn(n, 1, 1, 1, device=device) * self.P_std + self.P_mean)

    def sample_tr(self, n, device):
        t = self.sample_t(n, device)
        r = self.sample_t(n, device)
        # ensure t >= r
        # t, r = torch.maximum(t, r), torch.minimum(t, r)
        if self.tr_uniform:
            # 10% random tr samples
            unif_mask = torch.rand((n, 1, 1, 1), device=device) < 0.1
            t = torch.where(unif_mask, torch.rand((n, 1, 1, 1), device=device), t)
            r = torch.where(unif_mask, torch.rand((n, 1, 1, 1), device=device), r)

        # set r=t for FM samples first, then ensure t >= r (matches JAX ordering)
        data_size = int(n * self.ratio_r_neq_t)
        fm_mask = (torch.arange(n, device=device) < data_size).view(n, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, n, device):
        u = torch.rand(n, 1, 1, 1, device=device)
        if self.cfg_beta == 1.0:
            # log-uniform
            return torch.exp(u * math.log1p(self.cfg_omega_max))
        b = self.cfg_beta
        log_base = (1.0 - b) * math.log1p(self.cfg_omega_max)
        return torch.exp(torch.log1p(u * math.expm1(log_base)) / (1.0 - b))

    def sample_cfg_interval(self, n, device, fm_mask):
        t_min = torch.rand(n, 1, 1, 1, device=device) * 0.5
        t_max = torch.rand(n, 1, 1, 1, device=device) * 0.5 + 0.5
        # flow matching samples get full interval [0, 1]
        t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        return t_min, t_max

    def u_fn(self, x, t, h, omega, t_min, t_max, y, level=None, cond=None, skip_v=False,
             net_params=None):
        bz = x.shape[0]
        net_kwargs = dict(
            x=x, t=t.reshape(bz), h=h.reshape(bz),
            omega=omega.reshape(bz), t_min=t_min.reshape(bz),
            t_max=t_max.reshape(bz), y=y, level=level, cond=cond, skip_v=skip_v,
        )
        if net_params is None:
            return self.net(**net_kwargs)
        # Drive the shared backbone with an alternate parameter set (e.g. a
        # per-level differentiable clone). Buffers not in the dict fall back to
        # the module's own. Gradients land on whatever leaf tensors net_params
        # holds, isolating this invocation's contribution.
        return torch.func.functional_call(self.net, net_params, (), net_kwargs)

    def v_cond_fn(self, x, t, omega, y):
        bz = x.shape[0]
        h = torch.zeros(bz, device=x.device)
        t_min = torch.zeros(bz, device=x.device)
        t_max = torch.ones(bz, device=x.device)
        _, v = self.u_fn(x, t, h, omega, t_min, t_max, y)
        return v

    def v_fn(self, x, t, omega, y):
        bz = x.shape[0]
        x_double = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, device=y.device, dtype=y.dtype)
        y_double = torch.cat([y, y_null], dim=0)
        t_double = torch.cat([t, t], dim=0)
        omega_double = torch.cat([omega, torch.ones_like(omega)], dim=0)
        out = self.v_cond_fn(x_double, t_double, omega_double, y_double)
        return torch.chunk(out, 2, dim=0)

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        device = v_t.device
        rand_mask = torch.rand(bz, device=device) < self.label_drop_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=device)[:, None, None, None] < num_drop
        labels = torch.where(drop_mask.reshape(bz), torch.full_like(labels, self.num_classes), labels)
        v_g = torch.where(drop_mask, v_t, v_g)
        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, omega, t_min, t_max):
        v_c, v_u = self.v_fn(z_t, t, omega, y)

        # flow matching samples: no interval restriction
        v_g_fm = v_t + (1 - 1 / omega) * (v_c - v_u)

        # apply cfg only when t in [t_min, t_max]
        omega = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        v_c = self.v_cond_fn(z_t, t, omega, y)
        v_g = v_t + (1 - 1 / omega) * (v_c - v_u)

        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c

    def adaptive_weight(self, loss_per_sample):
        weight = (loss_per_sample + self.norm_eps) ** self.norm_p
        return loss_per_sample / weight.detach()

    def forward(self, x, y, aux_loss_fn=None):
        B, device = x.shape[0], x.device

        t, r, fm_mask = self.sample_tr(B, device)
        e = torch.randn_like(x) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(self.t_eps, 1.0)

        t_min, t_max = self.sample_cfg_interval(B, device, fm_mask)
        omega = self.sample_cfg_scale(B, device)
        v_g, v_c = self.guidance_fn(v_t, z_t, t, r, y, fm_mask, omega, t_min, t_max)

        labels, v_g = self.cond_drop(v_t, v_g, y)

        def u_fn_for_dudt(z_in, t_in, r_in):
            return self.u_fn(z_in, t_in, t_in - r_in, omega, t_min, t_max, labels)

        u, du_dt, v = torch.func.jvp(
            u_fn_for_dudt, (z_t, t, r),
            (v_c, torch.ones_like(t), torch.zeros_like(r)), has_aux=True,
        )

        # V = u + (t - r) * stop_grad(du/dt)
        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()

        loss_u = ((V - v_g) ** 2).sum(dim=(1, 2, 3))
        loss_v = ((v - v_g) ** 2).sum(dim=(1, 2, 3))

        loss_u_w = self.adaptive_weight(loss_u)
        loss_v_w = self.adaptive_weight(loss_v)

        if aux_loss_fn is not None and self.training:
            pred_x = z_t - t * u
            # only apply perceptual loss when t < threshold
            mask = t.view(-1) < self.perceptual_threshold
            aux_loss, aux_loss_dict = aux_loss_fn(pred_x, x, mask)
            
            if self.perceptual_loss_on_aux:
                pred_x_aux = z_t - t * v
                aux_loss_aux, aux_loss_dict_aux = aux_loss_fn(pred_x_aux, x, mask)
                aux_loss = aux_loss + 0.5 * aux_loss_aux
                aux_loss_dict.update(
                    {f"v_head_{k}": v for k, v in aux_loss_dict_aux.items()}
                )
        else:
            aux_loss_dict = {}
            aux_loss = torch.zeros(B, device=device)
        loss = (loss_u_w + loss_v_w + aux_loss).mean()

        loss_dict = {
            # "total_loss": loss.item(), # loss will be logged directly by the trainer, no need to log here
            "loss_u": ((V - v_g) ** 2).mean().item(),
            "loss_v": ((v - v_g) ** 2).mean().item(),
            **aux_loss_dict,
        }
        return loss, loss_dict
    
    def _sample_one_level(self, z, y, omega, t_min, t_max, num_steps, level=None, cond=None,
                          net_params=None):
        """One-level meanflow sampling (matches pmf.py:sample_one_step looped num_steps times).

        ``net_params`` (optional) routes the backbone forward through an alternate
        parameter dict via functional_call — used to give each level its own
        differentiable weight clone for per-level gradient attribution.
        """
        bsz, device = z.shape[0], z.device
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        x = z
        for i in range(num_steps):
            t_cur = t_steps[i].expand(bsz)
            h_t = (t_cur - t_steps[i + 1]).expand(bsz).view(-1, 1, 1, 1)
            # skip_v=True: sampling only consumes u, so don't compute v_heads
            # (saves ~8 transformer blocks of activation memory per step).
            u = self.u_fn(x, t_cur, h_t, omega, t_min, t_max, y,
                          level=level, cond=cond, skip_v=True, net_params=net_params)[0]
            x = x - h_t * u
        return x

    def sample_images_with_grad(self, x, y, sampling_args=None, level=None, cond=None):
        """Sample with gradients enabled.

        For hierarchical generators (use_level_cond=True), this runs the
        full level-cascade matching pmf.py: each level uses fresh Gaussian
        noise (input ``x`` becomes the level-0 noise) and conditions on the
        previous level's output. Otherwise it does a single-level sample.
        """
        bsz, device = x.shape[0], x.device
        if sampling_args is None:
            sampling_args = {}
        t_min_v = sampling_args.get("t_min", 0.4)
        t_max_v = sampling_args.get("t_max", 0.65)
        omega_v = sampling_args.get("cfg", 1.0)
        num_steps = sampling_args.get("num_steps", 1)

        t_min = torch.full((bsz,), t_min_v, device=device)
        t_max = torch.full((bsz,), t_max_v, device=device)
        omega = torch.full((bsz,), omega_v, device=device)

        if not self.use_level_cond:
            return self._sample_one_level(x, y, omega, t_min, t_max, num_steps,
                                          level=level, cond=cond)

        # Hierarchical cascade: each level uses fresh noise; first level seeded
        # by the input ``x``; gradients flow through every level.
        prev = torch.zeros_like(x) if cond is None else cond
        num_levels = int(self.net.num_levels)
        for level_id in range(num_levels):
            z_l = x if level_id == 0 else torch.randn_like(x) * float(self.noise_scale)
            level_t = torch.full((bsz,), level_id, dtype=torch.long, device=device)
            prev = self._sample_one_level(
                z_l, y, omega, t_min, t_max, num_steps,
                level=level_t, cond=prev,
            )
        return prev

    def sample_images_with_grad_per_level(self, x, y, level_param_dicts, sampling_args=None):
        """Hierarchical cascade where each level is driven by its own parameter dict.

        Identical forward to ``sample_images_with_grad`` (full L0->...->L{n-1}
        cascade, returns the final level's latent), except level ``k`` runs the
        shared backbone via ``functional_call`` with ``level_param_dicts[k]``.
        When those dicts are detached clones of the shared weights, a single
        backward of the final-image loss leaves, in each clone's ``.grad``,
        exactly that level's contribution to the final image — enabling per-level
        gradient balancing while still training one shared network.

        Requires ``use_level_cond=True``; ``len(level_param_dicts)`` must equal
        ``self.net.num_levels``.
        """
        assert self.use_level_cond, (
            "sample_images_with_grad_per_level requires a hierarchical "
            "(use_level_cond=True) model")
        num_levels = int(self.net.num_levels)
        assert len(level_param_dicts) == num_levels, (
            f"expected {num_levels} param dicts, got {len(level_param_dicts)}")

        bsz, device = x.shape[0], x.device
        sampling_args = sampling_args or {}
        t_min_v = sampling_args.get("t_min", 0.4)
        t_max_v = sampling_args.get("t_max", 0.65)
        omega_v = sampling_args.get("cfg", 1.0)
        num_steps = sampling_args.get("num_steps", 1)

        t_min = torch.full((bsz,), t_min_v, device=device)
        t_max = torch.full((bsz,), t_max_v, device=device)
        omega = torch.full((bsz,), omega_v, device=device)

        prev = torch.zeros_like(x)
        for level_id in range(num_levels):
            z_l = x if level_id == 0 else torch.randn_like(x) * float(self.noise_scale)
            level_t = torch.full((bsz,), level_id, dtype=torch.long, device=device)
            prev = self._sample_one_level(
                z_l, y, omega, t_min, t_max, num_steps,
                level=level_t, cond=prev,
                net_params=level_param_dicts[level_id],
            )
        return prev

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        device = labels.device
        dtype = next(self.parameters()).dtype

        num_steps = args.num_sampling_steps if args else 1
        t_min_val = args.interval_min if args else 0.4
        t_max_val = args.interval_max if args else 0.65

        x_shape = (n_samples, self.in_channels, self.input_size, self.input_size)
        if z_t is None: # sample noise if not provided
            if args.same_noise:
                z_t = torch.randn(1, *x_shape[1:], device=device, dtype=dtype)
                z_t = z_t.repeat(n_samples, *([1] * (len(x_shape) - 1)))
            else:
                z_t = torch.randn(x_shape, device=device, dtype=dtype)
            z_t = z_t * self.noise_scale

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=dtype, device=device)
        omega = torch.full((n_samples,), cfg, dtype=dtype, device=device)
        t_min = torch.full((n_samples,), t_min_val, dtype=dtype, device=device)
        t_max = torch.full((n_samples,), t_max_val, dtype=dtype, device=device)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        def _one_level(z_in, level=None, cond=None, desc="Generating"):
            pbar = (trange(num_steps, desc=f"[Rank{rank}] {desc}") if verbose
                    else range(num_steps))
            x = z_in
            for i in pbar:
                t_cur = t_steps[i].expand(n_samples)
                h_t = (t_cur - t_steps[i + 1]).expand(n_samples).view(-1, 1, 1, 1)
                u = self.u_fn(x, t_cur, h_t, omega, t_min, t_max, y=labels,
                              level=level, cond=cond, skip_v=True)[0]
                x = x - h_t * u
            return x

        if not self.use_level_cond:
            return _one_level(z_t)

        # Hierarchical cascade: 4 levels, each with fresh noise (matches pmf.py).
        prev = torch.zeros_like(z_t)
        num_levels = int(self.net.num_levels)
        for level_id in range(num_levels):
            z_l = z_t if level_id == 0 else torch.randn_like(z_t) * self.noise_scale
            level = torch.full((n_samples,), level_id, dtype=torch.long, device=device)
            prev = _one_level(z_l, level=level, cond=prev,
                              desc=f"Generating L{level_id}/{num_levels-1}")
        return prev

_LAYERS_RE = re.compile(r"\.layers_(\d+)\.")
_MLIST_RE = re.compile(r"\.(shared_blocks|u_heads|u_heads_wide|v_heads|v_heads_wide)_(\d+)\.")


def convert_pmf_checkpoint(state_dict):
    """Convert upstream pMF / pmfDiT checkpoint keys to torch state-dict.

    Handles both already-converted .pth files (FD-Loss-style cosmetic renames)
    and raw flat dumps from JAX/Flax (`._flax_linear.kernel` storing weights
    in (in, out) order, `_flax_embedding.embedding` triple, `kernel` for
    RMSNorm scales, nn.Sequential members named `layers_<i>`).
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        nk = key
        # Flax wrapper renames: TorchLinear._flax_linear (Dense) -> .linear (nn.Linear);
        # TorchEmbedding._flax_embedding (Embed) -> .embedding (nn.Embedding).
        nk = nk.replace("._flax_linear.", ".linear.")
        nk = nk.replace("._flax_embedding.", ".embedding.")
        # Flax nn.Sequential -> torch nn.Sequential indexing
        nk = _LAYERS_RE.sub(r".\1.", nk)
        # Flax ModuleList -> torch ModuleList indexing (shared_blocks_0 -> shared_blocks.0)
        nk = _MLIST_RE.sub(r".\1.\2.", nk)

        # Skip rope_freqs (computed on the fly).
        if "rope_freqs" in nk:
            continue

        is_tensor = hasattr(value, "ndim")

        # pos_embed stays (1, total_tokens, hidden_size); don't squeeze.
        is_pos_embed = nk.endswith("pos_embed")

        # Squeeze (1, N, D) -> (N, D) for *_tokens and embedding tables (not pos_embed).
        if is_tensor and value.ndim == 3 and value.shape[0] == 1 and not is_pos_embed:
            if hasattr(value, "squeeze"):
                value = value.squeeze(0)

        # Linear kernel (in, out) -> torch weight (out, in): rename and transpose.
        if nk.endswith(".linear.kernel"):
            nk = nk[: -len(".kernel")] + ".weight"
            if is_tensor and value.ndim == 2:
                if hasattr(value, "contiguous"):  # torch.Tensor
                    value = value.transpose(0, 1).contiguous()
                else:  # numpy array
                    value = value.T.copy()
        # TorchEmbedding leaf: <scope>.embedding_table.embedding (now squeezed (V, D))
        # -> <scope>.embedding_table.embedding.weight to match nn.Embedding.
        elif nk.endswith(".embedding_table.embedding"):
            nk = nk + ".weight"
        # RMSNorm scale param: stored as `kernel` in Flax, `weight` in torch.
        elif nk.endswith(".kernel"):
            nk = nk[: -len(".kernel")] + ".weight"

        new_state_dict[nk] = value
    return new_state_dict


# model registry
pMFDenoiser_models = {
    "pMF_T": lambda **kw: pMFDenoiser(backbone="MiT_T", bottleneck_dim=128, **kw),
    "pMF_B": lambda **kw: pMFDenoiser(backbone="MiT_B2", bottleneck_dim=128, **kw),
    "pMF_M": lambda **kw: pMFDenoiser(backbone="MiT_M", bottleneck_dim=128, **kw),
    "pMF_L": lambda **kw: pMFDenoiser(backbone="MiT_L", bottleneck_dim=128, **kw),
    "pMF_H": lambda **kw: pMFDenoiser(backbone="MiT_H", bottleneck_dim=256, **kw),
    "pMF_XL": lambda **kw: pMFDenoiser(backbone="MiT_XL", bottleneck_dim=256, **kw),
}


def _make_pmfDiT_B_16(**kw):
    """Hierarchical pmfDiT_B_16 (RAE-latent, 16x16x768, 4 levels, no CFG).

    The builder in utils/builders.py passes explicit CLI-derived kwargs to every
    registry entry; we override architecture-defining args here so the user
    doesn't need to remember to pass --img_size 16 --token_channels 768 etc.
    """
    forced = dict(
        backbone="MiT_B2",     # depth=16, hidden=768, heads=12
        img_size=16, patch_size=1, in_channels=768,
        bottleneck_dim=-1,
        use_token_embed=True,
        aux_head_depth=8,
        class_tokens=8, time_tokens=8,
        guidance_tokens=0, interval_tokens=0,    # use_cfg: false
        num_levels=4, use_level_cond=True, num_level_tokens=8,
        use_prev_cond=True, cond_token_mode="fused",
        rope_2d=True, learned_pe=True,
        weight_init_constant=1.0,
    )
    kw.update(forced)
    return pMFDenoiser(**kw)


pMFDenoiser_models["pmfDiT_B_16"] = _make_pmfDiT_B_16


def _make_pmfDiT_L_16(**kw):
    """Hierarchical pmfDiT_L_16 (RAE-latent, 16x16x768, 4 levels, no CFG).

    Same recipe as _make_pmfDiT_B_16 but the L backbone (MiT_L: depth=32,
    hidden=1024, heads=16)
    """
    forced = dict(
        backbone="MiT_L",      # depth=32, hidden=1024, heads=16
        img_size=16, patch_size=1, in_channels=768,
        bottleneck_dim=-1,
        use_token_embed=True,
        aux_head_depth=8,
        class_tokens=8, time_tokens=8,
        guidance_tokens=0, interval_tokens=0,    # use_cfg: false
        num_levels=4, use_level_cond=True, num_level_tokens=8,
        use_prev_cond=True, cond_token_mode="fused",
        rope_2d=True, learned_pe=True,
        weight_init_constant=1.0,
    )
    kw.update(forced)
    return pMFDenoiser(**kw)


pMFDenoiser_models["pmfDiT_L_16"] = _make_pmfDiT_L_16


def _make_pmfDiT_H_16(**kw):
    """Hierarchical pmfDiT_H_16 (RAE-latent, 16x16x768, 4 levels, no CFG).

    Same recipe as _make_pmfDiT_B_16 but the H backbone (MiT_H: depth=48,
    hidden=1280, heads=16)
    """
    forced = dict(
        backbone="MiT_H",      # depth=48, hidden=1280, heads=16
        img_size=16, patch_size=1, in_channels=768,
        bottleneck_dim=-1,
        # JAX pmfDiT_H_16 uses int(1280*8/3)=3413 for all MLPs (no rounding-to-8),
        # so disable the H/XL round-up to match the converted checkpoint exactly.
        mlp_round_to_8=False,
        use_token_embed=True,
        aux_head_depth=8,
        class_tokens=8, time_tokens=8,
        guidance_tokens=0, interval_tokens=0,    # use_cfg: false
        num_levels=4, use_level_cond=True, num_level_tokens=8,
        use_prev_cond=True, cond_token_mode="fused",
        rope_2d=True, learned_pe=True,
        weight_init_constant=1.0,
    )
    kw.update(forced)
    return pMFDenoiser(**kw)


pMFDenoiser_models["pmfDiT_H_16"] = _make_pmfDiT_H_16
