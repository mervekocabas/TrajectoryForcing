import flax.linen as nn
import jax
import jax.numpy as jnp

from models import pmfDiT
from utils.time_sampler import shifted_logit_normal_dist, plateau_logit_normal_dist, symmetric_plateau_logit_normal_dist

def apply_time_dist_shift(t, time_dist_shift=10.0):
    """Apply resolution-aware time distribution shift on [0, 1] timesteps."""
    shift = jnp.asarray(time_dist_shift, dtype=t.dtype)
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega=1.0,
    t_min=0.0,
    t_max=1.0,
    sample_idx=None,
    levels=None,
    prev_cond=None,
    return_all_levels=None,
):
    """
    Generate samples from the model

    Args:
        variable: Model parameters.
        model: pixelMeanFlow model.
        rng: JAX random key.
        n_sample: Number of samples to generate.
        config: Configuration object.
        num_steps: Number of sampling steps.
        omega: CFG scale.
        t_min, t_max: Guidance interval.
        sample_idx: Optional index for class-conditional sampling.
        levels: Optional level ids, shape (B,). If None and hierarchical
            conditioning is enabled, generation runs level-by-level from 0 to
            num_levels-1, feeding generated level k as prev_cond for k+1.
        prev_cond: Optional previous-level conditioning, shape (B, H, W, C).
        return_all_levels: Optional override for whether to return all generated
            levels stacked as (B, L, H, W, C) in hierarchical generation mode.

    Returns:
        images: Generated images.
    """
    num_classes = int(config.dataset.num_classes)
    img_size = int(config.dataset.image_size)
    img_channels = int(config.dataset.image_channels)
    x_shape = (n_sample, img_size, img_size, img_channels)

    rng, rng_sample = jax.random.split(rng)

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
    if getattr(model, "time_shift", False):
        t_steps = apply_time_dist_shift(t_steps, getattr(model, "time_dist_shift", 10.0))

    def sample_one_level(level_ids, cond_latent):
        nonlocal rng
        rng, rng_level = jax.random.split(rng)
        z_t = jax.random.normal(rng_level, x_shape, dtype=model.dtype) * model.noise_scale

        def step_fn(i, x_i):
            return model.apply(
                variable,
                x_i,
                y,
                level_ids,
                i,
                t_steps,
                omega,
                t_min,
                t_max,
                cond_latent,
                method=model.sample_one_step,
            )

        return jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    auto_hier = (
        levels is None
        and (bool(getattr(model, "use_prev_cond", False)) or bool(getattr(model, "use_level_cond", False)))
    )
    if auto_hier:
        num_levels = int(getattr(model, "num_levels", int(config.dataset.get("num_levels", 1))))
        prev = jnp.zeros(x_shape, dtype=model.dtype) if prev_cond is None else prev_cond.astype(model.dtype)
        level_outputs = []
        ret_all = (
            bool(config.sampling.get("return_all_levels", False))
            if return_all_levels is None
            else bool(return_all_levels)
        )
        for level_id in range(num_levels):
            level_ids = jnp.full((n_sample,), level_id, dtype=jnp.int32)
            level_latent = sample_one_level(level_ids, prev)
            level_outputs.append(level_latent)
            prev = level_latent

        if ret_all:
            return jnp.stack(level_outputs, axis=1)
        return level_outputs[-1]

    if levels is None:
        levels = jnp.zeros((n_sample,), dtype=jnp.int32)
    else:
        levels = jnp.asarray(levels, dtype=jnp.int32)
        if levels.ndim == 0:
            levels = jnp.full((n_sample,), levels, dtype=jnp.int32)

    if prev_cond is None:
        prev_cond = jnp.zeros(x_shape, dtype=model.dtype)
    else:
        prev_cond = prev_cond.astype(model.dtype)

    return sample_one_level(levels, prev_cond)


def generate_autoguidance(
    variable_good,
    variable_bad,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    guidance_scale=1.5,
    omega=1.0,
    t_min=0.0,
    t_max=1.0,
    sample_idx=None,
    return_all_levels=None,
):
    """
    Generate samples using AutoGuidance: u_guided = u_good + scale * (u_good - u_bad).

    Args:
        variable_good: Parameters of the well-trained model.
        variable_bad: Parameters of the degraded (early checkpoint) model.
        model: pixelMeanFlow model instance (shared architecture).
        guidance_scale: AutoGuidance scale (1.0 = no guidance).
        omega: In-context CFG scale passed to both models.
        t_min, t_max: In-context CFG interval passed to both models.
    """
    num_classes = int(config.dataset.num_classes)
    img_size = int(config.dataset.image_size)
    img_channels = int(config.dataset.image_channels)
    x_shape = (n_sample, img_size, img_size, img_channels)

    rng, rng_sample = jax.random.split(rng)

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
    if getattr(model, "time_shift", False):
        t_steps = apply_time_dist_shift(t_steps, getattr(model, "time_dist_shift", 10.0))

    def sample_one_level(level_ids, cond_latent):
        nonlocal rng
        rng, rng_level = jax.random.split(rng)
        z_t = jax.random.normal(rng_level, x_shape, dtype=model.dtype) * model.noise_scale

        def step_fn(i, x_i):
            # u from good model
            u_good = model.apply(
                variable_good, x_i, y, level_ids, i, t_steps,
                omega, t_min, t_max, cond_latent,
                method=model.sample_one_step_u_only,
            )
            # u from bad model
            u_bad = model.apply(
                variable_bad, x_i, y, level_ids, i, t_steps,
                omega, t_min, t_max, cond_latent,
                method=model.sample_one_step_u_only,
            )
            # AutoGuidance combination
            u_guided = u_good + guidance_scale * (u_good - u_bad)

            t = jnp.take(t_steps, i)
            r = jnp.take(t_steps, i + 1)
            return x_i - jnp.einsum("n,n...->n...", jnp.broadcast_to(t - r, (n_sample,)), u_guided)

        return jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    auto_hier = (
        bool(getattr(model, "use_prev_cond", False))
        or bool(getattr(model, "use_level_cond", False))
    )
    if auto_hier:
        num_levels = int(getattr(model, "num_levels", int(config.dataset.get("num_levels", 1))))
        prev = jnp.zeros(x_shape, dtype=model.dtype)
        level_outputs = []
        ret_all = (
            bool(config.sampling.get("return_all_levels", False))
            if return_all_levels is None
            else bool(return_all_levels)
        )
        for level_id in range(num_levels):
            level_ids = jnp.full((n_sample,), level_id, dtype=jnp.int32)
            level_latent = sample_one_level(level_ids, prev)
            level_outputs.append(level_latent)
            prev = level_latent

        if ret_all:
            return jnp.stack(level_outputs, axis=1)
        return level_outputs[-1]

    levels = jnp.zeros((n_sample,), dtype=jnp.int32)
    prev_cond = jnp.zeros(x_shape, dtype=model.dtype)
    return sample_one_level(levels, prev_cond)


class pixelMeanFlow(nn.Module):
    """pixel MeanFlow"""

    # Model and dataset
    model_str: str
    half_precision: bool = False
    num_classes: int = 1000
    input_size: int = 256
    in_channels: int = 3
    num_levels: int = 4
    use_level_cond: bool = False
    use_prev_cond: bool = False
    use_token_embed: bool = False
    num_level_tokens: int = 2
    num_time_tokens: int = 8
    cond_token_mode: str = "full_concat"  # full_concat | fused
    gradient_checkpointing: bool = False
    hidden_size: int = 0  # 0 = use model_str default
    num_heads: int = 0  # 0 = use model_str default
    mlp_ratio: float = 0.0  # 0 = use model_str default
    depth: int = 0  # 0 = use model_str default
    aux_head_depth: int = 8
    head_wide_layers: int = 0
    head_wide_size: int = 2048
    head_wide_num_heads: int = 16
    # v-side head overrides. -1 means "follow u-side".
    v_aux_head_depth: int = -1
    v_head_wide_layers: int = -1
    v_head_wide_size: int = -1
    v_head_wide_num_heads: int = -1
    # Set to True only to evaluate old checkpoints trained before the
    # DDT-style time re-injection / RoPE fix. See pmfDiT.legacy_wide_head.
    legacy_wide_head: bool = False

    # Noise distribution (t scheduler)
    P_mean: float = -0.4
    P_std: float = 1.0
    use_cfg: bool = True
    cfg_max: float = 7.0
    noise_scale: float = 1.0
    time_shift: bool = False
    time_dist_shift: float = 10.0

    # Separate r scheduler (defaults to same as t if r_scheduler is "same")
    r_scheduler: str = "same"  # "same" | "logit_normal" | "plateau_lognormal" | "shifted_logit_normal"
    r_P_mean: float = -0.4
    r_P_std: float = 1.0
    r_time_shift: bool = False
    r_time_dist_shift: float = 1.0

    # Noisy conditioning (exposure bias mitigation)
    noisy_cond_prob: float = 0.0   # probability of adding noise to prev conditioning
    noisy_cond_alpha: float = 0.25 # noise strength: prev_noisy = prev + alpha * noise

    # Loss
    data_proportion: float = 0.5
    v_loss_weight: float = 1.0  # weight for auxiliary v-head loss (0 to disable)
    cfg_beta: float = 1.0
    class_dropout_prob: float = 0.1
    fm_low_t_prob: float = 0.0  # fraction of FM samples that use r's (low-t) distribution

    # Training dynamics
    norm_p: float = 1.0
    norm_eps: float = 0.01
    struct_weight: float = 0.0
    region_var_weight: float = 0.0
    region_collapse_weight: float = 0.0

    # Evaluation mode
    eval: bool = False

    # perceptual
    lpips: bool = False
    lpips_lambda: float = 1.0
    convnext: bool = False
    convnext_lambda: float = 0.0
    perceptual_max_t: float = 1.0
    
    # tr_uniform bool
    tr_uniform: bool = False
    tr_uniform_prob: float = 0.1
    use_plateau_lognormal: bool = False
    
    @property
    def dtype(self):
        return jnp.bfloat16 if self.half_precision else jnp.float32

    def setup(self):
        """
        Setup pixel MeanFlow model.
        """
        net_fn = getattr(pmfDiT, self.model_str)
        net_kwargs = dict(
            name="net",
            num_classes=self.num_classes,
            eval=self.eval,
            input_size=self.input_size,
            in_channels=self.in_channels,
            num_levels=self.num_levels,
            use_level_cond=self.use_level_cond,
            use_prev_cond=self.use_prev_cond,
            use_token_embed=self.use_token_embed,
            num_level_tokens=self.num_level_tokens,
            num_time_tokens=self.num_time_tokens,
            cond_token_mode=self.cond_token_mode,
            use_cfg=self.use_cfg,
            gradient_checkpointing=self.gradient_checkpointing,
            v_loss_weight=self.v_loss_weight,
            aux_head_depth=self.aux_head_depth,
            head_wide_layers=self.head_wide_layers,
            head_wide_size=self.head_wide_size,
            head_wide_num_heads=self.head_wide_num_heads,
            v_aux_head_depth=self.v_aux_head_depth,
            v_head_wide_layers=self.v_head_wide_layers,
            v_head_wide_size=self.v_head_wide_size,
            v_head_wide_num_heads=self.v_head_wide_num_heads,
            legacy_wide_head=self.legacy_wide_head,
            dtype=jnp.bfloat16 if self.half_precision else None,
        )
        if self.hidden_size > 0:
            net_kwargs["hidden_size"] = self.hidden_size
        if self.num_heads > 0:
            net_kwargs["num_heads"] = self.num_heads
        if self.mlp_ratio > 0:
            net_kwargs["mlp_ratio"] = self.mlp_ratio
        if self.depth > 0:
            net_kwargs["depth"] = self.depth
        self.net: pmfDiT.pmfDiT = net_fn(**net_kwargs)

    #######################################################
    #                       Solver                        #
    #######################################################

    def sample_one_step(
        self, z_t, labels, levels, i, t_steps, omega, t_min, t_max, prev_cond=None
    ):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            levels: Level indices for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale (ignored when use_cfg=False).
            t_min, t_max: Guidance interval (ignored when use_cfg=False).
        """
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        if not self.use_cfg:
            omega = jnp.ones((bsz,))
            t_min = jnp.zeros((bsz,))
            t_max = jnp.ones((bsz,))
        else:
            omega = jnp.broadcast_to(omega, (bsz,))
            t_min = jnp.broadcast_to(t_min, (bsz,))
            t_max = jnp.broadcast_to(t_max, (bsz,))
        if levels is None:
            levels = jnp.zeros((bsz,), dtype=jnp.int32)
        if prev_cond is None:
            prev_cond = jnp.zeros_like(z_t)

        u = self.u_fn(
            z_t,
            t,
            t - r,
            omega,
            t_min,
            t_max,
            y=labels,
            level=levels,
            cond=prev_cond,
        )[0]

        return z_t - jnp.einsum("n,n...->n...", t - r, u)

    def sample_one_step_u_only(
        self, z_t, labels, levels, i, t_steps, omega, t_min, t_max, prev_cond=None
    ):
        """Return raw u prediction without applying the ODE step (for AutoGuidance)."""
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        omega = jnp.broadcast_to(jnp.asarray(omega), (bsz,))
        t_min = jnp.broadcast_to(jnp.asarray(t_min), (bsz,))
        t_max = jnp.broadcast_to(jnp.asarray(t_max), (bsz,))
        if levels is None:
            levels = jnp.zeros((bsz,), dtype=jnp.int32)
        if prev_cond is None:
            prev_cond = jnp.zeros_like(z_t)

        return self.u_fn(
            z_t, t, t - r, omega, t_min, t_max,
            y=labels, level=levels, cond=prev_cond,
        )[0]

    #######################################################
    #                       Schedule                      #
    #######################################################

    # def logit_normal_dist(self, bz):
    #     rnd_normal = jax.random.normal(
    #         self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
    #     )
    #     return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)
    
    def logit_normal_dist(self, bz):
        shape = (bz, 1, 1, 1)
        rng = self.make_rng("gen")
        alpha = self.time_dist_shift if self.time_shift else 1.0

        if getattr(self, "use_plateau_lognormal", False):
            return plateau_logit_normal_dist(rng, shape, alpha, self.P_mean, self.P_std).astype(self.dtype)

        if self.time_shift:
            return shifted_logit_normal_dist(rng, shape, alpha, self.P_mean, self.P_std).astype(self.dtype)

        # fallback: vanilla logit-normal
        rnd_normal = jax.random.normal(rng, shape, dtype=self.dtype)
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def _sample_r_dist(self, bz):
        """Sample r from its own scheduler (configured via r_scheduler, r_P_mean, etc.)."""
        shape = (bz, 1, 1, 1)
        rng = self.make_rng("gen")
        alpha = self.r_time_dist_shift if self.r_time_shift else 1.0

        if self.r_scheduler == "plateau_lognormal":
            return plateau_logit_normal_dist(rng, shape, alpha, self.r_P_mean, self.r_P_std).astype(self.dtype)
        elif self.r_scheduler == "symmetric_plateau_lognormal":
            return symmetric_plateau_logit_normal_dist(rng, shape, alpha, self.r_P_mean, self.r_P_std).astype(self.dtype)
        elif self.r_scheduler == "shifted_logit_normal":
            return shifted_logit_normal_dist(rng, shape, alpha, self.r_P_mean, self.r_P_std).astype(self.dtype)
        else:  # "logit_normal"
            rnd_normal = jax.random.normal(rng, shape, dtype=self.dtype)
            return nn.sigmoid(rnd_normal * self.r_P_std + self.r_P_mean).astype(self.dtype)

    def sample_tr(self, bz):
        """
        Sample t and r from their respective distributions.
        """
        t = self.logit_normal_dist(bz)
        r = self._sample_r_dist(bz) if self.r_scheduler != "same" else self.logit_normal_dist(bz)

        if self.tr_uniform:
            # 10% random tr samples split a single RNG key to make randomness explicit/stable 
            rng_mask, rng_t, rng_r = jax.random.split(self.make_rng("gen"), 3)
            unif_mask = (
                jax.random.uniform(rng_mask, (bz, 1, 1, 1), dtype=self.dtype) < self.tr_uniform_prob
            )
            t_unif = jax.random.uniform(rng_t, (bz, 1, 1, 1), dtype=self.dtype)
            r_unif = jax.random.uniform(rng_r, (bz, 1, 1, 1), dtype=self.dtype)
            t = jnp.where(unif_mask, t_unif, t)
            r = jnp.where(unif_mask, r_unif, r)
        
        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        if self.fm_low_t_prob > 0.0:
            # For some FM samples, use r's distribution (low-t) instead of t's (high-t)
            low_t_mask = jax.random.uniform(
                self.make_rng("gen"), (bz, 1, 1, 1), dtype=self.dtype
            ) < self.fm_low_t_prob
            # fm_mask & low_t_mask: set t = r (low values)
            # fm_mask & ~low_t_mask: set r = t (high values, original behavior)
            use_low = fm_mask & low_t_mask
            r = jnp.where(fm_mask & ~low_t_mask, t, r)
            t = jnp.where(use_low, r, t)
        else:
            r = jnp.where(fm_mask, t, r)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, s_max=7.0):
        """
        Sample CFG scale omega from power distribution.
        """
        ukey = self.make_rng("gen")
        u = jax.random.uniform(
            ukey, (bz, 1, 1, 1), minval=0.0, maxval=1.0, dtype=jnp.float32
        )

        if self.cfg_beta == 1.0:  # special case for \int 1/x
            s = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            smax = jnp.asarray(s_max, jnp.float32)
            b = jnp.asarray(self.cfg_beta, jnp.float32)

            log_base = (1.0 - b) * jnp.log1p(smax)
            log_inner = jnp.log1p(u * jnp.expm1(log_base))

            s = jnp.exp(log_inner / (1.0 - b))

        return jnp.asarray(s, jnp.float32)

    def sample_cfg_interval(self, bz, fm_mask=None):
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        """
        rng_start, rng_end = jax.random.split(self.make_rng("gen"))

        t_min = jax.random.uniform(
            rng_start, (bz, 1, 1, 1), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (bz, 1, 1, 1), minval=0.5, maxval=1.0, dtype=self.dtype
        )

        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y, level=None, cond=None):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v, x_u_pred)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        if level is None:
            level = jnp.zeros((bz,), dtype=jnp.int32)
        if cond is None:
            cond = jnp.zeros_like(x)
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
            level=level.reshape(bz),
            cond=cond,
        )

    def v_cond_fn(self, x, t, omega, y, level=None, cond=None):
        """
        Compute the predicted v component conditioned on class labels.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v: Predicted v component.
        """

        # Set h, t_min, t_max to dummy values for v prediction
        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)

        v = self.u_fn(
            x,
            t,
            h,
            omega,
            t_min,
            t_max,
            y=y,
            level=level,
            cond=cond,
        )[1]

        return v

    def v_fn(self, x, t, omega, y, level=None, cond=None):
        """
        Compute both conditioned and unconditioned predicted v components.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v_c: Predicted v component conditioned on class labels.
            v_u: Predicted v component without class labels.
        """
        bz = x.shape[0]
        if level is None:
            level = jnp.zeros((bz,), dtype=jnp.int32)
        if cond is None:
            cond = jnp.zeros_like(x)

        # Create duplicated batch for conditioned and unconditioned predictions
        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.array([self.num_classes] * bz)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        w = jnp.concatenate([omega, jnp.ones_like(omega)], axis=0)
        level = jnp.concatenate([level, level], axis=0)
        cond = jnp.concatenate([cond, cond], axis=0)

        out = self.v_cond_fn(x, t, w, y, level=level, cond=cond)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]

        # per-sample Bernoulli mask
        rng = self.make_rng("gen")
        rand_mask = jax.random.uniform(rng, shape=(bz,)) < self.class_dropout_prob
        drop_mask = rand_mask[:, None, None, None]
        labels = jnp.where(rand_mask, self.num_classes, labels)
        
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, level, cond, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        # compute CFG target
        v_c, v_u = self.v_fn(z_t, t, w, y=y, level=level, cond=cond)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)

        v_c = self.v_cond_fn(z_t, t, w, y=y, level=level, cond=cond)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = jnp.where(fm_mask, v_g_fm, v_g)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    @staticmethod
    def _safe_norm(x, axis=-1, eps=1e-8):
        return jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=axis), eps))

    @staticmethod
    def _cosine(x, y, axis=-1, eps=1e-8):
        x_n = x / pixelMeanFlow._safe_norm(x, axis=axis, eps=eps)[..., None]
        y_n = y / pixelMeanFlow._safe_norm(y, axis=axis, eps=eps)[..., None]
        return jnp.sum(x_n * y_n, axis=axis)

    @staticmethod
    def _region_mean_tensors(pred_flat, gt_flat, ids_flat):
        """Build per-region means with dense ids in [0, H*W), invalid = -1."""
        n_tokens = pred_flat.shape[0]
        dtype = pred_flat.dtype

        valid_token = ids_flat >= 0
        ids_safe = jnp.where(valid_token, ids_flat, 0)

        one_hot = jax.nn.one_hot(ids_safe, n_tokens, dtype=dtype)
        one_hot = one_hot * valid_token[:, None].astype(dtype)

        counts = jnp.sum(one_hot, axis=0)
        denom = jnp.maximum(counts[:, None], 1.0)

        pred_means = (one_hot.T @ pred_flat) / denom
        gt_means = (one_hot.T @ gt_flat) / denom
        valid_regions = counts > 0

        return pred_means, gt_means, valid_regions, ids_safe, valid_token

    @staticmethod
    def _structural_loss(pred, gt, region_ids):
        """Cosine similarity loss between region means."""
        bsz, h, w, c = pred.shape
        pred_flat = pred.reshape(bsz, h * w, c)
        gt_flat = gt.reshape(bsz, h * w, c)

        def per_sample(pf, gf, ids):
            pm, gm, valid_regions, _ids_safe, _valid_token = pixelMeanFlow._region_mean_tensors(
                pf, gf, ids.reshape(-1)
            )
            cos = pixelMeanFlow._cosine(pm, gm, axis=-1)
            losses = ((1.0 - cos) ** 2) * valid_regions.astype(pf.dtype)
            denom = jnp.maximum(jnp.sum(valid_regions.astype(pf.dtype)), 1.0)
            sample_loss = jnp.sum(losses) / denom
            sample_valid = jnp.any(valid_regions)
            return sample_loss, sample_valid

        losses, valid = jax.vmap(per_sample)(pred_flat, gt_flat, region_ids)
        valid_f = valid.astype(pred.dtype)
        return jnp.sum(losses * valid_f) / jnp.maximum(jnp.sum(valid_f), 1.0)

    @staticmethod
    def _region_variance_loss(pred, gt, region_ids):
        bsz, h, w, c = pred.shape
        pred_flat = pred.reshape(bsz, h * w, c)
        gt_flat = gt.reshape(bsz, h * w, c)

        def per_sample(pf, gf, ids):
            n_tokens = pf.shape[0]
            _pm, gm, _valid_regions, ids_safe, valid_token = pixelMeanFlow._region_mean_tensors(
                pf, gf, ids.reshape(-1)
            )
            gt_tok = gm[ids_safe]
            cos_tok = pixelMeanFlow._cosine(pf, gt_tok, axis=-1)
            tok_loss = ((1.0 - cos_tok) ** 2) * valid_token.astype(pf.dtype)

            region_sum = jnp.zeros((n_tokens,), dtype=pf.dtype).at[ids_safe].add(tok_loss)
            region_cnt = jnp.zeros((n_tokens,), dtype=pf.dtype).at[ids_safe].add(
                valid_token.astype(pf.dtype)
            )
            region_mean = region_sum / jnp.maximum(region_cnt, 1.0)

            valid_regions = region_cnt > 0
            denom = jnp.maximum(jnp.sum(valid_regions.astype(pf.dtype)), 1.0)
            sample_loss = jnp.sum(region_mean * valid_regions.astype(pf.dtype)) / denom
            sample_valid = jnp.any(valid_regions)
            return sample_loss, sample_valid

        losses, valid = jax.vmap(per_sample)(pred_flat, gt_flat, region_ids)
        valid_f = valid.astype(pred.dtype)
        return jnp.sum(losses * valid_f) / jnp.maximum(jnp.sum(valid_f), 1.0)

    @staticmethod
    def _region_collapse_loss(pred, region_ids):
        bsz, h, w, c = pred.shape
        pred_flat = pred.reshape(bsz, h * w, c)

        def per_sample(pf, ids):
            pm, _gm, valid_regions, _ids_safe, _valid_token = pixelMeanFlow._region_mean_tensors(
                pf, pf, ids.reshape(-1)
            )
            pm_n = pm / pixelMeanFlow._safe_norm(pm, axis=-1, eps=1e-8)[..., None]
            sims = pm_n @ pm_n.T

            n_tokens = sims.shape[0]
            off_diag = ~jnp.eye(n_tokens, dtype=bool)
            valid_pairs = valid_regions[:, None] & valid_regions[None, :] & off_diag
            valid_pairs_f = valid_pairs.astype(pf.dtype)
            denom = jnp.sum(valid_pairs_f)
            sample_loss = jnp.where(
                denom > 0,
                jnp.sum((sims**2) * valid_pairs_f) / denom,
                0.0,
            )
            sample_valid = jnp.sum(valid_regions.astype(jnp.int32)) > 1
            return sample_loss, sample_valid

        losses, valid = jax.vmap(per_sample)(pred_flat, region_ids)
        valid_f = valid.astype(pred.dtype)
        return jnp.sum(losses * valid_f) / jnp.maximum(jnp.sum(valid_f), 1.0)

    def forward(
        self,
        images,
        labels,
        prev=None,
        levels=None,
        region_ids=None,
        aux_fn=None,
        struct_weight=None,
        region_var_weight=None,
        region_collapse_weight=None,
    ):
        """
        Forward process of pixel MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).
            prev: Previous-level latent, shape (B, H, W, C).
            levels: Level ids in [0, num_levels-1], shape (B,).
            region_ids: Region ids with invalid entries -1, shape (B, H, W).

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]
        prev = jnp.zeros_like(x) if prev is None else prev.astype(self.dtype)
        levels = (
            jnp.zeros((bz,), dtype=jnp.int32)
            if levels is None
            else levels.astype(jnp.int32)
        )
        if region_ids is not None:
            region_ids = region_ids.astype(jnp.int32)

        # Noisy conditioning: add noise to prev for levels > 0 with some probability
        if self.noisy_cond_prob > 0:
            rng_nc_mask, rng_nc_noise = jax.random.split(self.make_rng("gen"))
            nc_mask = jax.random.uniform(rng_nc_mask, (bz, 1, 1, 1), dtype=self.dtype) < self.noisy_cond_prob
            nc_mask = nc_mask & (levels.reshape(bz, 1, 1, 1) > 0)  # skip level 0
            noise_cond = jax.random.normal(rng_nc_noise, prev.shape, dtype=self.dtype)
            prev = jnp.where(nc_mask, prev + self.noisy_cond_alpha * noise_cond, prev)

        if struct_weight is None:
            struct_weight = self.struct_weight
        if region_var_weight is None:
            region_var_weight = self.region_var_weight
        if region_collapse_weight is None:
            region_collapse_weight = self.region_collapse_weight

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz)

        rng = self.make_rng("gen")
        rng, rng_used1 = jax.random.split(rng)
        rng, rng_used2 = jax.random.split(rng)

        e = jax.random.normal(rng_used1, x.shape, dtype=self.dtype) * self.noise_scale
        z_t = (1 - t) * x + t * e # note to me t=1 noise t=0 clean
        v_t = (z_t - x) / jnp.clip(t.reshape((-1, 1, 1, 1)), 0.05, 1.0)

        if self.use_cfg:
            # Sample CFG scale and interval
            t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
            omega = self.sample_cfg_scale(bz, s_max=self.cfg_max)

            # Compute guided velocity v_g and conditioned velocity v_c
            v_g, v_c = self.guidance_fn(
                v_t,
                z_t,
                t,
                r,
                labels,
                levels,
                prev,
                fm_mask,
                omega,
                t_min,
                t_max,
            )

            # Cond dropout (dropout class labels)
            labels, v_g = self.cond_drop(v_t, v_g, labels)
        else:
            # No CFG: target is raw velocity, dummy values for omega/t_min/t_max
            omega = jnp.ones((bz, 1, 1, 1), dtype=self.dtype)
            t_min = jnp.zeros((bz, 1, 1, 1), dtype=self.dtype)
            t_max = jnp.ones((bz, 1, 1, 1), dtype=self.dtype)
            v_g = v_t
            if self.v_loss_weight > 0:
                v_c = self.v_cond_fn(z_t, t, omega, y=labels, level=levels, cond=prev)
            else:
                v_c = jax.lax.stop_gradient(v_t)  # use ground truth velocity as JVP tangent

        # jax.jvp(..., has_aux=True) expects fn -> (primal, aux)
        def u_fn_wrapped(z_t, t, r):
            u, v, x_u_pred = self.u_fn(
                z_t,
                t,
                t - r,
                omega,
                t_min,
                t_max,
                y=labels,
                level=levels,
                cond=prev,
            )
            return u, (v, x_u_pred)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, aux = jax.jvp(
            u_fn_wrapped, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True
        )
        v, x_u_pred = aux

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * jax.lax.stop_gradient(du_dt)

        v_g = jax.lax.stop_gradient(v_g)
        
        # Upcast to float32 for stable loss computation
        V = V.astype(jnp.float32)
        v = v.astype(jnp.float32)
        v_g = v_g.astype(jnp.float32)
        x_u_pred = x_u_pred.astype(jnp.float32)
        x = x.astype(jnp.float32)
        
        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # pixel MeanFlow objective is conceptually v-loss
        loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss
        loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        # aux loss
        if self.convnext or self.lpips:
            assert aux_fn is not None, "auxiliary loss function is not provided."

            pred_x = x_u_pred

            aux_loss_lpips, aux_loss_convnext = aux_fn(
                pred_x, x, rng_used2
            )  # shape (B,)

            mask = t.flatten() < self.perceptual_max_t

            aux_loss_lpips = jnp.where(mask, aux_loss_lpips, 0.0)
            aux_loss_convnext = jnp.where(mask, aux_loss_convnext, 0.0)

            aux_loss = (
                adp_wt_fn(aux_loss_lpips) * self.lpips_lambda
                + adp_wt_fn(aux_loss_convnext) * self.convnext_lambda
            )
        else:
            aux_loss = aux_loss_lpips = aux_loss_convnext = 0.0

        raw_mse_u = jnp.mean((V - v_g) ** 2)
        raw_mse_v = jnp.mean((v - v_g) ** 2)
        loss = jnp.mean(loss_u + self.v_loss_weight * loss_v + aux_loss)

        zero = jnp.asarray(0.0, dtype=x.dtype)
        struct_loss = zero
        var_loss = zero
        collapse_loss = zero
        if region_ids is not None:
            max_region_id = x.shape[1] * x.shape[2] - 1
            region_ids = jnp.where(
                (region_ids >= 0) & (region_ids <= max_region_id),
                region_ids,
                -1,
            )
            last_level_mask = (levels == (self.num_levels - 1)).reshape(-1, 1, 1)
            region_ids = jnp.where(last_level_mask, -1, region_ids)

            if struct_weight > 0:
                struct_loss = self._structural_loss(x_u_pred, x, region_ids)
                loss = loss + struct_weight * struct_loss
            if region_var_weight > 0:
                var_loss = self._region_variance_loss(x_u_pred, x, region_ids)
                loss = loss + region_var_weight * var_loss
            if region_collapse_weight > 0:
                collapse_loss = self._region_collapse_loss(x_u_pred, region_ids)
                loss = loss + region_collapse_weight * collapse_loss

        dict_losses = {
            "loss": loss,
            "loss_u": raw_mse_u,
            "loss_v": raw_mse_v,
            "raw_mse_u": raw_mse_u,
            "raw_mse_v": raw_mse_v,
            "struct_loss": struct_loss,
            "region_var_loss": var_loss,
            "region_collapse_loss": collapse_loss,
            "aux_loss_lpips": jnp.mean(aux_loss_lpips),
            "aux_loss_convnext": jnp.mean(aux_loss_convnext),
        }

        return loss, dict_losses

    def __call__(self, x, t, y, level=None, cond=None):
        if level is None:
            level = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        if cond is None:
            cond = jnp.zeros_like(x)
        return self.net(x, t, t, t, t, t, y, level=level, cond=cond)  # init only
