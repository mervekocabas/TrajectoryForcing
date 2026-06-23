"""
Training and evaluation for pixel MeanFlow.
"""

import os
import math
import time
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
from PIL import Image
from flax import jax_utils, serialization
from jax import lax, random
from functools import partial
from optax._src.alias import *

from pmf import pixelMeanFlow, generate, generate_autoguidance

import utils.input_pipeline as input_pipeline
from utils.ckpt_util import save_checkpoint, restore_checkpoint
from utils.ema_util import ema_schedules, update_ema
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.vis_util import make_grid_visualization, latent_levels_to_pca_column
from utils.lr_utils import lr_schedules
from utils.sample_util import get_fid_evaluator, run_p_sample_step
from utils.trainstate_util import create_train_state, TrainState

#######################################################
#                    Train Step                       #
#######################################################


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def maybe_save_train_input_latent_visuals(
    batch,
    out_dir,
    per_level_counts,
    max_per_level,
    gap=1,
    include_prev=True,
    step=None,
):
    """Save PCA-colored latent training inputs per level from the processed batch.

    This runs on host (rank 0 only) and stops automatically once each level reaches
    `max_per_level` saved samples.
    """
    if max_per_level <= 0 or "level" not in batch or "image" not in batch:
        return False

    if np.all(per_level_counts >= max_per_level):
        return True

    levels = np.asarray(jax.device_get(batch["level"])).reshape(-1)
    if levels.size == 0:
        return False

    images = np.asarray(jax.device_get(batch["image"]), dtype=np.float32)
    images = images.reshape((levels.shape[0],) + images.shape[-3:])

    labels = None
    if "label" in batch:
        labels = np.asarray(jax.device_get(batch["label"])).reshape(-1)

    prev = None
    if include_prev and "prev" in batch:
        prev = np.asarray(jax.device_get(batch["prev"]), dtype=np.float32)
        prev = prev.reshape((levels.shape[0],) + prev.shape[-3:])

    os.makedirs(out_dir, exist_ok=True)

    num_levels = int(per_level_counts.shape[0])
    for i, level_val in enumerate(levels):
        level_id = int(level_val)
        if level_id < 0 or level_id >= num_levels:
            continue
        if per_level_counts[level_id] >= max_per_level:
            continue

        if prev is not None and level_id > 0:
            stacked = np.stack([prev[i], images[i]], axis=0)
        else:
            stacked = images[i][None]

        panel = latent_levels_to_pca_column(stacked, gap=gap, bg_value=0)
        level_dir = os.path.join(out_dir, f"level_{level_id:02d}")
        os.makedirs(level_dir, exist_ok=True)

        sample_idx = int(per_level_counts[level_id])
        label_str = ""
        if labels is not None:
            label_str = f"_label{int(labels[i])}"
        step_str = ""
        if step is not None:
            step_str = f"_step{int(step):08d}"

        Image.fromarray(panel).save(
            os.path.join(
                level_dir,
                f"sample_{sample_idx:03d}_level{level_id:02d}{step_str}{label_str}.png",
            )
        )
        per_level_counts[level_id] += 1

        if np.all(per_level_counts >= max_per_level):
            return True

    return bool(np.all(per_level_counts >= max_per_level))


def train_step(state: TrainState, batch, rng_init, ema_fn, lr_fn, aux_fn=None):
    """
    Perform a single training step.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]  # [B, H, W, C]
    labels = batch["label"]
    prev = batch.get("prev", None)
    levels = batch.get("level", None)
    region_ids = batch.get("region_ids", None)

    def loss_fn(params):
        """loss function used for training."""
        loss, dict_loss = state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            prev=prev,
            levels=levels,
            region_ids=region_ids,
            aux_fn=aux_fn,
            rngs=dict(
                gen=rng_base,
            ),
        )
        return loss, dict_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    lr_value = lr_fn(state.step)

    dict_losses = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr_value
    
    # update ema params
    new_ema_params = {}
    for k, ema_param in new_state.ema_params.items():
        ema_value = ema_fn(new_state.step, k)
        new_ema = update_ema(ema_param, new_state.params, ema_value)
        new_ema_params[k] = new_ema
    new_state = new_state.replace(ema_params=new_ema_params)

    return new_state, metrics


#######################################################
#               Sampling and Metrics                  #
#######################################################


def sample_step(variable, sample_idx, model, rng_init, device_batch_size,
                config, num_steps, omega=1.0, t_min=0.0, t_max=1.0):
    """
    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold

    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max, sample_idx=sample_idx)

    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images


def latent_sample_step(
    variable, sample_idx, model, rng_init, device_batch_size, config, num_steps, omega=1.0, t_min=0.0, t_max=1.0
):
    """Sample latent hierarchy and return (B, L, H, W, C)."""
    rng_sample = random.fold_in(rng_init, sample_idx)
    return generate(
        variable,
        model,
        rng_sample,
        device_batch_size,
        config,
        num_steps,
        omega,
        t_min,
        t_max,
        sample_idx=sample_idx,
        return_all_levels=True,
    )


def autoguidance_sample_step(
    variable, variable_bad, sample_idx, model, rng_init, device_batch_size, config,
    num_steps, guidance_scale=1.5, omega=1.0, t_min=0.0, t_max=1.0,
):
    """Sample using AutoGuidance: u_guided = u_good + scale * (u_good - u_bad)."""
    rng_sample = random.fold_in(rng_init, sample_idx)
    images = generate_autoguidance(
        variable, variable_bad, model, rng_sample, device_batch_size,
        config, num_steps, guidance_scale=guidance_scale,
        omega=omega, t_min=t_min, t_max=t_max, sample_idx=sample_idx,
    )
    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images


#######################################################
#                       Main                          #
#######################################################


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ########### Walltime-aware checkpoint ###########
    _slurm_end = os.environ.get("SLURM_JOB_END_TIME")
    job_end_time = int(_slurm_end) if _slurm_end else None
    save_before_end = int(os.environ.get("PMF_SAVE_BEFORE_END_SECS", "300"))
    if job_end_time:
        log_for_0("Walltime guard enabled: will checkpoint %d s before job end (epoch %s).",
                   save_before_end, time.strftime("%H:%M:%S", time.localtime(job_end_time)))

    ########### Initialize ###########
    writer = Writer(config, workdir)

    config_save_path = os.path.join(workdir, "config.yml")
    if jax.process_index() == 0 and not os.path.exists(config_save_path):
        with open(config_save_path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        log_for_0("Saved training config to %s", config_save_path)

    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size

    log_for_0("config.training.batch_size: {}".format(config.training.batch_size))
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    ########### Create DataLoaders ###########
    train_loader, steps_per_epoch = input_pipeline.create_imagenet_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    use_flip = config.dataset.use_flip
    is_latent_kind = str(config.dataset.get("kind", "imagenet")).lower() in {"latent_hier", "latent"}
    pin_levels_to_device_groups = bool(config.dataset.get("pin_levels_to_device_groups", False))
    devices_per_level = int(config.dataset.get("devices_per_level", 1))
    num_levels = int(config.dataset.get("num_levels", 4))
    if pin_levels_to_device_groups:
        if not is_latent_kind:
            raise ValueError("dataset.pin_levels_to_device_groups is only supported for latent datasets.")
        if devices_per_level <= 0:
            raise ValueError(f"dataset.devices_per_level must be > 0, got {devices_per_level}")
        local_devices = jax.local_device_count()
        expected_devices = num_levels * devices_per_level
        if local_devices != expected_devices:
            raise ValueError(
                "To guarantee equal level counts per step, require "
                f"jax.local_device_count()==dataset.num_levels*dataset.devices_per_level "
                f"(got {local_devices} vs {num_levels}*{devices_per_level}={expected_devices})."
            )
        log_for_0(
            "Pinned level-by-device mapping enabled: %d local devices, %d levels, %d devices/level.",
            local_devices,
            num_levels,
            devices_per_level,
        )
    log_for_0("Steps per Epoch: {}".format(steps_per_epoch))

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model_config["num_classes"] = int(config.dataset.num_classes)
    model_config["input_size"] = int(config.dataset.image_size)
    model_config["in_channels"] = int(config.dataset.image_channels)
    model_config["num_levels"] = int(config.dataset.get("num_levels", 4))
    if is_latent_kind:
        model_config["use_token_embed"] = True
        model_config["use_prev_cond"] = True
        model_config["use_level_cond"] = True
    model_config["half_precision"] = bool(config.training.get("half_precision", False))
    # --- data_proportion schedule: list of [epoch, value] pairs ---
    dp_schedule = model_config.pop("data_proportion_schedule", None)
    model = pixelMeanFlow(**model_config)
    if dp_schedule:
        dp_schedule = sorted(dp_schedule, key=lambda x: x[0])
        log_for_0("data_proportion_schedule: %s", dp_schedule)

    def _get_scheduled_dp(epoch):
        """Return the data_proportion that should be active at the given epoch."""
        if not dp_schedule:
            return float(config.model.data_proportion)
        val = float(dp_schedule[0][1])  # default to first entry
        for ep, v in dp_schedule:
            if epoch >= ep:
                val = float(v)
        return val

    _current_dp = [float(model_config["data_proportion"])]  # mutable holder

    def _rebuild_for_dp(dp_value, state_unreplicated, rng, ema_fn, lr_fn, aux_fn, model_config):
        """Rebuild model + p_train_step for a new data_proportion, keeping params."""
        mc = dict(model_config)
        mc["data_proportion"] = dp_value
        new_model = pixelMeanFlow(**mc)
        new_apply_fn = partial(new_model.apply, method=new_model.forward)
        new_state = state_unreplicated.replace(apply_fn=new_apply_fn)
        new_p_train_step = jax.pmap(
            partial(
                train_step,
                rng_init=rng,
                ema_fn=ema_fn,
                lr_fn=lr_fn,
                aux_fn=aux_fn,
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
        _current_dp[0] = dp_value
        log_for_0("Rebuilt p_train_step with data_proportion=%.3f", dp_value)
        return new_model, new_state, new_p_train_step

    ########### Create Train State ###########
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)

    # Support resume via env var PMF_LOAD_FROM (set by shell script)
    load_from = config.load_from
    if not load_from:
        load_from = os.environ.get("PMF_LOAD_FROM", "")
    log_for_0("load_from resolved to: '%s' (config='%s', env='%s')",
              load_from, config.load_from, os.environ.get("PMF_LOAD_FROM", ""))
    if load_from:
        state = restore_checkpoint(state, load_from)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch

    # If resuming and dp_schedule is active, apply the correct data_proportion
    if dp_schedule:
        needed_dp = _get_scheduled_dp(epoch_offset)
        if needed_dp != _current_dp[0]:
            log_for_0("Resume at epoch %d: switching data_proportion %.3f -> %.3f",
                       epoch_offset, _current_dp[0], needed_dp)
            mc = dict(model_config)
            mc["data_proportion"] = needed_dp
            model = pixelMeanFlow(**mc)
            state = state.replace(apply_fn=partial(model.apply, method=model.forward))
            _current_dp[0] = needed_dp

    state = jax_utils.replicate(state)

    if config.model.convnext or config.model.lpips:
        log_for_0(f"Using perceptual auxiliary loss")
        aux_fn = None
    else:
        log_for_0("Not using perceptual auxiliary loss")
        aux_fn = None

    ########### Create train and sample pmap ###########
    p_process_batch = jax.pmap(
        partial(
            input_pipeline.process_batch_on_tpu,
            use_flip=use_flip,
            pin_levels_to_device_groups=pin_levels_to_device_groups,
            num_levels=num_levels,
            devices_per_level=devices_per_level,
        ),
        axis_name="batch",
    )

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            aux_fn=aux_fn,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    do_sampling = int(config.training.sample_per_epoch) > 0
    fid_cache_ref = str(config.fid.get("cache_ref", ""))
    do_fid = int(config.training.fid_per_epoch) > 0 and fid_cache_ref != "" and os.path.exists(fid_cache_ref)
    latent_save_every = int(config.training.get("latent_save_per_epoch", 0))
    latent_save_num = int(config.training.get("latent_save_num_samples", 5))
    latent_save_png = bool(config.training.get("latent_save_png", True))
    latent_save_npz = bool(config.training.get("latent_save_npz", False))
    latent_save_gap = int(config.training.get("latent_save_png_gap", 1))
    do_latent_save = is_latent_kind and latent_save_every > 0 and latent_save_num > 0
    train_input_vis_num = int(config.training.get("train_input_vis_num_samples_per_level", 0))
    train_input_vis_gap = int(config.training.get("train_input_vis_gap", 1))
    train_input_vis_include_prev = bool(config.training.get("train_input_vis_include_prev", True))
    do_train_input_vis = is_latent_kind and train_input_vis_num > 0 and (jax.process_index() == 0)
    if do_train_input_vis:
        train_input_vis_dir = os.path.join(workdir, "train_input_vis")
        train_input_vis_counts = np.zeros(int(config.dataset.get("num_levels", 1)), dtype=np.int32)
        log_for_0(
            "Training-input latent visualization enabled: saving up to %d samples per level to %s%s",
            train_input_vis_num,
            train_input_vis_dir,
            " (level>0 PNG rows are [prev, current])" if train_input_vis_include_prev else "",
        )
    else:
        train_input_vis_dir = None
        train_input_vis_counts = None

    if int(config.training.fid_per_epoch) > 0 and not do_fid:
        log_for_0(
            "FID is disabled: set training.fid_per_epoch > 0 and a valid fid.cache_ref path to enable."
        )

    use_cfg = bool(config.model.get("use_cfg", True))

    if do_sampling or do_fid:
        p_sample_step = jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(99),
                config=config,
                device_batch_size=device_bsz,
                num_steps=config.sampling.num_steps,
            ),
            axis_name="batch",
        )

        vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
            jax.local_device_count()
        )
        if use_cfg:
            sample_kwargs = {
                "omega": config.sampling.omega,
                "t_min": config.sampling.t_min,
                "t_max": config.sampling.t_max,
            }
            sample_kwargs = jax_utils.replicate(sample_kwargs)
        else:
            sample_kwargs = {}

        timer = Timer()
        log_for_0("Compiling sample step...")
        _ = p_sample_step.lower(
            {"params": state.params},
            sample_idx=vis_sample_idx,
            **sample_kwargs,
        ).compile()
        log_for_0(f"Sampling step compiled in {timer}")
    else:
        p_sample_step = None
        vis_sample_idx = None
        sample_kwargs = None

    if do_latent_save:
        latent_per_device = max(1, math.ceil(latent_save_num / jax.local_device_count()))
        p_latent_sample_step = jax.pmap(
            partial(
                latent_sample_step,
                model=model,
                rng_init=random.PRNGKey(12345),
                config=config,
                device_batch_size=latent_per_device,
                num_steps=config.sampling.num_steps,
            ),
            axis_name="batch",
        )
        if use_cfg:
            latent_sample_kwargs = jax_utils.replicate(
                {
                    "omega": config.sampling.omega,
                    "t_min": config.sampling.t_min,
                    "t_max": config.sampling.t_max,
                }
            )
        else:
            latent_sample_kwargs = {}
        latent_sample_idx = (
            jax.process_index() * jax.local_device_count()
            + jnp.arange(jax.local_device_count(), dtype=jnp.int32)
        )
        log_for_0(
            "Latent snapshot saving enabled: every %d epoch(s), %d samples.",
            latent_save_every,
            latent_save_num,
        )
    else:
        p_latent_sample_step = None
        latent_sample_kwargs = None
        latent_sample_idx = None

    fid_evaluator = get_fid_evaluator(config, writer) if do_fid else None

    ########### Training Loop ###########
    profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
    profile_mode = profile_steps > 0
    if profile_mode:
        log_for_0("=" * 60)
        log_for_0("PROFILING MODE: will run %d steps after compilation", profile_steps)
        log_for_0("=" * 60)
    profile_count = 0  # steps after compilation
    compilation_done = False

    metrics_tracker = MetricsTracker()
    for epoch in range(epoch_offset, config.training.num_epochs):
        # --- check data_proportion schedule ---
        if dp_schedule:
            needed_dp = _get_scheduled_dp(epoch)
            if needed_dp != _current_dp[0]:
                log_for_0("Epoch %d: switching data_proportion %.3f -> %.3f",
                           epoch, _current_dp[0], needed_dp)
                state = jax_utils.unreplicate(state)
                model, state, p_train_step = _rebuild_for_dp(
                    needed_dp, state, rng, ema_fn, lr_fn, aux_fn, model_config,
                )
                state = jax_utils.replicate(state)

        if jax.process_count() > 1:
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            else:
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        ########### Sampling ###########
        if do_sampling and (epoch + 1) % config.training.sample_per_epoch == 0:
            log_for_0(f"Samples at epoch {epoch}...")
            vis_sample = run_p_sample_step(
                p_sample_step,
                state,
                vis_sample_idx,
                ema=None,
                config=config,
                **sample_kwargs,
            )
            vis_sample = make_grid_visualization(vis_sample, grid=4)
            vis_sample = jax.device_get(vis_sample)[0]
            writer.write_images(step + 1, {"vis_sample": vis_sample})

        ########### Train ###########
        timer = Timer()
        log_for_0("epoch {}...".format(epoch))
        timer.reset()
        io_wait_acc = 0.0
        host_prep_acc = 0.0
        train_step_acc = 0.0
        dev_proc_acc = 0.0
        metrics_sync_acc = 0.0
        wait_t0 = time.perf_counter()
        for n_batch, batch in enumerate(train_loader):
            t_data = time.perf_counter() - wait_t0
            io_wait_acc += t_data
            step = epoch * steps_per_epoch + n_batch

            # Prepare batch (just reshaping, still uint8)
            prep_t0 = time.perf_counter()
            batch = input_pipeline.prepare_batch_data(batch)

            # Generate RNG keys for random flip
            rng_flip = random.fold_in(rng, step)
            rng_flip_split = random.split(rng_flip, jax.local_device_count())
            t_host_prep = time.perf_counter() - prep_t0

            # Process images on device (latent reconstruction)
            dev_t0 = time.perf_counter()
            batch = p_process_batch(batch, rng_key=rng_flip_split)
            if profile_mode and compilation_done:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), batch)
            t_dev_proc = time.perf_counter() - dev_t0

            host_prep_acc += t_host_prep + t_dev_proc
            dev_proc_acc += t_dev_proc

            # Optional one-time debug dump of the exact latent tensors fed to the model.
            if do_train_input_vis:
                train_input_vis_done = maybe_save_train_input_latent_visuals(
                    batch=batch,
                    out_dir=train_input_vis_dir,
                    per_level_counts=train_input_vis_counts,
                    max_per_level=train_input_vis_num,
                    gap=train_input_vis_gap,
                    include_prev=train_input_vis_include_prev,
                    step=step,
                )
                if train_input_vis_done:
                    do_train_input_vis = False
                    log_for_0(
                        "Completed training-input latent visualization dump: %s",
                        train_input_vis_dir,
                    )

            # one train step
            train_t0 = time.perf_counter()
            state, metrics = p_train_step(state, batch)
            if profile_mode and compilation_done:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            t_train = time.perf_counter() - train_t0
            train_step_acc += t_train

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                compilation_time = timer.elapse_with_reset()
                log_for_0("p_train_step compiled in {:.2f}s".format(compilation_time))
                io_wait_acc = 0.0
                host_prep_acc = 0.0
                train_step_acc = 0.0
                dev_proc_acc = 0.0
                metrics_sync_acc = 0.0
                compilation_done = True

            ########### Metrics ###########
            metrics_t0 = time.perf_counter()
            metrics_tracker.update(metrics)  # stream one step in
            t_metrics = time.perf_counter() - metrics_t0
            metrics_sync_acc += t_metrics

            # --- Per-step profiling output ---
            if profile_mode and compilation_done:
                profile_count += 1
                t_total = t_data + t_host_prep + t_dev_proc + t_train + t_metrics
                log_for_0(
                    "PROFILE step=%d | data=%.4f prep=%.4f dev_proc=%.4f train=%.4f metrics=%.4f | total=%.4f s/step",
                    step, t_data, t_host_prep, t_dev_proc, t_train, t_metrics, t_total,
                )
                if profile_count >= profile_steps:
                    log_for_0("=" * 60)
                    log_for_0("PROFILING SUMMARY (%d steps after compilation):", profile_count)
                    log_for_0("  avg data_wait:     %.4f s/step", io_wait_acc / profile_count)
                    log_for_0("  avg host_prep:     %.4f s/step", (host_prep_acc - dev_proc_acc) / profile_count)
                    log_for_0("  avg dev_process:   %.4f s/step", dev_proc_acc / profile_count)
                    log_for_0("  avg train_step:    %.4f s/step", train_step_acc / profile_count)
                    log_for_0("  avg metrics_sync:  %.4f s/step", metrics_sync_acc / profile_count)
                    total_avg = (io_wait_acc + host_prep_acc + train_step_acc + metrics_sync_acc) / profile_count
                    log_for_0("  avg total:         %.4f s/step", total_avg)
                    log_for_0("  throughput:        %.2f steps/s", 1.0 / total_avg if total_avg > 0 else 0)
                    log_for_0("=" * 60)
                    return state

            if (step + 1) % config.training.log_per_step == 0:
                summary = metrics_tracker.finalize()
                log_steps = config.training.log_per_step
                summary["steps_per_second"] = log_steps / timer.elapse_with_reset()
                summary["data_wait_s_per_step"] = io_wait_acc / log_steps
                summary["host_prep_s_per_step"] = host_prep_acc / log_steps
                summary["train_step_s_per_step"] = train_step_acc / log_steps
                summary["dev_proc_s_per_step"] = dev_proc_acc / log_steps
                summary["metrics_sync_s_per_step"] = metrics_sync_acc / log_steps
                summary["epoch"] = epoch
                summary["data_proportion"] = _current_dp[0]
                writer.write_scalars(step + 1, summary)
                io_wait_acc = 0.0
                host_prep_acc = 0.0
                train_step_acc = 0.0
                dev_proc_acc = 0.0
                metrics_sync_acc = 0.0

            ########### Per-step walltime guard ###########
            if job_end_time and time.time() > job_end_time - save_before_end:
                log_for_0("Walltime approaching mid-epoch — saving checkpoint at step %d (epoch %d, batch %d/%d).",
                           step, epoch, n_batch + 1, steps_per_epoch)
                save_checkpoint(
                    state,
                    workdir,
                    keep=config.training.get("checkpoint_keep", None),
                    backend=config.training.get("checkpoint_backend", None),
                )
                log_for_0("Exiting cleanly for resubmit.")
                jax.random.normal(jax.random.key(0), ()).block_until_ready()
                return state

            wait_t0 = time.perf_counter()

        ########### Save Checkpoint ###########
        if (epoch + 1) % config.training.checkpoint_per_epoch == 0 \
            or (epoch + 1) == config.training.num_epochs:
            save_checkpoint(
                state,
                workdir,
                keep=config.training.get("checkpoint_keep", None),
                backend=config.training.get("checkpoint_backend", None),
            )

        ########### Walltime guard ###########
        if job_end_time and time.time() > job_end_time - save_before_end:
            # Force-save if not already saved by the regular checkpoint logic above
            already_saved = (
                (epoch + 1) % config.training.checkpoint_per_epoch == 0
                or (epoch + 1) == config.training.num_epochs
            )
            if not already_saved:
                log_for_0("Walltime approaching — saving emergency checkpoint at epoch %d.", epoch)
                save_checkpoint(
                    state,
                    workdir,
                    keep=config.training.get("checkpoint_keep", None),
                    backend=config.training.get("checkpoint_backend", None),
                )
            log_for_0("Walltime approaching (epoch %d/%d done). Exiting cleanly for resubmit.",
                       epoch + 1, config.training.num_epochs)
            jax.random.normal(jax.random.key(0), ()).block_until_ready()
            return state

        ########### FID ###########
        if do_fid and (
            (epoch + 1) % config.training.fid_per_epoch == 0
            or (epoch + 1) == config.training.num_epochs
        ):
            fid_evaluator(state, p_sample_step, step, **sample_kwargs)

        if do_latent_save and ((epoch + 1) % latent_save_every == 0):
            save_on_rank0 = jax.process_index() == 0
            if save_on_rank0:
                use_ema = bool(config.training.get("latent_save_use_ema", False))
                if use_ema and len(state.ema_params) > 0:
                    requested_ema = config.training.get("latent_save_ema", None)
                    ema_keys = sorted(float(k) for k in state.ema_params.keys())
                    if requested_ema is None:
                        chosen_ema = ema_keys[-1]
                    else:
                        chosen_ema = min(ema_keys, key=lambda k: abs(k - float(requested_ema)))
                    params_for_save = state.ema_params[chosen_ema]
                else:
                    chosen_ema = None
                    params_for_save = state.params

                variable = {"params": params_for_save}
                epoch_sample_idx = latent_sample_idx + (epoch + 1) * jax.device_count()
                latents = p_latent_sample_step(
                    variable,
                    sample_idx=epoch_sample_idx,
                    **latent_sample_kwargs,
                )
                latents = latents.reshape(-1, *latents.shape[2:])
                latents = jax.device_get(latents)[:latent_save_num]

                out_dir = os.path.join(workdir, "latent_samples", f"epoch_{epoch + 1:04d}")
                os.makedirs(out_dir, exist_ok=True)
                for i, arr in enumerate(latents):
                    arr_np = np.asarray(arr, dtype=np.float32)  # (L,H,W,C)
                    if latent_save_npz:
                        np.savez_compressed(
                            os.path.join(out_dir, f"sample_{i:03d}.npz"),
                            latents=arr_np,
                        )
                    if latent_save_png:
                        strip = latent_levels_to_pca_column(
                            arr_np,
                            gap=latent_save_gap,
                            bg_value=0,
                        )
                        Image.fromarray(strip).save(
                            os.path.join(out_dir, f"sample_{i:03d}.png")
                        )
                if chosen_ema is None:
                    log_for_0("Saved %d latent samples to %s", len(latents), out_dir)
                else:
                    log_for_0(
                        "Saved %d latent samples to %s (ema=%s)",
                        len(latents),
                        out_dir,
                        chosen_ema,
                    )
    
    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state


########################################################
#                    Evaluation                        #
########################################################

def _discover_checkpoint_sequence(load_from_folder, start_step, step_size):
    """Discover checkpoint_<step> entries in a folder at regular intervals."""
    from pathlib import Path
    root = Path(load_from_folder)
    found = []
    step = int(start_step)
    while True:
        candidate = root / f"checkpoint_{step}"
        if candidate.exists():
            found.append((step, str(candidate)))
        else:
            break
        step += int(step_size)
    return found


def _evaluate_single_checkpoint(config, state_template, model, fid_evaluator, p_sample_step, checkpoint_path):
    """Load a checkpoint and evaluate FID, returning the best result dict."""
    state = restore_checkpoint(state_template, checkpoint_path, params_only=True)

    step = int(state.step)
    state = jax_utils.replicate(state)

    use_cfg = bool(config.model.get("use_cfg", True))

    best_fid = float("inf")
    best_is = float("-inf")
    best_config = None
    for ema in config.sampling.emas:
        if use_cfg:
            for interval in config.sampling.interval:
                t_min, t_max = interval
                for omega in config.sampling.omegas:
                    kwargs = {"omega": omega, "t_min": t_min, "t_max": t_max, "ema": ema}
                    kwargs = jax_utils.replicate(kwargs)
                    fid, is_score = fid_evaluator(state, p_sample_step, step, True, **kwargs)

                    if fid < best_fid:
                        best_fid, best_is, best_config = fid, is_score, (omega, t_min, t_max, ema)
        else:
            kwargs = {"ema": ema}
            kwargs = jax_utils.replicate(kwargs)
            fid, is_score = fid_evaluator(state, p_sample_step, step, True, **kwargs)
            omega, t_min, t_max = 1.0, 0.0, 1.0
            if fid < best_fid:
                best_fid, best_is, best_config = fid, is_score, (omega, t_min, t_max, ema)

    omega, t_min, t_max, ema = best_config
    log_for_0(
        f"best_fid={best_fid:.4f} best_is={best_is:.4f} "
        f"omega={omega} t_min={t_min} t_max={t_max} ema={ema}"
    )
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": _checkpoint_step_from_path(checkpoint_path),
        "restored_step": step,
        "best_fid": best_fid,
        "best_is": best_is,
        "omega": omega,
        "t_min": t_min,
        "t_max": t_max,
        "ema": ema,
    }


def _checkpoint_step_from_path(path):
    """Extract step number from checkpoint_<step> path."""
    from pathlib import Path
    name = Path(path).name
    prefix = "checkpoint_"
    if name.startswith(prefix):
        step_str = name[len(prefix):]
        if step_str.isdigit():
            return int(step_str)
    return None


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:

    assert config.eval_only, "config.eval_only must be True for just_evaluate"

    has_folder = bool(str(config.get("load_from_folder", "")).strip())
    has_single = bool(str(config.get("load_from", "")).strip())

    if not has_folder:
        assert has_single, "config.load_from or config.load_from_folder must be specified"

    ########### Initialize ###########
    writer = Writer(config, workdir)

    rng = random.key(0)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size
    config.training.ema_val = config.sampling.emas
    lr_fn = lr_schedules(config, 1000)  # dummy steps_per_epoch

    ########### Create Model ###########
    model_config = config.model.to_dict()
    model_config["num_classes"] = int(config.dataset.num_classes)
    model_config["input_size"] = int(config.dataset.image_size)
    model_config["in_channels"] = int(config.dataset.image_channels)
    model_config["num_levels"] = int(config.dataset.get("num_levels", 4))
    if str(config.dataset.get("kind", "imagenet")).lower() in {"latent_hier", "latent"}:
        model_config["use_token_embed"] = True
        model_config["use_prev_cond"] = True
        model_config["use_level_cond"] = True
    model_config.pop("data_proportion_schedule", None)
    model = pixelMeanFlow(**model_config, eval=True)

    ########### Create Train State (template) ###########
    state_template = create_train_state(rng, config, model, image_size, lr_fn)

    ########### Create sample pmap ###########

    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=random.PRNGKey(99),
            config=config,
            device_batch_size=device_bsz,
            num_steps=config.sampling.num_steps,
        ),
        axis_name="batch",
    )

    fid_evaluator = get_fid_evaluator(config, writer)

    ############ Folder sweep mode ###########
    if has_single: 
        checkpoint_path = str(config.get("load_from", "")).strip()
        
    if has_folder:
        import json
        from pathlib import Path

        load_from_folder = str(config.load_from_folder)
        start_step = int(config.get("load_from_start_step", 6255))
        step_size = int(config.get("load_from_step", 6255))
        save_json_path = str(config.get("save_json", ""))

        ckpt_seq = _discover_checkpoint_sequence(load_from_folder, start_step, step_size)
        log_for_0(
            "Discovered %d checkpoints in %s (start=%d, step=%d)",
            len(ckpt_seq), load_from_folder, start_step, step_size,
        )

        # Load existing results to skip already-evaluated checkpoints
        existing_results = []
        already_evaluated = set()
        if save_json_path:
            json_path = Path(save_json_path)
            if json_path.exists():
                try:
                    with json_path.open("r") as f:
                        prev = json.load(f)
                    if isinstance(prev, dict) and "results" in prev:
                        for r in prev["results"]:
                            existing_results.append(r)
                            if r.get("checkpoint_step") is not None:
                                already_evaluated.add(int(r["checkpoint_step"]))
                    log_for_0(
                        "Loaded %d existing results from %s, will skip steps: %s",
                        len(existing_results), json_path, sorted(already_evaluated),
                    )
                except Exception as e:
                    log_for_0("Warning: could not load existing results from %s: %s", json_path, e)

        all_results = list(existing_results)
        num_skipped = 0
        for ckpt_step, ckpt_path in ckpt_seq:
            if ckpt_step in already_evaluated:
                log_for_0("Skipping checkpoint %s (step %d) — already evaluated", ckpt_path, ckpt_step)
                num_skipped += 1
                continue

            log_for_0("Evaluating checkpoint: %s (step %d)", ckpt_path, ckpt_step)
            result = _evaluate_single_checkpoint(
                config, state_template, model, fid_evaluator, p_sample_step, ckpt_path,
            )
            all_results.append(result)

            # Save intermediate results after each checkpoint
            if save_json_path and jax.process_index() == 0:
                aggregate = {
                    "mode": "eval_only_checkpoint_sweep",
                    "load_from_folder": load_from_folder,
                    "load_from_start_step": start_step,
                    "load_from_step": step_size,
                    "num_checkpoints_discovered": len(ckpt_seq),
                    "num_checkpoints_with_fid": len(all_results),
                    "results": all_results,
                }
                out_path = Path(save_json_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(aggregate, indent=2) + "\n")
                log_for_0("Saved intermediate sweep results to %s", out_path)

        log_for_0("Sweep complete: %d evaluated, %d skipped (already done)", len(all_results) - len(existing_results), num_skipped)
        jax.random.normal(jax.random.key(0), ()).block_until_ready()
        return state_template

    ############ Single checkpoint mode ###########
    state = restore_checkpoint(state_template, checkpoint_path, params_only=True)

    step = int(state.step)
    state = jax_utils.replicate(state)

    best_fid = float("inf")
    best_is = float("-inf")
    best_config = None
    for ema in config.sampling.emas:
        for interval in config.sampling.interval:
            t_min, t_max = interval
            for omega in config.sampling.omegas:
                kwargs = {"omega": omega, "t_min": t_min, "t_max": t_max, "ema": ema}
                kwargs = jax_utils.replicate(kwargs)
                fid, is_score = fid_evaluator(state, p_sample_step, step, True, **kwargs)

                if fid < best_fid:
                    best_fid, best_is, best_config = fid, is_score, (omega, t_min, t_max, ema)

    omega, t_min, t_max, ema = best_config
    summary = {'best_fid': best_fid, 'best_is': best_is, 'omega': omega, 't_min': t_min, 't_max': t_max, 'ema': ema}
    log_for_0(
        f"Best FID achieved: {best_fid:.2f}, \n"
        f"IS achieved: {best_is:.2f}, \n"
        f"omega: {omega:.2f}, t_min: {t_min:.2f}, t_max: {t_max:.2f}, ema: {ema}"
    )
    writer.write_scalars(step + 1, summary)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    return state


def just_evaluate_autoguidance(config: ml_collections.ConfigDict, workdir: str):
    """Evaluate using AutoGuidance: good checkpoint guided by early (bad) checkpoint."""
    import json
    from pathlib import Path
    from utils import fid_util

    assert config.eval_only, "config.eval_only must be True"

    writer = Writer(config, workdir)
    rng = random.key(0)
    image_size = config.dataset.image_size
    device_bsz = config.fid.device_batch_size
    config.training.ema_val = config.sampling.emas
    lr_fn = lr_schedules(config, 1000)

    # Create model
    model_config = config.model.to_dict()
    model_config["num_classes"] = int(config.dataset.num_classes)
    model_config["input_size"] = int(config.dataset.image_size)
    model_config["in_channels"] = int(config.dataset.image_channels)
    model_config["num_levels"] = int(config.dataset.get("num_levels", 4))
    if str(config.dataset.get("kind", "imagenet")).lower() in {"latent_hier", "latent"}:
        model_config["use_token_embed"] = True
        model_config["use_prev_cond"] = True
        model_config["use_level_cond"] = True
    model_config.pop("data_proportion_schedule", None)
    model = pixelMeanFlow(**model_config, eval=True)

    state_template = create_train_state(rng, config, model, image_size, lr_fn)

    # Load bad (early) checkpoint
    bad_ckpt_path = str(config.autoguidance.bad_checkpoint)
    log_for_0("Loading bad (degraded) checkpoint: %s", bad_ckpt_path)
    state_bad = restore_checkpoint(state_template, bad_ckpt_path, params_only=True)

    guidance_scales = list(config.autoguidance.guidance_scales)
    log_for_0("AutoGuidance scales to sweep: %s", guidance_scales)

    # Discover good checkpoints
    load_from_folder = str(config.get("load_from_folder", "")).strip()
    load_from_single = str(config.get("load_from", "")).strip()
    save_json_path = str(config.get("save_json", ""))

    if load_from_folder:
        start_step = int(config.get("load_from_start_step", 6255))
        step_size = int(config.get("load_from_step", 6255))
        ckpt_seq = _discover_checkpoint_sequence(load_from_folder, start_step, step_size)
        log_for_0("Discovered %d good checkpoints in %s", len(ckpt_seq), load_from_folder)
    elif load_from_single:
        ckpt_step = _checkpoint_step_from_path(load_from_single)
        ckpt_seq = [(ckpt_step, load_from_single)]
        log_for_0("Single good checkpoint: %s (step %s)", load_from_single, ckpt_step)
    else:
        raise ValueError("Either load_from or load_from_folder must be specified")

    # Load existing results to skip already-evaluated
    existing_results = []
    already_evaluated = set()
    if save_json_path:
        json_path = Path(save_json_path)
        if json_path.exists():
            try:
                with json_path.open("r") as f:
                    prev_data = json.load(f)
                if isinstance(prev_data, dict) and "results" in prev_data:
                    for r in prev_data["results"]:
                        existing_results.append(r)
                        key = (r.get("checkpoint_step"), r.get("guidance_scale"))
                        if key[0] is not None:
                            already_evaluated.add(key)
                log_for_0("Loaded %d existing results", len(existing_results))
            except Exception as e:
                log_for_0("Warning: could not load existing results: %s", e)

    # Build FID infrastructure
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref)

    is_latent = str(config.dataset.get("kind", "imagenet")).lower() in {"latent_hier", "latent"}
    num_fid_steps = int(np.ceil(
        config.fid.num_samples / device_bsz / jax.device_count()
    ))

    all_results = list(existing_results)

    for ckpt_step, ckpt_path in ckpt_seq:
        log_for_0("Loading good checkpoint: %s (step %d)", ckpt_path, ckpt_step)
        state_good = restore_checkpoint(state_template, ckpt_path, params_only=True)
        step = int(state_good.step)

        for guidance_scale in guidance_scales:
            # Build pmap'd sample function with guidance_scale and CFG params baked in
            ag_omega = float(config.autoguidance.get("omega", 1.0))
            ag_t_min = float(config.autoguidance.get("t_min", 0.0))
            ag_t_max = float(config.autoguidance.get("t_max", 1.0))
            p_ag_sample = jax.pmap(
                partial(
                    autoguidance_sample_step,
                    model=model,
                    rng_init=random.PRNGKey(99),
                    device_batch_size=device_bsz,
                    config=config,
                    num_steps=config.sampling.num_steps,
                    guidance_scale=guidance_scale,
                    omega=ag_omega,
                    t_min=ag_t_min,
                    t_max=ag_t_max,
                ),
                axis_name="batch",
            )

            for ema in config.sampling.emas:
                if (ckpt_step, guidance_scale) in already_evaluated:
                    log_for_0("Skipping step %d, scale %.2f — already evaluated", ckpt_step, guidance_scale)
                    continue

                log_for_0("Evaluating step=%d, guidance_scale=%.2f, ema=%s", ckpt_step, guidance_scale, ema)

                good_params = state_good.ema_params[ema] if ema is not None else state_good.params
                bad_params = state_bad.ema_params.get(ema, state_bad.params) if ema is not None else state_bad.params

                var_good = jax_utils.replicate({"params": good_params})
                var_bad = jax_utils.replicate({"params": bad_params})

                # Generate samples
                samples_all = []
                for fid_step in range(num_fid_steps):
                    sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
                        jax.local_device_count()
                    )
                    sample_idx = jax.device_count() * fid_step + sample_idx
                    log_for_0(f"  Sampling step {fid_step}/{num_fid_steps}...")

                    samples = p_ag_sample(
                        var_good, var_bad,
                        sample_idx=sample_idx,
                    )
                    samples = samples.reshape(-1, *samples.shape[2:])

                    assert not jnp.any(jnp.isnan(samples)), "NaN in samples!"

                    if is_latent:
                        from utils import rae_decoder as rae_decoder_util
                        batch_decode = int(config.rae_decoder.get("batch_size", 16))
                        jax.random.normal(random.key(0), ()).block_until_ready()
                        samples = np.asarray(jax.device_get(samples), dtype=np.float32)
                        samples = rae_decoder_util.decode_bchw_to_uint8(samples, config, batch_size=batch_decode)
                    else:
                        samples = samples.transpose(0, 2, 3, 1)
                        samples = 127.5 * samples + 128.0
                        samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)
                        jax.random.normal(random.key(0), ()).block_until_ready()

                    samples_all.append(jax.device_get(samples) if not isinstance(samples, np.ndarray) else samples)

                samples_all = np.concatenate(samples_all, axis=0)

                # Compute FID
                stats = fid_util.compute_stats(samples_all, inception_net)
                fid = fid_util.compute_fid(
                    stats_ref["mu"], stats["mu"], stats_ref["sigma"], stats["sigma"]
                )
                is_score, _ = fid_util.compute_inception_score(stats["logits"])

                result = {
                    "checkpoint_path": ckpt_path,
                    "checkpoint_step": ckpt_step,
                    "restored_step": step,
                    "best_fid": float(fid),
                    "best_is": float(is_score),
                    "guidance_scale": guidance_scale,
                    "bad_checkpoint": bad_ckpt_path,
                    "ema": ema,
                }
                all_results.append(result)
                log_for_0(
                    "FID=%.4f IS=%.4f (step=%d, scale=%.2f, ema=%s)",
                    fid, is_score, ckpt_step, guidance_scale, ema,
                )

                # Save intermediate
                if save_json_path and jax.process_index() == 0:
                    aggregate = {
                        "mode": "autoguidance_sweep",
                        "load_from_folder": load_from_folder,
                        "bad_checkpoint": bad_ckpt_path,
                        "results": all_results,
                    }
                    Path(save_json_path).write_text(json.dumps(aggregate, indent=2) + "\n")

    log_for_0("AutoGuidance sweep complete: %d results", len(all_results))
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state_template
