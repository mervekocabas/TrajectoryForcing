import os

import jax
from flax.training import checkpoints
from flax import serialization

from utils.logging_util import log_for_0


_ORBAX_DISABLE_ATTEMPTED = False
_KEEP_ALL_SENTINEL = 2**31 - 1


def _disable_flax_orbax_backend_if_possible():
    """Best-effort switch to legacy Flax checkpoint backend."""
    global _ORBAX_DISABLE_ATTEMPTED
    if _ORBAX_DISABLE_ATTEMPTED:
        return
    _ORBAX_DISABLE_ATTEMPTED = True
    try:
        from flax import config as flax_config

        flax_config.update("flax_use_orbax_checkpointing", False)
        log_for_0("Disabled Flax Orbax checkpoint backend (legacy checkpoints enabled).")
    except Exception as e:
        log_for_0("Could not disable Flax Orbax checkpoint backend automatically: %s", e)


def _looks_like_orbax_locking_error(exc: Exception) -> bool:
    msg = str(exc)
    needles = (
        "Failed to lock file",
        "ENOSYS",
        "orbax",
        "tensorstore",
        "ocdbt",
    )
    return any(n in msg for n in needles)


def _get_checkpoint_keep_count() -> int:
    """Returns checkpoint retention count (supports keep-all via env var)."""
    raw = str(os.environ.get("PMF_CKPT_KEEP", "3")).strip().lower()
    if raw in {"all", "none", "inf", "infinite", "unlimited"}:
        return _KEEP_ALL_SENTINEL
    try:
        keep = int(raw)
    except ValueError:
        log_for_0("Invalid PMF_CKPT_KEEP=%r, defaulting to 3.", raw)
        return 3
    if keep <= 0:
        return _KEEP_ALL_SENTINEL
    return keep


def _normalize_checkpoint_keep(keep) -> int:
    """Normalize keep count from code/config values (supports keep-all)."""
    if keep is None:
        return _get_checkpoint_keep_count()
    if isinstance(keep, str):
        raw = keep.strip().lower()
        if raw in {"all", "none", "inf", "infinite", "unlimited"}:
            return _KEEP_ALL_SENTINEL
        try:
            keep = int(raw)
        except ValueError:
            raise ValueError(f"Invalid checkpoint keep value: {keep!r}") from None
    keep = int(keep)
    if keep <= 0:
        return _KEEP_ALL_SENTINEL
    return keep


def _save_checkpoint_legacy_single_writer(workdir, state, step, keep=None):
    """Legacy Flax checkpoint save from process 0 only."""
    if jax.process_count() > 1 and jax.process_index() != 0:
        return
    # Best-effort: disable Orbax-backed implementation in newer Flax.
    _disable_flax_orbax_backend_if_possible()
    checkpoints.save_checkpoint(workdir, state, step, keep=_normalize_checkpoint_keep(keep))


def _sanitize_source_to_target(target, source, path=""):
    """Recursively match `source` to `target`'s structure for lenient restore.

    - Keys present in target but missing in source -> filled from the initialized
      target (via `to_state_dict`), so downstream `from_state_dict` keeps init value.
    - Keys present in source but missing in target -> dropped with a log message.
    - Leaves are passed through unchanged.

    Returns a plain dict shaped like target, with values drawn from source where
    available.
    """
    # Treat anything dict-like (plain dict or FrozenDict) as a subtree.
    target_is_map = hasattr(target, "keys") and hasattr(target, "__getitem__") and not isinstance(
        target, (str, bytes)
    )
    if not target_is_map:
        return source  # leaf: use checkpoint value as-is

    source_map = source if isinstance(source, dict) else {}
    result = {}
    target_keys = list(target.keys())
    filled_missing = []
    for key in target_keys:
        sub_target = target[key]
        sub_path = f"{path}/{key}" if path else str(key)
        if key in source_map:
            result[key] = _sanitize_source_to_target(sub_target, source_map[key], sub_path)
        else:
            # Use the already-initialized target as a state dict so
            # serialization.from_state_dict keeps the init value.
            result[key] = serialization.to_state_dict(sub_target)
            filled_missing.append(sub_path)

    extras = [k for k in source_map.keys() if k not in target]
    if filled_missing:
        log_for_0(
            "Lenient restore: filled %d missing key(s) from init at %s (examples: %s)",
            len(filled_missing),
            path or "<root>",
            ", ".join(filled_missing[:5]),
        )
    if extras:
        log_for_0(
            "Lenient restore: dropping %d unexpected checkpoint key(s) at %s (examples: %s)",
            len(extras),
            path or "<root>",
            ", ".join(str(k) for k in extras[:5]),
        )
    return result


def _restore_params_only_from_raw(state, raw_state, workdir, allow_missing=False):
    """Restore only step/params/ema_params from a raw checkpoint payload.

    If ``allow_missing`` is True, any parameters present in the initialized
    ``state`` but absent from the checkpoint will keep their initialized values
    (instead of raising). Extra checkpoint keys are dropped with a warning.
    """
    if raw_state is None:
        raise ValueError(f"No checkpoint found at {workdir}")

    if not isinstance(raw_state, dict):
        raise TypeError(
            f"Expected raw checkpoint payload to be dict-like, got {type(raw_state)!r}"
        )

    new_state = state
    restored_fields = []

    if "step" in raw_state:
        step_val = serialization.from_state_dict(state.step, raw_state["step"])
        new_state = new_state.replace(step=step_val)
        restored_fields.append("step")

    if "params" in raw_state:
        src = raw_state["params"]
        if allow_missing:
            src = _sanitize_source_to_target(state.params, src, path="params")
        params = serialization.from_state_dict(state.params, src)
        new_state = new_state.replace(params=params)
        restored_fields.append("params")

    if "ema_params" in raw_state:
        src = raw_state["ema_params"]
        if allow_missing:
            src = _sanitize_source_to_target(state.ema_params, src, path="ema_params")
        ema_params = serialization.from_state_dict(state.ema_params, src)
        new_state = new_state.replace(ema_params=ema_params)
        restored_fields.append("ema_params")

    if not restored_fields:
        raise KeyError(
            "Checkpoint payload does not contain any of: step, params, ema_params"
        )

    log_for_0(
        "Partially restored checkpoint (params-only mode%s) from %s with fields: %s",
        ", lenient" if allow_missing else "",
        workdir,
        ", ".join(restored_fields),
    )
    return new_state


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def restore_checkpoint(state, workdir, params_only=False, allow_missing=None):
    """
    Restores the model state from a checkpoint located in the specified working directory.

    When ``params_only=True`` and ``allow_missing`` is True (or the env var
    ``PMF_CKPT_ALLOW_MISSING`` is set), parameters present in the initialized
    ``state`` but absent from the checkpoint keep their initialized values.
    This lets newer model architectures (with extra modules) evaluate older
    checkpoints that don't yet contain those parameters.
    """
    if allow_missing is None:
        allow_missing = _env_flag("PMF_CKPT_ALLOW_MISSING", default=False)

    if not params_only:
        state = checkpoints.restore_checkpoint(workdir, state)
        log_for_0("Restored from checkpoint at {}".format(workdir))
        return state

    # First try the fast, strict path.
    try:
        restored = checkpoints.restore_checkpoint(workdir, state)
        log_for_0("Restored from checkpoint at {} (full restore succeeded).".format(workdir))
        return restored
    except Exception as e:
        log_for_0(
            "Full checkpoint restore failed in params-only mode (%s). "
            "Falling back to restoring step/params/ema_params only.",
            e,
        )

    raw_state = checkpoints.restore_checkpoint(workdir, target=None)

    # Strict params-only first; fall back to lenient if requested or if strict fails
    # with a key-mismatch error and lenient mode is allowed.
    try:
        return _restore_params_only_from_raw(
            state, raw_state, workdir, allow_missing=allow_missing
        )
    except ValueError as e:
        if not allow_missing:
            log_for_0(
                "Strict params-only restore failed (%s). Retrying with "
                "lenient mode (missing keys will keep initialized values). "
                "Set PMF_CKPT_ALLOW_MISSING=1 to enable lenient restore by default.",
                e,
            )
            return _restore_params_only_from_raw(
                state, raw_state, workdir, allow_missing=True
            )
        raise


def save_checkpoint(state, workdir, keep=None, backend=None):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    # Save only one copy from device 0.
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    ckpt_backend = (
        str(backend).lower()
        if backend is not None
        else str(os.environ.get("PMF_CKPT_BACKEND", "auto")).lower()
    )
    keep = _normalize_checkpoint_keep(keep)

    if ckpt_backend == "legacy":
        _save_checkpoint_legacy_single_writer(workdir, state, step, keep=keep)
        log_for_0("Checkpoint step %d saved (legacy backend).", step)
        return

    try:
        if ckpt_backend == "orbax":
            checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=keep)
        else:
            # Default path: try the multiprocess API first.
            checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=keep)
    except Exception as e:
        if _looks_like_orbax_locking_error(e):
            log_for_0(
                "Orbax checkpoint save failed due to filesystem locking support (%s). "
                "Falling back to legacy Flax checkpoint save on process 0.",
                e,
            )
            _save_checkpoint_legacy_single_writer(workdir, state, step, keep=keep)
            log_for_0("Checkpoint step %d saved (legacy fallback).", step)
            return
        raise
    log_for_0("Checkpoint step %d saved.", step)
