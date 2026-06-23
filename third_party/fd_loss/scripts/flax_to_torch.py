import argparse
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

# Match pmf_fd_loss save path: legacy Flax backend, not Orbax.
try:
    from flax import config as _flax_config
    _flax_config.update("flax_use_orbax_checkpointing", False)
except Exception:
    pass

from flax.training import checkpoints


def flatten_pytree(tree: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    """Flatten nested-dict / list pytree to {dotted.key: np.ndarray}."""
    out: Dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_pytree(v, sub))
        return out
    if isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            sub = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_pytree(v, sub))
        return out
    out[prefix] = np.asarray(tree)
    return out


def select_ema_branch(state: Dict, ema_key: str) -> Tuple[Dict, str]:
    """Pick the requested EMA params from state['ema_params']. Returns (params, label)."""
    if ema_key == "raw":
        return state["params"], "params (live, non-EMA)"
    ema = state.get("ema_params", {}) or {}
    candidates = list(ema.keys())
    # Flax legacy serializer encodes float keys as their str repr.
    for k in candidates:
        if str(k) == str(ema_key) or str(k) == str(float(ema_key)):
            return ema[k], f"ema_params[{k}]"
    raise KeyError(
        f"EMA key {ema_key!r} not found in checkpoint; available: {candidates}. "
        f"Use --ema raw to dump live params."
    )


def fd_loss_rename(flat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """FD-Loss-style cosmetic renames (matches convert_pmf_checkpoint)."""
    out: Dict[str, np.ndarray] = {}
    dropped = []
    squeezed = []
    renamed = 0
    for k, v in flat.items():
        nk = k.replace("._flax_linear.", ".linear.").replace(
            "._flax_embedding.", ".embedding."
        )
        if nk != k:
            renamed += 1
        if "rope_freqs" in nk:
            dropped.append(k)
            continue
        if nk.endswith("_tokens") and v.ndim == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
            squeezed.append(nk)
        out[nk] = v
    print(f"[rename] flax_linear/flax_embedding renames applied: {renamed}")
    print(f"[rename] rope_freqs buffers dropped: {len(dropped)}")
    print(f"[rename] _tokens (1,N,D)->(N,D) squeezed: {len(squeezed)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", default=None,
                    help="Direct path to a legacy-Flax checkpoint file (any name, no "
                         "checkpoint_<step> convention required) or a directory (latest "
                         "checkpoint_* is used). Takes precedence over --ckpt_dir/--step.")
    ap.add_argument("--ckpt_dir", default=None,
                    help="Directory holding checkpoint_<step> (legacy Flax format).")
    ap.add_argument("--step", type=int, default=None,
                    help="Specific step file to load. Defaults to latest in dir.")
    ap.add_argument("--ema", default="2000",
                    help="EMA params to dump (key from training.ema_val), or 'raw' "
                         "for live (non-EMA) params. Default: 2000.")
    ap.add_argument("--out_npz", default=None,
                    help="If set, write flat numpy arrays here (no key renames).")
    ap.add_argument("--out_pth", default=None,
                    help="If set, write torch state-dict here with FD-Loss "
                         "rename heuristics applied. Requires torch.")
    ap.add_argument("--print_n", type=int, default=200,
                    help="How many keys to print after flattening. Default: 200. "
                         "Use 0 for all.")
    ap.add_argument("--save_keylist", default=None,
                    help="If set, write the full sorted key+shape+dtype list to this file.")
    args = ap.parse_args()

    if args.ckpt_path is not None:
        # Direct path: a checkpoint file of any name, or a directory.
        if not os.path.exists(args.ckpt_path):
            print(f"[err] no such path: {args.ckpt_path}", file=sys.stderr)
            sys.exit(1)
        if os.path.isdir(args.ckpt_path):
            print(f"[load] restoring latest checkpoint_* under: {args.ckpt_path}")
            state = checkpoints.restore_checkpoint(args.ckpt_path, target=None,
                                                   prefix="checkpoint_")
        else:
            print(f"[load] restoring single file: {args.ckpt_path}")
            state = checkpoints.restore_checkpoint(args.ckpt_path, target=None)
    elif args.ckpt_dir is None:
        print("[err] pass --ckpt_path <file|dir> (or --ckpt_dir [--step]).", file=sys.stderr)
        sys.exit(1)
    elif args.step is not None:
        ckpt_path = os.path.join(args.ckpt_dir, f"checkpoint_{args.step}")
        if not os.path.exists(ckpt_path):
            print(f"[err] no such file: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[load] restoring single file: {ckpt_path}")
        state = checkpoints.restore_checkpoint(ckpt_path, target=None)
    else:
        print(f"[load] restoring latest checkpoint_* under: {args.ckpt_dir}")
        state = checkpoints.restore_checkpoint(args.ckpt_dir, target=None,
                                               prefix="checkpoint_")

    if state is None:
        src = args.ckpt_path or args.ckpt_dir
        print(f"[err] no checkpoint loaded from {src}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] top-level keys: {list(state.keys())}")
    if "step" in state:
        try:
            print(f"[load] step: {int(np.asarray(state['step']))}")
        except Exception:
            print(f"[load] step (raw): {state['step']!r}")
    if "ema_params" in state and isinstance(state["ema_params"], dict):
        print(f"[load] ema_params keys: {list(state['ema_params'].keys())}")

    params, label = select_ema_branch(state, args.ema)
    print(f"[load] using {label}")

    flat = flatten_pytree(params, prefix="")
    n_params = sum(int(v.size) for v in flat.values())
    print(f"[flatten] {len(flat)} tensors, {n_params/1e6:.2f}M params total")

    sorted_keys = sorted(flat.keys())
    n_to_print = len(sorted_keys) if args.print_n == 0 else min(args.print_n, len(sorted_keys))
    print(f"[keys] first {n_to_print}:")
    for k in sorted_keys[:n_to_print]:
        v = flat[k]
        print(f"  {k:<80} {tuple(v.shape)}  {v.dtype}")

    if args.save_keylist:
        with open(args.save_keylist, "w") as f:
            for k in sorted_keys:
                v = flat[k]
                f.write(f"{k}\t{tuple(v.shape)}\t{v.dtype}\n")
        print(f"[save] wrote key list -> {args.save_keylist}")

    if args.out_npz:
        np.savez(args.out_npz, **flat)
        print(f"[save] wrote {args.out_npz}")

    if args.out_pth:
        try:
            import torch
        except ImportError:
            print("[err] torch not installed in this env; rerun with torch available "
                  "or omit --out_pth.", file=sys.stderr)
            sys.exit(1)
        renamed = fd_loss_rename(flat)
        sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in renamed.items()}
        torch.save(sd, args.out_pth)
        print(f"[save] wrote {args.out_pth} ({len(sd)} tensors after renames)")


if __name__ == "__main__":
    main()
