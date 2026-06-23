import argparse
import os
import re
import sys

import numpy as np

# legacy-flax backend (matches pmf_fd_loss save path)
try:
    from flax import config as _flax_config
    _flax_config.update("flax_use_orbax_checkpointing", False)
except Exception:
    pass

from flax.training import checkpoints
import torch


# --------------------------------------------------------------------------- #
# torch state_dict -> flax flat dict (reverse of convert_pmf_checkpoint)
# --------------------------------------------------------------------------- #

# torch -> flax: shared_blocks.<i>.   ->  shared_blocks_<i>.
_MLIST_RE_BACK = re.compile(r"\.(shared_blocks|u_heads|u_heads_wide|v_heads|v_heads_wide)\.(\d+)\.")
# torch -> flax: mlp.<i>.  ->  mlp.layers_<i>.
_LAYERS_RE_BACK = re.compile(r"\.mlp\.(\d+)\.")


def torch_to_flax_keys(state_dict: dict) -> dict:
    """Reverse of FD-Loss's convert_pmf_checkpoint.

    Returns a *flat* dotted-key dict whose keys/values match the user's
    original Flax param tree (after flattening).
    """
    out = {}
    for k, v in state_dict.items():
        nk = k

        # 1. ModuleList indexing: torch ".N." -> flax "_N."
        nk = _MLIST_RE_BACK.sub(r".\1_\2.", nk)
        # 2. nn.Sequential indexing: mlp.0./mlp.2. -> mlp.layers_0./mlp.layers_2.
        nk = _LAYERS_RE_BACK.sub(r".mlp.layers_\1.", nk)

        # 3. Leaf renames + (un)transpose / (un)squeeze.
        # Handle two input formats:
        #   (a) Real torch state_dict ('.linear.weight' (out, in), 1D '.weight' for RMSNorm,
        #       2D embedding_table.embedding.weight, 2D *_tokens) -> requires transpose/unsqueeze.
        #   (b) Half-converted .pth from flax_to_torch.py ('.linear.kernel' (in, out), '.kernel'
        #       for RMSNorm, 3D embedding_table.embedding, 3D pos_embed already in flax shape) ->
        #       only the path renames are needed; no transpose, no unsqueeze.
        if nk.endswith(".linear.weight"):
            nk = nk[: -len(".linear.weight")] + "._flax_linear.kernel"
            if v.ndim == 2:
                if hasattr(v, "contiguous"):  # torch.Tensor
                    v = v.transpose(0, 1).contiguous()
                else:                         # numpy.ndarray
                    v = np.ascontiguousarray(v.T)
        elif nk.endswith(".linear.kernel"):
            # already in flax (in, out) shape; just rewrap as _flax_linear
            nk = nk[: -len(".linear.kernel")] + "._flax_linear.kernel"
        elif nk.endswith(".linear.bias"):
            nk = nk[: -len(".linear.bias")] + "._flax_linear.bias"
        elif nk.endswith(".embedding_table.embedding.weight"):
            # torch (V, D) -> flax (1, V, D)
            nk = nk[: -len(".weight")]
            if hasattr(v, "unsqueeze"):
                v = v.unsqueeze(0)
            else:
                v = v[None]
        elif nk.endswith(".embedding_table.embedding") and v.ndim == 2:
            # half-converted: (V, D) but missing leading 1 -> add it
            v = v.unsqueeze(0) if hasattr(v, "unsqueeze") else v[None]
        elif nk.endswith("_tokens"):
            if v.ndim == 2:
                v = v.unsqueeze(0) if hasattr(v, "unsqueeze") else v[None]
        elif nk.endswith(".weight") and v.ndim == 1:
            # RMSNorm scale: torch <scope>.weight -> flax <scope>.kernel
            nk = nk[: -len(".weight")] + ".kernel"
        # else (.kernel for RMSNorm, pos_embed (1, T, D), attn_scale, mlp_scale): keep as-is.

        out[nk] = v
    return out


# --------------------------------------------------------------------------- #
# flat dotted dict <-> nested dict (Flax pytree shape)
# --------------------------------------------------------------------------- #

def unflatten_dotted(flat: dict) -> dict:
    """Convert {'a.b.c': v} -> {'a': {'b': {'c': v}}}."""
    nested = {}
    for k, v in flat.items():
        parts = k.split(".")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return nested


def flatten_pytree(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_pytree(v, sub))
        return out
    out[prefix] = np.asarray(tree)
    return out


# --------------------------------------------------------------------------- #
# Conversion driver
# --------------------------------------------------------------------------- #

def torch_state_dict_to_flax_net(sd: dict) -> dict:
    """Take a torch state_dict (flat torch keys, prefixed `net.<...>`) and
    return the nested `{net: {...}}` dict in Flax param-tree shape, as numpy
    arrays."""
    # tensors -> numpy (Flax/JAX consumes np arrays)
    sd_np = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
             for k, v in sd.items()}
    flat_flax = torch_to_flax_keys(sd_np)
    nested = unflatten_dotted(flat_flax)
    return nested  # already has top-level `net: {...}` if input keys were `net.<...>`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True,
                    help="PyTorch FD-Loss training checkpoint (.pth) to convert.")
    ap.add_argument("--template_ckpt",
                    required=True,
                    help="Path to an existing Flax checkpoint file (any step) to use as TrainState template.")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write the new Flax checkpoint into.")
    ap.add_argument("--step", type=int, default=None,
                    help="Step number for the saved checkpoint filename. Defaults to ckpt['step']+1.")
    ap.add_argument("--out_name", default=None,
                    help="If set, rename the written file to this name (within --out_dir), "
                         "dropping flax's checkpoint_<step> convention. The step is still "
                         "stored inside the checkpoint, and downstream loaders read by path "
                         "so any name works.")
    ap.add_argument("--ema_label_map", default="edm_500=500,edm_1000=1000,edm_2000=2000",
                    help="Comma-separated torch_label=jax_key pairs for EMA branches.")
    ap.add_argument("--also_set_params", action="store_true",
                    help="Also overwrite state['params']['net'] (the live params) from torch ckpt['model']. "
                         "Default: only overwrite ema_params (which is what your eval reads).")
    ap.add_argument("--keep", default="all",
                    help="checkpoints.save_checkpoint keep arg. Default keep all.")
    args = ap.parse_args()

    if not os.path.exists(args.pth):
        print(f"[err] no such file: {args.pth}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.template_ckpt):
        print(f"[err] no such template: {args.template_ckpt}", file=sys.stderr); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load torch ----
    print(f"[load] torch ckpt: {args.pth}")
    ckpt = torch.load(args.pth, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"[err] expected dict-like torch ckpt, got {type(ckpt).__name__}", file=sys.stderr)
        sys.exit(1)

    # ---- load template flax state ----
    print(f"[load] flax template: {args.template_ckpt}")
    raw_state = checkpoints.restore_checkpoint(args.template_ckpt, target=None)
    if raw_state is None:
        print(f"[err] flax restore failed", file=sys.stderr); sys.exit(1)
    print(f"[load] template top-level keys: {list(raw_state.keys())}")
    if "ema_params" in raw_state and isinstance(raw_state["ema_params"], dict):
        print(f"[load] template ema_params keys: {list(raw_state['ema_params'].keys())}")

    # ---- parse ema map ----
    ema_map = dict(pair.split("=") for pair in args.ema_label_map.split(",") if pair)
    print(f"[map] ema label map (torch -> jax): {ema_map}")

    # ---- detect ckpt format ----
    is_flat_sd = (
        "model" not in ckpt and "model_ema" not in ckpt
        and any(str(k).startswith("net.") for k in ckpt.keys())
    )

    def _splice_ema(jax_key: str, sd: dict, source_label: str):
        print(f"[ema] converting {source_label} -> ema_params['{jax_key}'] ({len(sd)} tensors)")
        nested = torch_state_dict_to_flax_net(sd)
        if jax_key not in raw_state["ema_params"]:
            print(f"[warn] template ema_params has no key '{jax_key}'; creating new entry")
        target = raw_state["ema_params"].setdefault(jax_key, {})
        if "net" in target or not target:
            target["net"] = nested.get("net", nested)
        else:
            raw_state["ema_params"][jax_key] = nested

    # ---- splice EMA branches ----
    if is_flat_sd:
        # Raw state_dict form (e.g. dumped by flax_to_torch.py). Use it for
        # every requested ema slot.
        print(f"[detect] flat state_dict format ({len(ckpt)} tensors). "
              f"Using as source for all ema slots.")
        for torch_label, jax_key in ema_map.items():
            _splice_ema(jax_key, ckpt, source_label=f"flat_sd ({torch_label})")
    elif "model_ema" in ckpt and ckpt["model_ema"] is not None \
            and "shadows" in ckpt["model_ema"]:
        shadows = ckpt["model_ema"]["shadows"]
        print(f"[ema] torch ema labels in ckpt: {list(shadows.keys())}")
        for torch_label, jax_key in ema_map.items():
            if torch_label not in shadows:
                print(f"[warn] '{torch_label}' not found in torch ckpt; skipping")
                continue
            _splice_ema(jax_key, shadows[torch_label], source_label=torch_label)
    else:
        print(f"[warn] no model_ema in torch ckpt and not a flat state_dict; "
              f"skipping EMA splice")

    # ---- splice live params (optional) ----
    if args.also_set_params:
        if is_flat_sd:
            sd = ckpt
            label = "flat_sd"
        elif "model" in ckpt:
            sd = ckpt["model"]
            label = "model"
        else:
            sd = None
            label = None
        if sd is not None:
            print(f"[params] converting {label} -> params ({len(sd)} tensors)")
            nested = torch_state_dict_to_flax_net(sd)
            if "net" in raw_state["params"] or not raw_state["params"]:
                raw_state["params"]["net"] = nested.get("net", nested)
            else:
                raw_state["params"] = nested

    # ---- determine step ----
    if args.step is None:
        if "step" in ckpt:
            try:
                step = int(ckpt["step"]) + 1
            except Exception:
                step = 0
        else:
            step = 0
    else:
        step = int(args.step)
    raw_state["step"] = step
    print(f"[save] target step: {step}")

    # ---- save ----
    keep = 2**31 - 1 if args.keep == "all" else int(args.keep)
    out_path = checkpoints.save_checkpoint(
        ckpt_dir=args.out_dir,
        target=raw_state,
        step=step,
        keep=keep,
        overwrite=True,
    )
    print(f"[save] wrote {out_path}")

    # Optionally drop flax's checkpoint_<step> filename for a step-free name.
    # (The step lives inside the checkpoint; downstream loaders read by path.)
    if args.out_name:
        dst = os.path.join(args.out_dir, os.path.basename(args.out_name))
        if os.path.abspath(dst) != os.path.abspath(out_path):
            os.replace(out_path, dst)
            out_path = dst
            print(f"[save] renamed -> {out_path}")

    print(f"\n[ok] you can now point your JAX FID eval at:")
    print(f"     {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
