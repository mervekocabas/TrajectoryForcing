"""Fetch the siglip + mae FD reference statistics into the repo's fid_ref/.

Downloads once (skips if already present). Source: the FD-Loss release bundle
`data/fid_stats/paper_ref_stats.pkl` on Hugging Face (jjiaweiyang/FD-Loss),
unpacked into individual `<name>_stats.npz` files (keys: mu/sigma).

Usage:  python scripts/fetch_repr_stats.py [out_dir]   # default out_dir=../../fid_ref
"""
import os
import sys
import pickle

import numpy as np

HF_REPO = "jjiaweiyang/FD-Loss"
BUNDLE = "data/fid_stats/paper_ref_stats.pkl"
NEEDED = [
    "vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
    "vit_large_patch16_224_mae_in256_t224_stats.npz",
]


class _NpCompatUnpickler(pickle.Unpickler):
    """Load numpy-2.x pickles under numpy<2 (remaps numpy._core -> numpy.core)."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "fid_ref")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    missing = [n for n in NEEDED if not os.path.exists(os.path.join(out_dir, n))]
    if not missing:
        print(f"[repr-stats] siglip + mae stats already present in {out_dir}")
        return

    from huggingface_hub import hf_hub_download
    print(f"[repr-stats] downloading {BUNDLE} from {HF_REPO} ...")
    pkl_path = hf_hub_download(repo_id=HF_REPO, filename=BUNDLE)
    bundle = _NpCompatUnpickler(open(pkl_path, "rb")).load()

    for name in missing:
        if name not in bundle:
            raise KeyError(f"{name} not in {BUNDLE}; available: {list(bundle.keys())}")
        out = os.path.join(out_dir, name)
        np.savez(out, **bundle[name])
        print(f"[repr-stats] wrote {out}")


if __name__ == "__main__":
    main()
