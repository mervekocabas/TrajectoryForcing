import jax.numpy as jnp
import numpy as np


def _top3_pca_dirs(flat_centered, c):
    """Top-3 principal directions over the channel dim, without a full SVD.

    The right singular vectors of X (N x C) are the eigenvectors of the C x C gram
    matrix X^T X. Since C (=768) << N and only 3 components are needed, an eigh of the
    small gram matrix is ~10x cheaper than np.linalg.svd, which solves for all C
    singular vectors. Returns a (C, 3) array of principal directions (zero-padded if
    the decomposition degenerates). Used for the PCA latent-colour visualization only.
    """
    try:
        gram = flat_centered.T @ flat_centered            # (C, C)
        evals, evecs = np.linalg.eigh(gram)               # ascending eigenvalues
        pcs = evecs[:, ::-1][:, :3]                        # top-3 by eigenvalue
    except np.linalg.LinAlgError:
        pcs = np.zeros((c, 0), dtype=np.float32)
    if pcs.shape[1] < 3:
        pad = np.zeros((c, 3 - pcs.shape[1]), dtype=np.float32)
        pcs = np.concatenate([pcs, pad], axis=1)
    return pcs[:, :3]


def make_grid_visualization(vis, grid=8, max_bz=8):
    assert vis.ndim == 4
    n, h, w, c = vis.shape

    col = grid
    row = min(grid, n // col)
    if n % (col * row) != 0:
        n = col * row * max_bz
        vis = vis[:n]
        n, h, w, c = vis.shape
    assert n % (col * row) == 0

    vis = vis.reshape((-1, col, row * h, w, c))
    vis = jnp.einsum("mlhwc->mhlwc", vis)
    vis = vis.reshape((-1, row * h, col * w, c))

    bz = min(vis.shape[0], max_bz)
    vis = vis[:bz]
    return vis


def latent_levels_to_pca_column(latents, gap=1, bg_value=0):
    """Colorize latent levels with PCA and stack vertically.

    Args:
        latents: numpy array of shape (L, H, W, C)
        gap: number of pixels between rows
        bg_value: background value for gaps in [0, 255]

    Returns:
        uint8 image of shape (L*H + (L-1)*gap, W, 3)
    """
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected latents with shape (L,H,W,C), got {x.shape}")
    l, h, w, c = x.shape
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    flat = x.reshape(l * h * w, c)
    flat_centered = flat - flat.mean(axis=0, keepdims=True)

    # Top principal directions over the channel dim (eigh of the small gram matrix).
    pcs = _top3_pca_dirs(flat_centered, c)

    rgb = flat_centered @ pcs[:, :3]
    lo = np.percentile(rgb, 1.0, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99.0, axis=0, keepdims=True)
    scale = np.maximum(hi - lo, 1e-6)
    rgb = np.clip((rgb - lo) / scale, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8).reshape(l, h, w, 3)

    if gap <= 0:
        return np.concatenate([rgb[i] for i in range(l)], axis=0)

    out_h = l * h + (l - 1) * gap
    out = np.full((out_h, w, 3), int(bg_value), dtype=np.uint8)
    for i in range(l):
        y0 = i * (h + gap)
        out[y0 : y0 + h] = rgb[i]
    return out


def fit_pca_palette(latents_list):
    """Fit a SHARED PCA color basis + normalization from one or more (L,H,W,C) arrays.

    Returns a palette dict {mean, pcs, lo, hi} that can be passed to apply_pca_palette
    so several latent stacks (reference / target / edited) share the same color mapping.
    """
    flats = []
    c = None
    for lat in latents_list:
        a = np.asarray(lat, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        c = a.shape[-1]
        flats.append(a.reshape(-1, c))
    flat = np.concatenate(flats, axis=0)
    mean = flat.mean(axis=0, keepdims=True)
    flat_centered = flat - mean
    pcs = _top3_pca_dirs(flat_centered, c)
    proj = flat_centered @ pcs
    lo = np.percentile(proj, 1.0, axis=0, keepdims=True)
    hi = np.percentile(proj, 99.0, axis=0, keepdims=True)
    return {
        "mean": mean.astype(np.float32),
        "pcs": pcs.astype(np.float32),
        "lo": lo.astype(np.float32),
        "hi": hi.astype(np.float32),
    }


def apply_pca_palette(latents, palette, gap=1, bg_value=0):
    """Colorize (L,H,W,C) latents using a precomputed palette (no refit), then stack
    vertically. Mirrors latent_levels_to_pca_column's output layout."""
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected latents with shape (L,H,W,C), got {x.shape}")
    l, h, w, c = x.shape
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(l * h * w, c) - palette["mean"]
    rgb = flat @ palette["pcs"]
    scale = np.maximum(palette["hi"] - palette["lo"], 1e-6)
    rgb = np.clip((rgb - palette["lo"]) / scale, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8).reshape(l, h, w, 3)

    if gap <= 0:
        return np.concatenate([rgb[i] for i in range(l)], axis=0)
    out_h = l * h + (l - 1) * gap
    out = np.full((out_h, w, 3), int(bg_value), dtype=np.uint8)
    for i in range(l):
        y0 = i * (h + gap)
        out[y0 : y0 + h] = rgb[i]
    return out
