# FD-Loss Post-Training

Post-train a **Trajectory Forcing** `pmfDiT` checkpoint with a **Fréchet-Distance (FD)
loss** to improve sample quality. Training optimizes the few-step generator so its
samples match ImageNet-256 reference statistics in several representation spaces
(SigLIP, MAE, Inception) simultaneously.

## Requirements

- A CUDA GPU node. The provided configs assume **8 GPUs** (`torchrun --nproc_per_node=8`).
- The training stack from the main repo (PyTorch, timm, etc.). FD reference statistics
  are downloaded automatically on first run.
- Run all commands **from this directory** (`third_party/fd_loss/`). The wrapper
  scripts `cd` here themselves, so they can be launched from anywhere.

## Quick start

```bash
# bash configs/post_train_<SIZE>.sh <TF_ckpt> [output_dir]
bash configs/post_train_L.sh /path/to/run/ckpt
```

- `<TF_ckpt>` — a TF (flax) checkpoint **file of any name** (e.g.
  `<run>/checkpoint_<step>`, `TF_B`, `TF_L_edit`); no `checkpoint_<step>`
  naming convention is required. Set `EMA=` to pick which EMA copy to convert
  from (default `500`; whatever keys the checkpoint actually contains —
  e.g. `500`, `1000`, `2000`, or only `500`).
- `[output_dir]` — optional; **defaults to the project `outputs/` directory**
  (`../../outputs`), not alongside the TF checkpoint.

That single command runs the whole pipeline end to end.

## What the wrapper does

1. **`TF (flax) → torch`** (once): converts `<TF_ckpt>` at EMA `500` to
   `init_from_tf_<step>_ema500.pth` (skipped if it already exists).
2. **Fetch FD reference stats**: downloads SigLIP + MAE (+ uses Inception) reference
   statistics into `../../fid_ref/` (skipped if present).
3. **Train** (`main_fd.py` via `torchrun`): FD-loss post-training for 80 epochs
   × 1250 steps, `lr 1e-6` cosine, with `--auto_resume` (safe to re-launch to continue).
4. **`torch → TF (flax)`** on exit (always, via a `trap`): exports the latest checkpoint
   back to flax so you can drop it straight into the inference / editing env.

### Output

```
<output_dir>/pmfDiT_fd/pmfDiT_<SIZE>_16-fd/
├── checkpoints/step_*.pth        # torch training checkpoints
├── init_from_tf_<step>_ema500.pth
└── tf_checkpoint/<TF_ckpt name>   # ← final TF (flax) checkpoint to use for inference
```

The exported file is named after your input `<TF_ckpt>` (e.g. `TF_B`), not
flax's `checkpoint_<step>` convention — the step is still stored inside the
checkpoint. Downstream eval / editing load whatever path you point `load_from`
at, so the filename doesn't matter and you can rename it freely. (To keep the
raw `checkpoint_<step>` name instead, drop the `--out_name` flag from the
`torch_to_flax.py` call in the wrapper.)

## Configuration

Override via environment variables (no need to edit the scripts):

```bash
EMA=1000 OUT_DIR=/scratch/my_fd_run bash configs/post_train_L.sh /path/to/ckpt
```

- `EMA` (default `500`) — which EMA copy of the source checkpoint to convert and train from.
- `OUT_DIR` (default: the project `outputs/` directory, `../../outputs`) — where outputs are written.

For fewer/more GPUs, edit `--nproc_per_node` (and `--batch_size`, which is **per GPU**)
in the relevant `configs/post_train_<SIZE>.sh`. Other hyperparameters
(`--epochs`, `--lr`, `--num_sampling_steps`, FD representation models, etc.) are set in
the same script and passed through to `main_fd.py`.

## Manual checkpoint conversion (optional)

The wrapper handles conversion automatically; use these only if you need it standalone.

```bash
# TF (flax) -> torch
JAX_PLATFORMS=cpu python scripts/flax_to_torch.py \
    --ckpt_dir <run_dir> --step <step> --ema 500 --out_pth out.pth

# torch -> TF (flax)
JAX_PLATFORMS=cpu python scripts/torch_to_flax.py \
    --pth step_XXXX.pth --template_ckpt <TF_ckpt> --out_dir tf_checkpoint \
    --ema_label_map edm_500.0=500 --also_set_params
```
