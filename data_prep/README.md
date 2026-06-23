# Data preparation (`data_prep/`)

This stage encodes ImageNet images into the hierarchical latents that
TrajectoryForcing trains on. A frozen representation encoder (DINOv2-with-
registers by default) maps each image to a latent grid `z`; a 4-level region
hierarchy is computed on top of it; and everything is written into sharded
`.tar` archives that the training data loader reads.

## Input: ImageNet as an ImageFolder tree

The encoder reads images with torchvision's `ImageFolder`
(`data_prep/dataloader/imagenet_loader.py`), so ImageNet must be extracted into
the standard per-class folder layout. By default the encoder expects it at
`data/imagenet/` (set via `base_dir` in `configs/choose_encoder.yaml`):

```
data/imagenet/
└── train/
    ├── n01440764/   *.JPEG
    └── ...
```

- Class labels come from `ImageFolder`'s sorted-WNID ordering — the canonical
  ImageNet `0..999`.

## Configure the encoder

Edit [`configs/choose_encoder.yaml`](configs/choose_encoder.yaml):

| Field | Meaning |
|---|---|
| `base_dir` | directory containing the `train/` folder |
| `out_root` | output directory for the encoded latents |
| `max_batches` | `-1` for the whole split; a small number for a quick test |
| `dataloader.imagenetsmall` | `null` to encode all classes (WNIDs come from the folder names); or a JSON `{"root": {"<wnid>": "...", ...}}` to restrict to a subset (only the keys are read) |
| `encoder_type` | `dinov2` (default), `siglip2`, or `dinov3` |
| `encoders.<type>.normalization_stat_path` | latent normalization stats (`NORMALIZATION_STAT_PATH`) |
| `save.*` | archive/sharding options (`z_dtype`, `samples_per_tar`, `shard_prefix`, ...) |

## Run

For full ImageNet, use the **multi-GPU wrapper**. It fans out one encoder
process per visible GPU (uses a `DistributedSampler` so each rank only iterates
its 1/N share of the dataset) and runs `--finalize_only` at the end to write
the unified manifest + `class_map.json`:

```bash
bash scripts/preprocess_data.sh
# custom config / extra args forwarded to every rank:
# bash scripts/preprocess_data.sh path/to/config.yaml --encoder_amp_dtype bfloat16
# subset of GPUs:
# CUDA_VISIBLE_DEVICES=0,2,3 bash scripts/preprocess_data.sh
# force the per-rank cluster-worker count if the scheduler's `nproc` is wrong:
# CLUSTER_WORKERS=8 bash scripts/preprocess_data.sh ...
```

The same script handles 1 GPU and N GPUs — it auto-detects from
`CUDA_VISIBLE_DEVICES` / `nvidia-smi` and skips the post-hoc
`--finalize_only` pass when `WORLD_SIZE=1`.

### Recommended hardware

| GPUs | CPUs | CPU/GPU | wall time |
|---|---|---|---|
| **8** | **64** | **8** | **~18-20 min** 

Aim for **≥8 CPU per GPU**. Each rank runs `cluster_workers=CPU/GPU` worker
processes plus a dataloader (4 procs) and a main thread; below ~8 CPU/GPU,
those processes oversubscribe the cores and the GPU sits idle waiting on CPU
work. The opposite end (e.g. 8 GPU × 8 CPU = 1 CPU/GPU) is the worst case —
single-GPU on the same total CPU budget will finish faster.

Per-rank logs land under `logs/preprocess/<timestamp>/`. Requires
`dataloader.shuffle: false` and `save.archive_mode: sharded` (both are the
defaults in `configs/choose_encoder.yaml`).

### Encoder flags

The script forwards any extra args after the config path straight to
`python data_prep/imagenet1k_encoder.py`. Useful flags:

- `--cluster_workers N` — number of CPU worker processes per rank for the
  per-image hierarchy compute. Default `-1` = **auto** (`cpu_count //
  world_size`, capped at 16); `0` = serial; `>0` = explicit. The script's
  `CLUSTER_WORKERS=N` env var is the recommended way to set this so the
  same value reaches every rank.
- `--encoder_amp_dtype bfloat16` — bf16 autocast on the encoder forward
  (~2× GPU throughput on bf16-capable GPUs).
- `--shard_prefix NAME` — override the output shard name prefix.
- `--finalize_only` — rebuild `.latent_sample_index.pkl` (+ `class_map.json`)
  from existing `.tar` shards; the script runs this automatically after a
  multi-rank encode.
- `--image path.jpg --single_out DIR` — encode a single image (handy smoke test).

## Output

For each image the encoder writes a sample containing:

- `z` — the latent grid `[C, H, W]` (e.g. `[768, 16, 16]` for DINOv2-base at 224 px).
- `objbg_ids`, `parts_ids`, `subparts_ids_global` — `[H, W]` region-id maps for
  the three coarser levels.
- `meta`, `label`.

These are archived into `.tar` shards under `out_root/<split>/`, alongside a
`.latent_sample_index.pkl` manifest (byte offsets for fast random access) and
`class_map.json`. Point the training config's `dataset.root` at `out_root`.

## The 4-level hierarchy

Training reconstructs four levels per sample, coarse to fine:

| Level | Built from |
|---|---|
| 0 | `objbg_ids` — object vs. background |
| 1 | `parts_ids` — object parts (hierarchical-clustering cut `h_part = 0.65`) |
| 2 | `subparts_ids_global` — finer subparts (cut `h_sub = 0.35`) |
| 3 | `z` — the raw latent (finest) |

The cut heights are overridable via `save_cfg` (`h_part`, `h_sub`); a lower cut
yields more (finer) regions. Cuts from the same linkage tree are nested, so
`objbg ⊇ parts ⊇ subparts` holds by construction.

## Contents

```
data_prep/
├── imagenet1k_encoder.py   # entry point: encode a split into latents
├── encoder.py              # EncoderOnly wrapper around the representation encoder
├── dataloader/             # ImageFolder-based ImageNet loader
├── models/                 # encoder definitions (dinov2 / dinov3 / siglip2 / mae) + registry
├── configs/                # choose_encoder.yaml
└── visualize_4levels.py    # optional: PCA visualization of the 4 levels
```
