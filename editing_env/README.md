# Interactive editing environment

A small [Gradio](https://www.gradio.app/) app for **interactive latent
token-exchange editing** with TrajectoryForcing. Generate a *reference* and a
*target* image from ImageNet classes, copy latent tokens between them across the
coarse→fine hierarchy levels, and regenerate — all in a few clicks. Can be also
tried on [Colab](https://colab.research.google.com/drive/1CZgGT1rEJ5nQ2D8fLYRyTqUIvLpwzCSP?usp=sharing).

## Demo

https://github.com/user-attachments/assets/ef48cbc4-5349-4d03-a251-b3ef4e7880fa

▶️ If the video does not render inline, open
[`assets/editing_env_demo.mp4`](../assets/editing_demo.mp4) directly.

## What it does

- **Generate** a reference (and optional target) from ImageNet class ids; each
  image is shown across all hierarchy levels (coarse → fine) as both the
  **latent** (PCA visualization) and the **decoded RGB**.
- **Pick an edit level** per side and **click tokens** on the 16×16 grids to copy
  tokens FROM the reference INTO the target (optionally grabbing a whole
  cosine-similarity cluster in one click).
- **Run edit + generate** to re-sample the downstream levels and decode the
  edited result.

Latent decoding here is intentionally **not** normalized (for both generation
and edits).

## Launch

From inside a GPU environment (with the project venv active, so `python` has
JAX/Gradio/PyTorch):

```bash
cd editing_env
./run.sh            # default port 7860 (Gradio's standard default)
./run.sh 8001       # or choose a port
```

`run.sh` points CUDA at the assigned GPU, sets the Gradio/proxy env, enables a
persistent JAX compilation cache, then serves on `http://0.0.0.0:<PORT>`. First
boot warms up the model (~1–2 min) so the first real click is fast.

Some clusters expose the node directly, so you can just open the URL in your browser. If yours doesn't, forward the port over SSH:

```bash
ssh -L <PORT>:<node>:<PORT> <user>@<host>
# then open http://localhost:<PORT>
```

```bash
ssh -L 8001:<node>:8001 <user>@<host>
# then open http://localhost:8001
```

## Models (downloaded automatically)

- **Editing checkpoint** — the inference EMA-500 pMF-L/16 checkpoint, released as
  the single file
  [`TF_L_edit`](https://huggingface.co/mervekocabas/TrajectoryForcing/blob/main/TF_L_edit)
  on `mervekocabas/TrajectoryForcing`. `tf_pipeline.py` fetches it on first use.
  Override with the `TF_LOAD_FROM` env var (local file) or `TF_CKPT_REPO` /
  `TF_CKPT_FILE`.
- **RAE decoder** — `model.pt` is auto-downloaded from the public
  [`nyu-visionx/RAE-collections`](https://huggingface.co/nyu-visionx/RAE-collections)
  into `checkpoints/rae/` on first use (shared with the main eval pipeline).

> **Behind a proxy:** HuggingFace's Xet transfer can hang. If a download stalls,
> fetch the files with `curl` from their `…/resolve/main/…` URLs (the classic
> HTTPS path works), or pre-place them at the paths above.

## Files

```
editing_env/
├── app.py                  # Gradio UI + generate/edit callbacks
├── tf_pipeline.py          # load-once JAX model + RAE decoder; generate/edit/decode
├── imagenet_classes.json   # ImageNet-1k id -> class name (0–999)
└── run.sh                  # launcher (GPU + proxy + Gradio env)
```

The app reuses the repo-root machinery (`pmf.py`, `utils/`,
`third_party/rae_decoder/`) and the config
[`configs/edit_env_config.yml`](../configs/edit_env_config.yml), whose model
architecture matches the `TF_L_edit` checkpoint.
