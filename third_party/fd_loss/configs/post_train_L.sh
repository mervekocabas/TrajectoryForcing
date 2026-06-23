#!/bin/bash
set -euo pipefail
# FD-loss post-training for a pmfDiT_L_16 TF checkpoint.
# You only ever handle TF (flax) checkpoints -- conversion both ways is automatic:
# TF(flax) -> torch before training, torch -> TF(flax) after.
#   Usage  : bash configs/post_train_L.sh <TF_ckpt> [output_dir]
#            TF_ckpt = a TF/flax checkpoint file of any name (e.g. <run>/checkpoint_<step> or .../TF_B)
#                      output_dir defaults to the project outputs/ dir
#   Output : <output_dir>/pmfDiT_fd/pmfDiT_L_16-fd/tf_checkpoint/<TF_ckpt name> (TF/flax)
#            (named after your input <TF_ckpt>; step is stored inside the checkpoint)
cd "$(dirname "$0")/.."   # -> third_party/fd_loss

TF_CKPT="${1:-${TF_CKPT:-}}";   : "${TF_CKPT:?usage: $(basename "$0") <TF_ckpt> [output_dir]}"
OUT_DIR="${2:-${OUT_DIR:-../../outputs}}"   # default: project outputs/ dir (not alongside the TF checkpoint under checkpoints/)
EMA="${EMA:-500}"

EXP=pmfDiT_L_16-fd; OUT="$OUT_DIR/pmfDiT_fd/$EXP"
mkdir -p "$OUT/checkpoints"
# Label the cached torch conversion by step if the name carries one, else by basename.
# (TF_CKPT may be a checkpoint file of any name -- the step lives inside the checkpoint.)
CK_STEP=$(basename "$TF_CKPT" | grep -oE '[0-9]+' | head -1 || true)
CK_LABEL="${CK_STEP:-$(basename "$TF_CKPT")}"
INIT_PTH="$OUT/init_from_tf_${CK_LABEL}_ema${EMA}.pth"

# TF (flax) -> torch (once). --ckpt_path accepts a checkpoint file (or dir) of any name.
[ -f "$INIT_PTH" ] || JAX_PLATFORMS=cpu python scripts/flax_to_torch.py \
    --ckpt_path "$TF_CKPT" --ema "$EMA" --out_pth "$INIT_PTH"

# fetch siglip + mae FD reference stats into fid_ref/ (download once if missing)
python scripts/fetch_repr_stats.py ../../fid_ref

# on exit: export the latest checkpoint back to TF (flax) format
export_tf() {
    local last; last=$(ls -t "$OUT/checkpoints/step_"*.pth 2>/dev/null | grep -v corrupt | head -1) || true
    [ -n "${last:-}" ] && JAX_PLATFORMS=cpu python scripts/torch_to_flax.py \
        --pth "$last" --template_ckpt "$TF_CKPT" --out_dir "$OUT/tf_checkpoint" \
        --out_name "$(basename "$TF_CKPT")" \
        --ema_label_map edm_${EMA}.0=${EMA} --also_set_params || true
}
trap export_tf EXIT

torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29516 --nproc_per_node=8 \
    main_fd.py \
    --project pmfDiT_fd \
    --exp_name pmfDiT_L_16-fd \
    --output_dir "$OUT_DIR" \
    --batch_size 32 \
    --load_from "${INIT_PTH}" \
    --model pmfDiT_L_16 --tokenizer rae_dinov2_b_vitxl \
    --rae_stats_path none \
    --img_size 16 --patch_size 1 --token_channels 768 \
    --noise_scale 1.0 \
    --cfg 1.0 --interval_min 0.0 --interval_max 1.0 \
    --num_sampling_steps 1 \
    --eval_bsz 32 --num_images_for_eval_and_search 50000 \
    --vis_freq 1000 --eval_freq 1000 \
    --print_freq 1 --milestone_interval 10 --save_freq 5 \
    --epochs 80 --steps_per_epoch 1250 --warmup_epochs 5 \
    --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
    --ema_halflife_kimg 500 \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --fid_stats_path ../../fid_ref/imagenet_256_fid_stats.npz \
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception \
    --fd_repr_stats_paths \
        ../../fid_ref/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz \
        ../../fid_ref/vit_large_patch16_224_mae_in256_t224_stats.npz \
        ../../fid_ref/imagenet_256_fid_stats.npz \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --auto_resume --disable_wandb --disable_vis
