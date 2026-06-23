#!/bin/bash
set -euo pipefail

DEST="${1:-checkpoints/rae}"
SIZE="${SIZE:-ViTXL}"
REPO="nyu-visionx/RAE-collections"

DECODER="decoders/dinov2/wReg_base/${SIZE}_n08/model.pt"
STATS="stats/dinov2/wReg_base/imagenet1k/stat.pt"

echo "Downloading RAE ${SIZE} decoder + ImageNet stats from ${REPO} into ${DEST}/ ..."
hf download "${REPO}" "${DECODER}" "${STATS}" --local-dir "${DEST}"

echo
if [[ "${SIZE}" == "ViTXL" ]]; then
    echo "Done. configs/eval_config.yml already defaults to:"
    echo "  rae_decoder.pretrained_decoder_path: ${DEST}/${DECODER}"
    echo "  rae_decoder.normalization_stat_path: ${DEST}/${STATS}"
else
    echo "Done. To use this ${SIZE} decoder instead of the default ViTXL:"
    echo "  1. Place a matching HF-style ViTMAE config at:"
    echo "       third_party/rae_decoder/configs/${SIZE}/config.json"
    echo "     (copy from the RAE repo / adjust decoder_hidden_size etc.)"
    echo "  2. In configs/eval_config.yml set:"
    echo "       rae_decoder.decoder_config_path: third_party/rae_decoder/configs/${SIZE}"
    echo "       rae_decoder.pretrained_decoder_path: ${DEST}/${DECODER}"
    echo "       rae_decoder.normalization_stat_path: ${DEST}/${STATS}"
fi
