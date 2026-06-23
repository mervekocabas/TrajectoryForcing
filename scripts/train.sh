#!/bin/bash

# `dataset.root` in configs/TF_B_config.yml is pre-filled to the default
# encode output (preprocessed_data/train). Update `fid.cache_ref` there if
# you've placed FID stats somewhere other than the default.
export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${now}_$1

# Model variant (TF_B, TF_L, TF_H). Defaults to TF_B if not given.
export MODEL=${2:-TF_B}

# Checkpoints and config.yml are written to the workdir, so point it at outputs/.
export OUT_DIR=outputs/$JOBNAME
# Training log goes under logs/training/.
export LOG_DIR=logs/training/$JOBNAME

mkdir -p ${OUT_DIR}
mkdir -p ${LOG_DIR}
chmod 755 -R ${OUT_DIR}
chmod 755 -R ${LOG_DIR}

python3 main.py \
    --workdir=${OUT_DIR} \
    --config=configs/load_config.py:${MODEL} \
    2>&1 | tee -a $LOG_DIR/output.log
