#!/bin/bash

# Note: update `fid.cache_ref` in configs/eval_config.yml to point at your
# FID stats file.
export now=`date '+%Y%m%d_%H%M%S'`
export JOBNAME=${now}_$1
export LOG_DIR=logs/eval/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 755 -R ${LOG_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:eval \
    2>&1 | tee -a $LOG_DIR/output.log