import argparse
import datetime
import json
import logging
import os

import torch

from utils.distributed_util import enable_distributed, get_global_rank, get_local_rank, get_world_size
from utils.logging_util import setup_logging, setup_wandb
from utils.rng_util import fix_random_seeds

logger = logging.getLogger("FD_loss")

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_TOKENIZER_SPECS = {
    "sdvae":   {"channels": 4,  "patch_size": 8},
    "fluxvae": {"channels": 16, "patch_size": 8},
    "sd35vae": {"channels": 16, "patch_size": 8},
    "wanvae":  {"channels": 16, "patch_size": 8},
}

def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration."""
    enable_distributed()

    # experiment directories
    if args.exp_name is None:
        args.exp_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp"
    base = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base
    args.ckpt_dir = os.path.join(base, "checkpoints")
    args.vis_dir = os.path.join(base, "visualization")
    args.eval_dir = os.path.join(base, "eval")
    if args.local_eval_dir:
        args.eval_dir = args.eval_dir.replace(args.output_dir, args.local_eval_dir)

    # distributed / global config
    rank, world_size = get_global_rank(), get_world_size()
    args.world_size = world_size
    args.rank = rank
    args.local_rank = get_local_rank()
    args.global_bsz = args.batch_size * world_size
    fix_random_seeds(args.seed + rank)

    if args.warmup_epochs == -1:
        args.warmup_epochs = int(args.warmup_rate * args.epochs)
        
    args.enable_amp = args.dtype != "fp32"

    # tokenizer defaults
    if args.tokenizer and args.tokenizer in _TOKENIZER_SPECS:
        spec = _TOKENIZER_SPECS[args.tokenizer]
        args.token_channels = spec["channels"]
        args.tokenizer_patch_size = spec["patch_size"]
        
    # compute input size to the model (for preparing dummy inputs)
    input_size = args.img_size // args.tokenizer_patch_size
    args.input_size = (args.token_channels, input_size, input_size)
    
    # set up step-based schedule
    steps_per_epoch = args.steps_per_epoch
    args.total_steps = steps_per_epoch * args.epochs
    args.save_every = steps_per_epoch * args.save_freq
    args.vis_every = steps_per_epoch * args.vis_freq
    args.val_every = steps_per_epoch * args.val_freq
    args.eval_every = steps_per_epoch * args.eval_freq
    args.milestone_every = steps_per_epoch * args.milestone_interval
    args.warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    logger.info(f"step-based schedule: total_steps={args.total_steps}, "
                f"save_every={args.save_every}, vis_every={args.vis_every}, "
                f"val_every={args.val_every}, eval_every={args.eval_every}, "
                f"milestone_every={args.milestone_every}")

    # logging / wandb (rank 0 only)
    wandb_logger = None
    if rank == 0:
        for d in (args.log_dir, args.ckpt_dir, args.vis_dir, args.eval_dir):
            os.makedirs(d, exist_ok=True)

        if args.enable_wandb:
            wandb_logger = setup_wandb(args, args.entity, args.project,
                                       args.exp_name, args.log_dir)

        setup_logging(output=args.log_dir)
        logger.info(f"logging to {args.log_dir}")
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
        cfg_path = os.path.join(args.log_dir, f"args_{timestamp}.json")
        with open(cfg_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"args saved to {cfg_path}")

    args.amp_dtype = _DTYPE_MAP[args.dtype]
    return wandb_logger
