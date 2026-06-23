import os
import re
from datetime import datetime
import jax


def maybe_init_distributed():
    """Initialize JAX distributed only when launcher provides config."""
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = os.environ.get("JAX_NUM_PROCESSES")
    process_id = os.environ.get("JAX_PROCESS_ID")

    if coordinator_address and num_processes and process_id:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=int(num_processes),
            process_id=int(process_id),
        )


maybe_init_distributed()

from absl import app, flags
from ml_collections import config_flags

import train
from utils import logging_util
from utils.logging_util import log_for_0

logging_util.supress_checkpt_info()

import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("debug", False, "Debugging mode.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("FLAGS.config: \n{}".format(FLAGS.config))

    def _sanitize_name(name: str) -> str:
        name = str(name).strip()
        if not name:
            return ""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    def _resolve_run_workdir(base_workdir, config):
        logging_cfg = getattr(config, "logging", None)
        if logging_cfg is None:
            return base_workdir

        use_subdir = bool(logging_cfg.get("timestamped_run_subdir", False))
        if not use_subdir:
            return base_workdir

        exp_name = _sanitize_name(logging_cfg.get("exp_name", ""))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dirname = f"{ts}_{exp_name}" if exp_name else ts
        run_workdir = os.path.join(base_workdir, run_dirname)

        # Safe under multiprocess launch with exist_ok=True.
        os.makedirs(run_workdir, exist_ok=True)
        return run_workdir

    run_workdir = _resolve_run_workdir(FLAGS.workdir, FLAGS.config)
    log_for_0("Run workdir: %s", run_workdir)

    if FLAGS.config.eval_only and FLAGS.config.autoguidance.get("enabled", False):
        train.just_evaluate_autoguidance(FLAGS.config, run_workdir)
    elif FLAGS.config.eval_only:
        train.just_evaluate(FLAGS.config, run_workdir)
    else:
        train.train_and_evaluate(FLAGS.config, run_workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
