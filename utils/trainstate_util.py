import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Dict
from functools import partial
from copy import deepcopy
import ml_collections
import optax

from flax.training import train_state
from utils.logging_util import log_for_0
from utils.state_util import print_params
from utils.ema_util import update_ema


#######################################################
#                    Initialize                       #
#######################################################


def initialized(key, image_size, image_channels, model):
    input_shape = (1, image_size, image_size, image_channels)
    x = jnp.ones(input_shape)
    t = jnp.ones((1,), dtype=int)
    y = jnp.ones((1,), dtype=int)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing params done.")

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: " + str(param_count))
    return variables, variables["params"]


#######################################################
#                     Train State                     #
#######################################################


class TrainState(train_state.TrainState):
    ema_params: Dict[float, Any] # {float: params}. e.g., {0.9995: ..., 0.9999: ...}

def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, lr_fn,
):
    """
    Create initial training state.
    ---
    apply_fn: output a dict, with key 'loss', 'mse'
    """

    rng, rng_init = random.split(rng)

    _, params = initialized(
        rng_init,
        image_size,
        int(config.dataset.image_channels),
        model,
    )
    ema_vals = config.training.ema_val
    if isinstance(ema_vals, float) or isinstance(ema_vals, int): # only maintain one EMA
        ema_vals = [ema_vals]
    ema_params = {}
    for ema_val in ema_vals:
        if not isinstance(ema_val, float) and not isinstance(ema_val, int):
            raise ValueError("EMA values must be float / int scalars.")
        ema_params[ema_val] = deepcopy(params)
        ema_params[ema_val] = update_ema(
            ema_params[ema_val], params, alpha=0.0
        )

    if "net" in params:
        print_params(params["net"])
    else:
        print_params(params)

    optimizer_type = str(config.training.get("optimizer_type", "muon")).lower()

    if config.eval_only:
        tx = optax.sgd(learning_rate=0.0)
    else:
        if optimizer_type == "adam":
            tx = optax.adam(
                learning_rate=lr_fn,
                b2=config.training.adam_b2,
                b1=config.training.adam_b1,
            )
        elif optimizer_type == "muon":
            tx = optax.contrib.muon(
                learning_rate=lr_fn,
                adam_b2=config.training.adam_b2,
            )
        else:
            raise ValueError(
                f"Unknown training.optimizer_type: {config.training.optimizer_type!r}. "
                "Expected one of: 'adam', 'muon'."
            )

    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        tx=tx,
    )
    return state
