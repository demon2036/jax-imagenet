import importlib
import json
import jax.numpy as jnp
import numpy as np
import orbax
import yaml
from orbax import checkpoint


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def read_yaml(config_path):
    with open(config_path, 'r') as f:
        res = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(res, indent=5))
        return res


def create_checkpoint_manager(save_path, max_to_keep=10, ):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        save_path, orbax_checkpointer, options)
    return checkpoint_manager


def load_ckpt(checkpoint_manager: orbax.checkpoint.CheckpointManager, model_ckpt):
    step = checkpoint_manager.latest_step()
    print(f'load ckpt {step}')
    raw_restored = checkpoint_manager.restore(step, items=model_ckpt)
    return raw_restored


def get_obj_from_str(string: str):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)


def torch_to_jax(x):
    x = x.numpy()
    x = jnp.asarray(x)
    return x
