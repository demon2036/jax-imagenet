import importlib
import json
import jax.numpy as jnp
import numpy as np
import orbax
import yaml
from orbax import checkpoint
import jax


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
    x = np.asarray(x)
    x = jnp.asarray(x)
    return x







def mixup(rng, *things, p=0.1, fold_in=None, n=2, **more_things):
  """Perform mixup https://arxiv.org/abs/1710.09412.

  Args:
    rng: The random key to use.
    *things: further arguments are the arrays to be mixed.
    p: the beta/dirichlet concentration parameter, typically 0.1 or 0.2.
    fold_in: One of None, "host", "device", or "sample". Whether to sample a
      global mixing coefficient, one per host, one per device, or one per
      example, respectively. The latter is usually a bad idea.
    n: with how many other images an image is mixed. Default mixup is n=2.
    **more_things: further kwargs are arrays to be mixed.  See also (internal link)
      for further experiments and investigations.

  Returns:
    A new rng key. A list of mixed *things. A dict of mixed **more_things.
  """
  rng, rng_m = jax.random.split(rng, 2)
  if fold_in == "host":
    rng_m = jax.random.fold_in(rng_m, jax.process_index())
  elif fold_in in ("device", "sample"):
    rng_m = jax.random.fold_in(rng_m, jax.lax.axis_index("batch"))
  ashape = (len(things[0]),) if fold_in == "sample" else (1,)
  alpha = jax.random.dirichlet(rng_m, jnp.array([p]*n), ashape)
  # Sort alpha values in decreasing order. This avoids destroying examples when
  # the concentration parameter p is very small, due to Dirichlet's symmetry.
  alpha = -jnp.sort(-alpha, axis=-1)
  def mix(batch):
    if batch is None: return None  # For call-side convenience!
    def mul(a, b):  # B * BHWC -> B111 * BHWC
      return b * jnp.expand_dims(a, tuple(range(1, b.ndim)))
    return sum(mul(alpha[:, i], jnp.roll(batch, i, axis=0)) for i in range(n))
  return rng, map(mix, things), {k: mix(v) for k, v in more_things.items()}





