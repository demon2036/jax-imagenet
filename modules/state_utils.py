import optax
from flax.training.train_state import TrainState
from typing import Any
import jax.numpy as jnp
import jax
from modules.utils import get_obj_from_str, default
from functools import partial


class MyTrainState(TrainState):
    batch_stats: Any = None
    ema_params: Any = None


def create_state(rng, model_cls, input_shapes, train_state, print_model=True, optimizer_dict=None, batch_size=1,
                 model_kwargs=None, ):
    model = model_cls(**model_kwargs)

    inputs = list(map(lambda shape: jnp.empty(shape), input_shapes))

    if print_model:
        print(model.tabulate(rng, *inputs, z_rng=rng, depth=1, console_kwargs={'width': 200}))

    variables = model.init(rng, *inputs, z_rng=rng)
    optimizer = get_obj_from_str(optimizer_dict['optimizer'])

    args = tuple()
    if 'clip_norm' in optimizer_dict and optimizer_dict['clip_norm']:
        args += (optax.clip_by_global_norm(1),)

    optimizer_dict['optimizer_configs']['learning_rate'] *= batch_size
    print(optimizer_dict['optimizer_configs']['learning_rate'])

    args += (optimizer(**optimizer_dict['optimizer_configs']),)
    tx = optax.chain(
        *args
    )
    return train_state.create(apply_fn=model.apply,
                              params=variables['params'],
                              tx=tx,
                              batch_stats=variables['batch_stats'] if 'batch_stats' in variables.keys() else None,
                              )


def create_obj_by_config(config):
    assert 'target', 'params' in config
    obj = get_obj_from_str(config['target'])
    params = config['params']
    return obj(**params)


def create_state_by_config(rng, print_model=True, state_configs={}):
    inputs = list(map(lambda shape: jnp.empty(shape), state_configs['Input_Shape']))
    model = create_obj_by_config(state_configs['Model'])

    if print_model:
        print(model.tabulate(rng, *inputs, z_rng=rng, depth=1, console_kwargs={'width': 200}))
    variables = model.init(rng, *inputs, z_rng=rng)

    args = tuple()
    args += (create_obj_by_config(state_configs['Optimizer']),)
    tx = optax.chain(
        *args
    )
    train_state = get_obj_from_str(state_configs['target'])

    return train_state.create(apply_fn=model.apply,
                              params=variables['params'],
                              tx=tx,
                              batch_stats=variables['batch_stats'] if 'batch_stats' in variables.keys() else None,
                              ema_params=variables['params'])


def create_learning_rate_fn(
        base_learning_rate: float = 0.1,
        steps_per_epoch: int = 1250,
        warmup_epochs=0,
        num_epochs=90
):
    print(base_learning_rate,warmup_epochs,num_epochs)
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def create_state_by_config2(rng, print_model=True, state_configs={}, lr_fn=None):
    inputs = list(map(lambda shape: jnp.empty(shape), state_configs['Input_Shape']))
    model = create_obj_by_config(state_configs['Model'])

    if print_model:
        print(model.tabulate(rng, *inputs, z_rng=rng, depth=1, console_kwargs={'width': 200}))
    variables = model.init(rng, *inputs, z_rng=rng)

    if lr_fn is None:
        lr_fn = create_learning_rate_fn(base_learning_rate=state_configs['Optimizer']['params']['learning_rate'])
    else:
        lr_fn = lr_fn()

    # print(learning_rate_fn,create_learning_rate_fn)
    #
    learning_rate_fn = lr_fn
    state_configs['Optimizer']['params']['learning_rate'] = learning_rate_fn

    args = tuple()
    args += (create_obj_by_config(state_configs['Optimizer']),)
    tx = optax.chain(
        optax.clip_by_global_norm(0.01),
        *args,
        # optax.clip_by_global_norm(1),
    )

    train_state = get_obj_from_str(state_configs['target'])

    return train_state.create(apply_fn=model.apply,
                              params=variables['params'],
                              tx=tx,
                              batch_stats=variables['batch_stats'] if 'batch_stats' in variables.keys() else None,
                              ema_params=variables['params'])
