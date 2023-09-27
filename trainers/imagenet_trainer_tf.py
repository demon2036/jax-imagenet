import functools
import os
from functools import partial

import jax
import tensorflow_datasets as tfds
import flax
import flax.jax_utils
from flax.training.common_utils import shard
from tqdm import tqdm
from flax.training import orbax_utils, common_utils

from input_pipeline import create_split
from modules.utils import create_checkpoint_manager, default, load_ckpt, torch_to_jax
from trainers.basic_trainer_tf import Trainer
from modules.state_utils import *

NUM_CLASSES = 1000


def stack_forest(forest):
  """Helper function to stack the leaves of a sequence of pytrees.

  Args:
    forest: a sequence of pytrees (e.g tuple or list) of matching structure
      whose leaves are arrays with individually matching shapes.
  Returns:
    A single pytree of the same structure whose leaves are individually
      stacked arrays.
  """
  stack_args = lambda *args: jnp.stack(args)
  return jax.tree_util.tree_map(stack_args, *forest)

""""

@partial(jax.pmap, axis_name='batch')
def train_step(state: MyTrainState, batch, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch, )
        loss = cross_entropy_loss(logits, labels)
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)

        return loss, (logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits)), grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads, )
    # print(jnp.argmax(on))
    # metric = {"loss": loss, 'delta': jnp.sum(one_hot_labels - logits, axis=1)}
    metrics = compute_metrics(logits, labels)
    return new_state, metrics

"""


def acc_topk(logits, labels, topk=(1,)):
    top = jax.lax.top_k(logits, max(topk))[1].transpose()
    correct = top == labels.reshape(1, -1)
    return [correct[:k].reshape(-1).sum(axis=0) * 100 / labels.shape[0] for k in topk]


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    # top1, top5 = acc_topk(logits, labels, (1, 5))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        # 'top1': top1,
        # 'top5': top5,
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


"""
@partial(jax.pmap, axis_name='batch', )
def train_step(state, batch):
    def loss_fn(params):
      
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            mutable=['batch_stats'],
        )
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, (new_model_state, logits)

    step = state.step

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats']
    )

    return new_state, metrics

"""


@partial(jax.pmap, axis_name='batch')
def train_step(state: MyTrainState, batch):
    def loss_fn(params):
        variable = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(variable, batch['image'], mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)

        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['label'])
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'] if 'batch_stats' in new_model_state else None
    )

    return new_state, metrics


@partial(jax.pmap, axis_name='batch')
def train_step_without_bn(state: MyTrainState, batch):
    def loss_fn(params):
        variables = {'params': params, }
        logits = state.apply_fn(variables, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = jax.lax.pmean(grads, axis_name='batch')
    logits = aux[1]

    metrics = compute_metrics(logits, batch['label'])
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, metrics


@partial(jax.pmap, axis_name='batch', )
def eval_step(state, batch):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    # variables = {'params': state.params, }
    #
    # if state.batch_stats is not None:
    #     variables.update({'batch_stats': state.batch_stats})

    logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)

    # logits, new_model_state = state.apply_fn({'params': state.params}, batch['image'], mutable=['batch_stats'])
    return compute_metrics(logits, batch['label'])


def get_metrics(device_metrics):
    """Helper utility for pmap, gathering replicated timeseries metric data.

  Args:
   device_metrics: replicated, device-resident pytree of metric data,
     whose leaves are presumed to be a sequence of arrays recorded over time.
  Returns:
   A pytree of unreplicated, host-resident, stacked-over-time arrays useful for
   computing host-local statistics and logging.
  """
    # We select the first element of x in order to get a single copy of a
    # device-replicated metric.
    device_metrics = jax.tree_util.tree_map(lambda x: x[0], device_metrics)
    metrics_np = jax.device_get(device_metrics)
    return stack_forest(metrics_np)


cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


class ImageNetTrainer(Trainer):
    def __init__(self,
                 state,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # dataset_builder = tfds.builder('imagenet2012', try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets')
        # dataset_builder = tfds.builder('imagenet2012',try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets' )  # try_gcs=True,data_dir='gs://jtitor-eu/data/tensorflow_datasets'
        # self.dl = create_split(dataset_builder, batch_size=1024, train=True, cache=True)
        self.state = state
        self.template_ckpt = {'model': self.state, 'steps': self.finished_steps}

    def load(self, model_path=None, template_ckpt=None):
        if model_path is not None:
            checkpoint_manager = create_checkpoint_manager(model_path, max_to_keep=1)
        else:
            checkpoint_manager = self.checkpoint_manager

        model_ckpt = default(template_ckpt, self.template_ckpt)
        if len(os.listdir(self.model_path)) > 0:
            model_ckpt = load_ckpt(checkpoint_manager, model_ckpt)
        self.state = model_ckpt['model']
        self.finished_steps = model_ckpt['steps']

    def save(self):
        model_ckpt = {'model': self.state, 'steps': self.finished_steps}
        save_args = orbax_utils.save_args_from_target(model_ckpt)
        self.checkpoint_manager.save(self.finished_steps, model_ckpt, save_kwargs={'save_args': save_args}, force=False)

    def eval(self):
        eval_metrics = []
        print(self.steps_per_eval)
        for _ in range(self.steps_per_eval):  # self.steps_per_eval
            eval_batch = next(self.dl_eval)
            print(eval_batch['image'].shape)
            metrics = eval_step(self.state, eval_batch)
            # print(metrics)
            eval_metrics.append(metrics)
        eval_metrics = common_utils.get_metrics(eval_metrics)
        summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
        print('\n' * 3)
        print(summary)
        print('\n' * 3)

    def test(self):
        #self.state = flax.jax_utils.replicate(self.state)
        eval_metrics = []
        for _ in range(2):  # self.steps_per_eval
            eval_batch = next(self.dl)
            metrics = eval_step(self.state, eval_batch)
            # print(metrics)
            eval_metrics.append(metrics)
        eval_metrics = common_utils.get_metrics(eval_metrics)
        print(eval_metrics)
        summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
        print(summary)
        #self.state = flax.jax_utils.unreplicate(self.state)

    def train(self):
        has_bn=self.state.batch_stats is not None
        self.state = flax.jax_utils.replicate(self.state)

        with tqdm(total=self.total_epoch * self.steps_per_epoch) as pbar:
            for epoch in range(self.total_epoch):
                for _ in range(50):
                    batch = next(self.dl)
                    # x, y = batch['image'],batch['label']
                    # x, y = torch_to_jax(x), torch_to_jax(y)
                    # x, y = shard(x), shard(y)
                    # print(x.shape)
                    if has_bn:
                        self.state, metrics = train_step(self.state, batch)
                    else:
                        self.state, metrics = train_step_without_bn(self.state, batch)
                    for k, v in metrics.items():
                        metrics.update({k: v[0]})
                    pbar.set_postfix(metrics)
                    pbar.update(1)
                    self.finished_steps += 1
                print()

                if (epoch + 1) % 10 == 0:
                    if has_bn:
                        self.state = sync_batch_stats(self.state)
                    self.eval()
                    self.state = flax.jax_utils.unreplicate(self.state)
                    self.save()
                    self.state = flax.jax_utils.replicate(self.state)


if __name__ == "__main__":
    pass
