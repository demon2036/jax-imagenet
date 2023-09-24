import os
from functools import partial
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
    #top1, top5 = acc_topk(logits, labels, (1, 5))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        # 'top1': top1,
        # 'top5': top5,
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics

"""
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


@partial(jax.pmap, axis_name='batch')
def train_step(state: MyTrainState, batch):
    def loss_fn(params):
        logits, new_model_state = state.apply_fn({'params': params}, batch['image'], mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)

        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'] if 'batch_stats' in new_model_state else None )
    # print(jnp.argmax(on))
    # metric = {"loss": loss, 'delta': jnp.sum(one_hot_labels - logits, axis=1)}
    metrics = compute_metrics(logits, batch['label'])
    return new_state, metrics







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
        pass

    def train(self):
        state = flax.jax_utils.replicate(self.state)

        with tqdm(total=1000000) as pbar:
            for epoch in range(self.total_epoch):
                for batch in self.dl:
                    #x, y = batch['image'],batch['label']
                    #x, y = torch_to_jax(x), torch_to_jax(y)
                    #x, y = shard(x), shard(y)
                    # print(x.shape)
                    state, metrics = train_step(state, batch, y)
                    for k, v in metrics.items():
                        metrics.update({k: v[0]})
                    pbar.set_postfix(metrics)
                    pbar.update(1)
            print()
            if (epoch + 1) % 10 == 0:
                self.eval()


if __name__ == "__main__":
   pass
