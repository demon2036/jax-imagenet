import os
from functools import partial

import flax
import flax.jax_utils
from flax.training.common_utils import shard
from tqdm import tqdm
from flax.training import orbax_utils, common_utils

from modules.utils import create_checkpoint_manager, default, load_ckpt, torch_to_jax
from trainers.basic_trainer import Trainer
from modules.state_utils import *

NUM_CLASSES = 1000


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)

    assert logits.shape == labels.shape

    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)

    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


@partial(jax.pmap, axis_name='batch')
def train_step(state: MyTrainState, x, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = cross_entropy_loss(logits, labels)

        one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)

        return loss, (logits, one_hot_labels)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, one_hot_labels)), grads = grad_fn(state.params)
    #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
    grads = jax.lax.pmean(grads, axis_name='batch')

    new_state = state.apply_gradients(grads=grads)
    # loss = jax.lax.pmean(loss, axis_name='batch')

    # one_hot_labels = one_hot_labels[0]
    # logits = logits[0]

    # print(jnp.argmax(on))

    # metric = {"loss": loss, 'delta': jnp.sum(one_hot_labels - logits, axis=1)}
    metrics = compute_metrics(logits, labels)
    return new_state, metrics


class ImageNetTrainer(Trainer):
    def __init__(self,
                 state,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
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

        with tqdm(total=100000) as pbar:
            for epoch in range(self.total_epoch):
                for batch in self.dl:
                    x, y = batch
                    x, y = torch_to_jax(x), torch_to_jax(y)
                    x, y = shard(x), shard(y)
                    # print(x.shape)
                    state, metrics = train_step(state, x, y)
                    for k, v in metrics.items():
                        metrics.update({k: v[0]})
                    pbar.set_postfix(metrics)
                    pbar.update(1)

                    if (epoch + 1) % 10 == 0:
                        self.eval()


if __name__ == "__main__":
    model = create_state_by_config()
