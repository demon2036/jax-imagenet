import argparse
from functools import partial

import flax.jax_utils
import jax.random

from experimental.test_rep import switch_to_deploy
from modules.state_utils import create_obj_by_config, create_state_by_config, create_state_by_config2, \
    create_learning_rate_fn
from modules.utils import read_yaml
import os
from jax_smi import initialise_tracking
from trainers.imagenet_trainer_tf import ImageNetTrainer

initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='configs/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)

    print('test')

    trainer = ImageNetTrainer( **config['train'])
    trainer.create_state(state_configs=config['State'])
    trainer.load()
    # trainer.state = flax.jax_utils.replicate(trainer.state)
    # trainer.eval()
    # trainer.state = flax.jax_utils.unreplicate(trainer.state)
    # trainer.state = switch_to_deploy(trainer.state, config)
    # trainer.state = flax.jax_utils.replicate(trainer.state)
    # trainer.eval()

    trainer.train()
    # trainer.test()
