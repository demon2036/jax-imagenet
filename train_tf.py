import argparse

import flax.jax_utils
import jax.random
from modules.state_utils import create_obj_by_config, create_state_by_config, create_state_by_config2
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

    train_state = create_state_by_config2(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                          state_configs=config['State'])

    trainer = ImageNetTrainer(train_state, **config['train'])

    trainer.load()
    trainer.eval()
    # trainer.train()
