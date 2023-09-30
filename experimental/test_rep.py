import argparse
import jax.random
from modules.state_utils import create_state_by_config2
from modules.utils import read_yaml
import os
from jax_smi import initialise_tracking
import jax.numpy as jnp
from jax import config

# config.update("jax_enable_x64", True)
initialise_tracking()

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'


def check_prefix_leaf(tree):
    if isinstance(tree, dict):
        if any(key.startswith('RepVGGBlock') or key == 'classifier' for key in tree.keys()):
            return True
        else:
            if any(check_prefix_leaf(value) for value in tree.values()):
                return True
    else:
        return False


def padding_1x1_to_3x3(kernel_1x1):
    # print(kernel_1x1.shape)
    # padding = ((1, 1), (1, 1), (0, 0), (0, 0))
    # return jnp.pad(kernel_1x1, padding, constant_values=0)

    # kernel_1x1=kernel_1x1.reshape()

    padding = ((1, 1), (1, 1), (0, 0), (0, 0))
    return jnp.pad(kernel_1x1, padding, constant_values=0)


def fuse_bn(kernel, bias, mean, var, gamma, beta):
    # print(kernel.shape, bias.shape, mean.shape, var.shape, gamma.shape, beta.shape)
    std = jnp.sqrt(var + 1e-5)

    # fused_kernel = kernel * (gamma / std).reshape(1, 1, 1, -1)
    # fused_bias = bias + beta - mean * gamma / std

    t = (gamma / std).reshape(1, 1, 1, -1)
    return kernel * t, beta - mean * gamma / std

def switch_to_deploy(train_state, config):
    params = train_state.params
    batch_stats = train_state.batch_stats

    def test(path, param, ):
        # print(path)
        name = path[-1].key
        if name == 'classifier':
            return param
        bn_stats = batch_stats[name]
        conv_1x1, conv_3x3, bn_1x1_stats, bn_3x3_stats, bn_1x1, bn_3x3 = param['conv_1x1'], param['conv_3x3'], bn_stats[
            'bn_1x1'], bn_stats['bn_3x3'], param['bn_1x1'], param['bn_3x3']

        # print(conv_1x1['kernel'].shape)
        fused_kernel_1x1, fused_bias_1x1 = fuse_bn(conv_1x1['kernel'], 0, bn_1x1_stats['mean'],
                                                   bn_1x1_stats['var'], bn_1x1['scale'], bn_1x1['bias'])

        # print(fused_kernel_1x1.shape)
        fused_kernel_3x3, fused_bias_3x3 = fuse_bn(conv_3x3['kernel'], 0, bn_3x3_stats['mean'],
                                                   bn_3x3_stats['var'], bn_3x3['scale'], bn_3x3['bias'])

        if 'identity_bn' in param:
            bn_identity_stats, bn_identity = bn_stats['identity_bn'], param['identity_bn']
            identity_kernel = jnp.zeros_like(fused_kernel_3x3)

            for i in range(identity_kernel.shape[-1]):
                identity_kernel = identity_kernel.at[1, 1, i, i].set(1)

            # print(identity_kernel)

            fused_kernel_identity, fused_bias_identity = fuse_bn(identity_kernel, 0, bn_identity_stats['mean'],
                                                                 bn_identity_stats['var'], bn_identity['scale'],
                                                                 bn_identity['bias'])
            # print(fused_kernel_identity)
        else:
            fused_kernel_identity, fused_bias_identity = 0, 0
        # print(fused_bias_identity, fused_bias_identity)
        # print(fused_kernel_3x3.shape, padding_1x1_to_3x3(fused_kernel_1x1).shape)
        conv_deploy = {'kernel': fused_kernel_3x3 + padding_1x1_to_3x3(fused_kernel_1x1) + fused_kernel_identity,
                       'bias': fused_bias_1x1 + fused_bias_3x3 + fused_bias_identity
                       }

        return {'conv_deploy': conv_deploy}

    params = jax.tree_util.tree_map_with_path(test, params, is_leaf=lambda x: not check_prefix_leaf(x))

    config['State']['Model']['params']['deploy'] = True
    deploy_state = create_state_by_config2(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                           state_configs=config['State'])

    # print(params)
    return deploy_state.replace(params=params, batch_stats=batch_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', default='../configs/test.yaml')
    args = parser.parse_args()
    print(args)
    config = read_yaml(args.config_path)

    train_state = create_state_by_config2(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                          state_configs=config['State'])
    batch_stats = train_state.batch_stats

    # batch_stats = jax.tree_util.tree_map(temp, batch_stats)

    shape = (1, 224, 224, 3)
    params = train_state.params
    key = jax.random.PRNGKey(42)
    dummy_x = jax.random.normal(key, shape)
    # print(params)
    # print(dummy_x)
    # print('\n' * 10)
    dummy_x = jnp.ones(shape) * 2

    out1 = train_state.apply_fn({'params': params, 'batch_stats': train_state.batch_stats}, dummy_x, train=False,
                                mutable=False)

    """
    
    params = jax.tree_util.tree_map_with_path(test, params, is_leaf=lambda x: not check_prefix_leaf(
        x))  # lambda x: not check_prefix_leaf(x))

    config['State']['Model']['params']['deploy'] = True
    train_state = create_state_by_config2(rng=jax.random.PRNGKey(seed=config['train']['seed']),
                                          state_configs=config['State'])
    """
    train_state = switch_to_deploy(train_state, config)

    out = train_state.apply_fn({'params': train_state.params}, dummy_x, )
    print(out.shape)

    # print(out - out1)
    print(out - out1)

    # jax.tree_util.tree_map_with_path(test, params, is_leaf=lambda x: not check_prefix_leaf(x))

    # print(train_state)
