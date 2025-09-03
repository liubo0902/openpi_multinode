import dataclasses

import jax
import time
import os
MASTER_ADDR = os.environ.get("MASTER_ADDR", None)
MASTER_PORT = os.environ.get("MASTER_PORT", None)

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

def test_with_distributed_torch_dataset():
    
    ### Set up jax distributed env (I am using Nvidia A800)
    if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
        jax.distributed.initialize()
    # Set master addr and port after jax distributed initialization
    if MASTER_ADDR:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
    if MASTER_PORT:
        os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    print(f"total rank = {jax.process_count()}")


    config = _config.get_config("debug")
    model_config = config.model

    loader = _data_loader.create_distributed_torch_data_loader(
        config.data, 
        model_config, 
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        rank=jax.process_index(), 
        world_size=jax.process_count(),
        skip_norm_stats=True, 
        num_workers=2,
        seed=0,
        shuffle=True,
    )
    data_iter = iter(loader)

    for _ in range(10):
        batch = next(data_iter)
        print(f"load one data from rank {jax.process_index()}", flush=True)
        time.sleep(1)

if __name__ == "__main__":
    test_with_distributed_torch_dataset()