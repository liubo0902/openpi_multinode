import sys
import os
import torch
cudas = torch.cuda.device_count()
world_size = int(os.environ.get("WORLD_SIZE", 1)) * cudas
rank_start = int(os.environ.get("RANK", 0)) * cudas
run_cmd = ' '.join(sys.argv[1:])
for i in range(1, cudas):
    os.system(f"export CUDA_VISIBLE_DEVICES={i} && export WORLD_SIZE={world_size} && export RANK={rank_start + i} && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && nohup {run_cmd} >> rank_{i}.out &")
os.system(f"export CUDA_VISIBLE_DEVICES={0} && export WORLD_SIZE={world_size} && export RANK={rank_start} && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && {run_cmd}")