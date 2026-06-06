# init_worker.py
import torch


def dask_setup(dask_worker):
    # set threads at worker startup
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
