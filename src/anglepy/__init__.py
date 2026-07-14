from anglepy import ANGLE, circular_simulators, kernels, metrics

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch'."
        "Use pip install torch, or visit https://pytorch.org/ for installation instructions.")
