import torch
from peft import get_peft_model_state_dict

def state_dict_delta(a, b):
    """a - b, elementwise, per tensor. Both dicts must share the same keys/shapes."""
    return {k: (a[k].float() - b[k].float()) for k in a.keys()}

def state_dict_add(a, delta):
    return {k: (a[k].float() + delta[k].float()).to(a[k].dtype) for k in a.keys()}

def get_cloned_peft_state(model):
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

def fedavg_aggregate(state_dicts, sizes):
    total = sum(sizes)
    avg_state = {}
    for k in state_dicts[0].keys():
        weighted_sum = sum((sizes[i] / total) * state_dicts[i][k].float() for i in range(len(state_dicts)))
        avg_state[k] = weighted_sum.to(state_dicts[0][k].dtype)
    return avg_state

def state_dict_bytes(state_dict):
    return sum(v.numel() * v.element_size() for v in state_dict.values())