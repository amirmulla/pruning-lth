import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import prune


######################################
# Weight Init functions.             #
######################################

def save_model_weights(model):
    init_params = []
    for name, param in model.named_parameters():
        init_params.append(param.data.clone())
    return init_params


def rewind_model_weights(model, init_params):
    i = 0
    for _, param in model.named_parameters():
        param.data = init_params[i].clone()
        i += 1


######################################
# Pruning functions.                 #
######################################

def remove_pruning(model):
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            prune.remove(layer, name='weight')
        if isinstance(layer, nn.Conv2d):
            prune.remove(layer, name='weight')


def init_prune_model(model):
    if prune.is_pruned(model):
        remove_pruning(model)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            prune.identity(layer, name='weight')
        if isinstance(layer, nn.Conv2d):
            prune.identity(layer, name='weight')


def get_idx_of_output_layer(model):
    o_layer_idx = None
    i = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            o_layer_idx = i
        i += 1
    return o_layer_idx


def prune_model(model, prune_ratio=0.2, prune_method='local', prune_output_layer=True):
    o_layer_idx = get_idx_of_output_layer(model)
    i = 0

    if prune_method == 'local':
        for layer in model.children():
            if i == o_layer_idx:
                if prune_output_layer:
                    prune.l1_unstructured(layer, name='weight', amount=prune_ratio / 2)
            else:
                if isinstance(layer, nn.Linear):
                    prune.l1_unstructured(layer, name='weight', amount=prune_ratio)
                if isinstance(layer, nn.Conv2d):
                    prune.l1_unstructured(layer, name='weight', amount=prune_ratio)
            i += 1

    elif prune_method == 'global':
        parameters_to_prune = []
        for layer in model.children():
            if i == o_layer_idx:
                if prune_output_layer:
                    parameters_to_prune.append(tuple([layer, 'weight']))
            else:
                if isinstance(layer, nn.Linear):
                    parameters_to_prune.append(tuple([layer, 'weight']))
                if isinstance(layer, nn.Conv2d):
                    parameters_to_prune.append(tuple([layer, 'weight']))
            i += 1

        prune.global_unstructured(tuple(parameters_to_prune), pruning_method=prune.L1Unstructured, amount=prune_ratio)


######################################
# Sparsity calculation functions.    #
######################################

def calc_model_sparsity(model):
    zero_weights = 0.0
    n_elements = 0.0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            zero_weights += torch.sum(layer.weight_mask == 0)
            n_elements += layer.weight.nelement()
        if isinstance(layer, nn.Conv2d):
            zero_weights += torch.sum(layer.weight_mask == 0)
            n_elements += layer.weight.nelement()

    return 100. * zero_weights / n_elements


def print_sparsity(model):
    print("Model sparsity: {:.2f}%".format(calc_model_sparsity(model)))
