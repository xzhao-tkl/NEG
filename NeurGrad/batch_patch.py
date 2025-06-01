import collections
from typing import List, Callable

import torch
import torch.nn as nn


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def set_attribute_recursive(x: nn.Module, attributes: "str", new_attribute: nn.Module):
    """
    Given a list of period-separated attributes - set the final attribute in that list to the new value
    i.e set_attribute_recursive(model, 'transformer.encoder.layer', NewLayer)
        should set the final attribute of model.transformer.encoder.layer to NewLayer
    """
    for attr in attributes.split(".")[:-1]:
        x = getattr(x, attr)
    setattr(x, attributes.split(".")[-1], new_attribute)


def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Gets the feedforward layer of a model within the transformer block
    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer


def register_hook(
    model: nn.Module,
    layer_idxs: List[int],
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate"
):
    """
    Registers a forward hook in a pytorch transformer model that applies some function, f, to the intermediate
    activations of the transformer model.

    specify how to access the transformer layers (which are expected to be indexable - i.e a ModuleList) with transformer_layers_attr
    and how to access the ff layer with ff_attrs

    `model`: torch.nn.Module
      a torch.nn.Module
    `layer_idx`: int
      which transformer layer to access
    `f`: Callable
      a callable function that takes in the intermediate activations
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    
    def hook_forward_fn(m, i, o):
        f(o)

    hooks = []
    for layer_idx in layer_idxs:
        ff_layer = get_ff_layer(
            model,
            layer_idx,
            transformer_layers_attr=transformer_layers_attr,
            ff_attrs=ff_attrs,
        )
        hooks.append(ff_layer.register_forward_hook(hook_forward_fn))
    return hooks

def register_hook_by_mask(
    model: nn.Module,
    layer_idxs: List[int],
    mask_idxs: torch.Tensor,
    f: Callable,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    def hook_forward_fn(m, i, o):
        f(o)

    hooks = []
    for layer_idx in layer_idxs:
        ff_layer = get_ff_layer(
            model,
            layer_idx,
            transformer_layers_attr=transformer_layers_attr,
            ff_attrs=ff_attrs,
        )
        hooks.append(ff_layer.register_forward_hook(hook_forward_fn))
    return hooks


class BatchedPatch(torch.nn.Module):
    """
    Patches a torch module to replace/suppress/enhance the intermediate activations
    """

    def __init__(
        self,
        ff_layer: nn.Module,
        mask_idxs: torch.Tensor,
        pos_positions: List[List[int]] = None,
        neg_positions: List[List[int]] = None,
        pos_changes: torch.Tensor = None,
        neg_changes: torch.Tensor = None,
        mode: str = "bienhance",
        change_ratio: float = None,
    ):
        super().__init__()
        self.ff = ff_layer
        self.mask_idxs = mask_idxs
        self.pos_positions = pos_positions
        self.neg_positions = neg_positions
        self.pos_changes = pos_changes
        self.neg_changes = neg_changes
        self.change_ratio = change_ratio
        assert mode in ["bisuppress", "bienhance"], mode
        self.mode = mode
        assert len(self.pos_positions) + len(self.neg_positions) > 0
        assert len(self.pos_changes) == len(self.pos_positions)
        assert len(self.neg_changes) == len(self.neg_positions)
        
        if len(self.pos_changes) > 0:
            assert torch.all(self.pos_changes >= 0), f"The change value must >= 0, but got {self.pos_changes.tolist()}"
        if len(self.neg_changes) > 0:
            assert torch.all(self.neg_changes >= 0), f"The change value must >= 0, but got {self.neg_changes.tolist()}"

    def neuron_modify_value(self, value, change):
        if change != None:
            if change.device != value.device:
                change = change.to(value.device)
            return change * torch.sign(value)
        else:
            return self.change_ratio * value

    def forward(self, x: torch.Tensor):
        x = self.ff(x)
        batch_indexs = torch.arange(x.size(0)).unsqueeze(1)
        mask_indexs = self.mask_idxs.unsqueeze(1)
        if self.mode == "bisuppress":
            if len(self.pos_positions) > 0:
                pos_values = x[batch_indexs, mask_indexs, self.pos_positions]
                x[batch_indexs, mask_indexs, self.pos_positions] -= self.neuron_modify_value(pos_values, self.pos_changes)
            if len(self.neg_positions) > 0:
                neg_values = x[batch_indexs, mask_indexs, self.neg_positions]
                x[batch_indexs, mask_indexs, self.neg_positions] += self.neuron_modify_value(neg_values, self.neg_changes)
        elif self.mode == "bienhance":
            if len(self.pos_positions) > 0:
                pos_values = x[batch_indexs, mask_indexs, self.pos_positions]
                x[batch_indexs, mask_indexs, self.pos_positions] += self.neuron_modify_value(pos_values, self.pos_changes)
            if len(self.neg_positions) > 0:
                neg_values = x[batch_indexs, mask_indexs, self.neg_positions]
                x[batch_indexs, mask_indexs, self.neg_positions] -= self.neuron_modify_value(neg_values, self.neg_changes)
        else:
            raise NotImplementedError
        return x

def patch_ff_layer_batch(
    model: nn.Module,
    mask_idxs: torch.Tensor,
    layer_idx: int = None,
    mode: str = "replace",
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
    pos_neurons: List[List[int]] = None,
    neg_neurons: List[List[int]] = None,
    pos_changes: torch.Tensor = None,
    neg_changes: torch.Tensor = None,    
    change_ratio: float = 1
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations.

    `model`: nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
      which transformer layer to access
    `mask_idx`: int
      the index (along the sequence length) of the activation to replace.
      TODO: multiple indices
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)

    pos_neurons_dict = collections.defaultdict(list)
    neg_neurons_dict = collections.defaultdict(list)
    pos_changes_dict = collections.defaultdict(list)
    neg_changes_dict = collections.defaultdict(list)
    layers = set()
    for i, neuron in enumerate(pos_neurons):
        layer_idx, pos = neuron
        pos_neurons_dict[layer_idx].append(pos)
        pos_changes_dict[layer_idx].append(i) # Save index for temporary
        layers.add(layer_idx)
    
    for i, neuron in enumerate(neg_neurons):
        layer_idx, neg = neuron
        neg_neurons_dict[layer_idx].append(neg)
        neg_changes_dict[layer_idx].append(i) # Save index for temporary
        layers.add(layer_idx)
    
    for layer_idx in layers:
        if layer_idx in pos_changes_dict:
            pos_changes_dict[layer_idx] = pos_changes[pos_changes_dict[layer_idx]]
        if layer_idx in neg_changes_dict:
            neg_changes_dict[layer_idx] = neg_changes[neg_changes_dict[layer_idx]]
            
    layers = sorted(list(layers))
    for layer_idx in layers:
        assert layer_idx < len(transformer_layers)
        ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
        set_attribute_recursive(
            transformer_layers[layer_idx],
            ff_attrs,
            BatchedPatch(
                ff_layer,
                mask_idxs,
                mode=mode,
                change_ratio=change_ratio,
                pos_positions=pos_neurons_dict[layer_idx],
                neg_positions=neg_neurons_dict[layer_idx],
                pos_changes=pos_changes_dict[layer_idx],
                neg_changes=neg_changes_dict[layer_idx],
            ),
        )

def unpatch_ff_layer_batch(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.

    `model`: torch.nn.Module
      a torch.nn.Module [currently only works with HF Bert models]
    `layer_idx`: int
      which transformer layer to access
    `transformer_layers_attr`: str
      chain of attributes (separated by periods) that access the transformer layers within `model`.
      The transformer layers are expected to be indexable - i.e a Modulelist
    `ff_attrs`: str
      chain of attributes (separated by periods) that access the ff block within a transformer layer
    """
    transformer_layers = get_attributes(model, transformer_layers_attr)
    assert layer_idx < len(
        transformer_layers
    ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    assert isinstance(ff_layer, BatchedPatch), "Can't unpatch a layer that hasn't been patched"
    set_attribute_recursive(
        transformer_layers[layer_idx],
        ff_attrs,
        ff_layer.ff,
    )


def unpatch_ff_layers_batch(
    model: nn.Module,
    layer_indices: int,
    transformer_layers_attr: str = "bert.encoder.layer",
    ff_attrs: str = "intermediate",
):
    """
    Calls unpatch_ff_layer for all layers in layer_indices
    """
    for layer_idx in layer_indices:
        unpatch_ff_layer_batch(model, layer_idx, transformer_layers_attr, ff_attrs)
