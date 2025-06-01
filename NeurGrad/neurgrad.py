import einops
import random
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from functools import partial
from typing import List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from .batch_patch import unpatch_ff_layers_batch, patch_ff_layer_batch
from .batch_patch import register_hook, register_hook_by_mask, get_attributes


class NeurGrad:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str,
        device: int = -1,
    ):
        self.model = model
        self.model_type = model_type
        self.device = "cuda" if device < 0 else f"cuda:{device}"
        self.tokenizer = tokenizer

        self.baseline_activations = None
        self.layerwise_activations = None

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.ff_attr = "intermediate"
        elif self.model_type in ["llama2", "llama3", "qwen"]:
            self.transformer_layers_attr = "model.layers"
            self.ff_attr = "mlp.up_proj"
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported")
        
    def calculate_scores(self, acts, grads, act_threshold):
        acts = (acts > act_threshold).type(torch.int8) * 2 - 1
        signs = torch.sign(acts)
        return grads * signs

    def _move_to_same_device(self, tensors: List[torch.Tensor], device: str="cpu") -> List[torch.Tensor]:
        tensors = [tensor.to(device) for tensor in tensors]
        return tensors
        
    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def encode_prompt(self, prompt):
        if isinstance(prompt, list) or isinstance(prompt, tuple):
            encoded_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        elif isinstance(prompt, str):
            encoded_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        encoded_inputs = {k: v.to(self.model.device) for k, v in encoded_inputs.items()}
        return encoded_inputs
        
    def encode_token(self, target):
        if self.model_type == "llama2":
            token_ids = self.tokenizer.encode(target)
            assert len(token_ids) == 2 and token_ids[0] == 1, f"Target {target} should be a single token, but got {token_ids}"
            return [token_ids[1]]
        elif self.model_type == "llama3":
            token_ids = self.tokenizer.encode(target)
            assert len(token_ids) == 2 and token_ids[0] == 128000, f"Target {target} should be a single token, but got {token_ids}"
            return [token_ids[1]]
        elif self.model_type == "qwen":
            token_ids = self.tokenizer.encode(target)
            assert len(token_ids) == 1, f"Target {target} should be a single token, but got {token_ids}"
            return [token_ids[1]]
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported for encoding tokens")

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.encode_prompt(prompt)
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        else:
            mask_idx = -1
        
        target_tokens = [self.encode_token(target) if target is not None else None]
        return encoded_input, mask_idx, target_tokens

    def tokenize_prompts(self, prompts, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.encode_prompt(prompts)
        if self.model_type == "bert":
            mask_idxs = torch.where(encoded_input["input_ids"] == self.tokenizer.mask_token_id)[1]
        else:
            mask_idxs = torch.ones(len(prompts), dtype=torch.int64).to(self.device) * -1
        return encoded_input, mask_idxs

    def tokenize_labels(self, labels):
        return torch.tensor([self.encode_token(label) for label in labels])
        
    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type in ["bert", "llama2", "llama3", "qwen"]:
            return self.model.config.intermediate_size
        else:
            raise NotImplementedError()

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        )
        return out
        
    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idxs=[layer_idx],
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.ff_attr
            )[0]

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        prompt: str,
        label: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = False,
        act_threshold: float = 0,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `label`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert isinstance(prompt, str) and isinstance(label, str)
        if attribution_method == "cgrad":
            _, scores, _ = self.get_neg_per_prompt(
                prompt=prompt, 
                label=label, 
                act_threshold=act_threshold,
                gc=True)
        elif attribution_method == "gradient":
            _, scores, _ = self.get_neg_per_prompt(
                prompt=prompt, 
                label=label, 
                act_threshold=act_threshold,
                gc=False)
        elif attribution_method == "integrated_grads":
            scores = self.get_integrated_grads_per_prompt(
                prompt=prompt, 
                label=label,
                batch_size=batch_size, 
                steps=steps, 
                attribution_method=attribution_method, 
                pbar=pbar
            )
        else:
            raise NotImplementedError(f"{attribution_method} must be either `gradient` or `integrated_grads`")
        return scores

    def get_pos_neg_neurons(self, neuron_values: torch.Tensor, neuron_indices: torch.Tensor):
        assert len(neuron_values.shape) == 1 and len(neuron_indices.shape) == 2, \
            f"Type: {type(neuron_values), type(neuron_indices)}, Shape: {neuron_values.shape, neuron_indices.shape}"
        pos_neurons, neg_neurons = [], []
        for value, indice in zip(neuron_values.tolist(), neuron_indices.tolist()):
            if value > 0:
                pos_neurons.append(indice)
            else:
                neg_neurons.append(indice)
        return pos_neurons, neg_neurons 

    def get_sampled_neurons(
            self,
            attribution_scores: torch.Tensor,
            num_neurons: int,
            threshold: float = 0,
            seed: int = 42,
        ) ->Tuple[float, float, float]:
        assert threshold >= 0

        all_neurons = []
        for scores in attribution_scores:
            try:
                rows, cols = torch.where(scores.abs() > threshold)
                neurons_per_prompt = list(zip(rows.tolist(), cols.tolist()))
                assert len(neurons_per_prompt) >= num_neurons, f"neurons_per_prompt: {neurons_per_prompt}, num_neurons: {num_neurons}" 
                random.seed(seed)
                neurons = random.sample(neurons_per_prompt, k=num_neurons)
            except Exception as e:
                print("Failed to sample enough neurons above threshold, use random neurons instead")
                rows = [random.randint(0, attribution_scores.shape[0]-1) for i in range(num_neurons)]
                cols = [random.randint(0, attribution_scores.shape[1]-1) for i in range(num_neurons)]
                neurons = list(zip(rows, cols))
            neurons = torch.tensor([list(neuron) for neuron in neurons])
            all_neurons.append(neurons)
        all_neurons = torch.stack(all_neurons)
        rows = all_neurons[:, :, 0]
        cols = all_neurons[:, :, 1]
        batch_indices = torch.arange(len(attribution_scores)).reshape(len(attribution_scores), 1).expand(-1, num_neurons)
        all_scores = attribution_scores[batch_indices, rows, cols]
        return all_scores.cpu(), all_neurons.cpu()

    def get_topk_neurons(
            self,
            attribution_scores: torch.Tensor,
            topk: int = 100, 
            abs_val: bool = False,
            reverse: bool = False,
        ) ->Tuple[float, float, float]:
        
        def process_score(scores):
            if abs_val:
                assert reverse is False, f"reverse must be set as None if abs_val is True, but get {reverse}"
                return scores.abs()
            return scores
        
        scores = process_score(attribution_scores)
        num_prompts, num_layers, num_neurons = scores.shape
        _, topk_indices_flat = scores.flatten(start_dim=1).topk(topk, largest=not reverse, dim=-1)
        
        rows = topk_indices_flat // num_neurons
        cols = topk_indices_flat % num_neurons

        # topk_indices.shape = (num_prompts, topk_neurons, 2)
        topk_neurons = torch.stack((rows, cols), dim=-1)
        rows = topk_neurons[:, :, 0]
        cols = topk_neurons[:, :, 1]
        batch_indices = torch.arange(num_prompts).reshape(num_prompts, 1).expand(-1, topk)
        topk_values = attribution_scores[batch_indices, rows, cols]
        return topk_values.cpu(), topk_neurons.cpu()
    
    def get_cumlative_neurons(
            self, 
            attribution_scores: torch.Tensor,
            cumulative_threshold: float, 
            return_value_indices: bool=False):
        assert len(attribution_scores.shape) == 2
        rows, cols = attribution_scores.shape
        values, indices = attribution_scores.cpu().flatten().abs().sort(descending=True)
        cumulated_abs_score = attribution_scores.abs().sum().item()
        threshold = cumulated_abs_score * cumulative_threshold
        cumulative_sum = torch.cumsum(values, dim=0)
        cutoff_index = torch.searchsorted(cumulative_sum, threshold, right=True)
        selected_indices = indices[:cutoff_index].cpu()
        row_indices = (selected_indices // cols).tolist()
        col_indices = (selected_indices % cols).tolist()

        if return_value_indices:
            top_neurons = list(zip(row_indices, col_indices))
            top_values = attribution_scores[(selected_indices // cols), (selected_indices % cols)]
            return top_values, top_neurons
        else:
            pos_neurons, neg_neurons = [], []
            for row, col in zip(row_indices, col_indices):
                if attribution_scores[row][col] >=0:
                    pos_neurons.append([row, col])
                else:
                    neg_neurons.append([row, col])
            return pos_neurons, neg_neurons

    def get_neg_per_prompt(
            self, 
            prompt: str, 
            label: str, 
            act_threshold: float=0,
            gc: bool=False): 
        self.baseline_activations = []
        encoded_input = self.encode_prompt(prompt)
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, label, encoded_input)
        
        def get_activations(model, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations.append(acts)

            return register_hook(
                model,
                layer_idxs=list(range(self.n_layers())),
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.ff_attr
            )

        handles = get_activations(self.model, mask_idx=mask_idx)
        outputs = self.model(**encoded_input, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)[:, target_label]
        raw_grads = torch.autograd.grad(
            outputs=probs, 
            inputs=self.baseline_activations,
            grad_outputs=torch.ones_like(probs),
            retain_graph=True)
        
        raw_grads = self._move_to_same_device(raw_grads) 
        raw_grads = torch.stack(raw_grads)[:, :, mask_idx, :]
        acts = self._move_to_same_device(self.baseline_activations)
        acts = torch.stack(acts)[:, :, mask_idx, :]
        
        if gc is False:
            grads = self.calculate_scores(acts, raw_grads, act_threshold).transpose(0, 1)
        else:
            grads = raw_grads.transpose(0, 1)
        
        for handle in handles:
            handle.remove()
        self.baseline_activations = []
        return probs.squeeze(1).detach().cpu(), grads.detach().cpu(), acts.detach().cpu()
    
    def get_negs_in_batch(
            self, 
            prompts: list[str], 
            labels: list[str], 
            act_threshold: float=0, 
            return_cg: float=False):
        self.baseline_activations = []
        encoded_inputs = self.encode_prompt(prompts)
        encoded_inputs, mask_idxs = self.tokenize_prompts(prompts, encoded_inputs)
        target_labels = self.tokenize_labels(labels)
        
        def get_activations(model, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations.append(acts)

            return register_hook_by_mask(
                model,
                layer_idxs=list(range(self.n_layers())),
                mask_idxs=mask_idxs,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.ff_attr
            )
    
        handles = get_activations(self.model, mask_idx=mask_idxs)
        outputs = self.model(**encoded_inputs)
        
        # Define shapes
        num_prompts, num_tokens, num_vocabs = outputs.logits.shape
        assert len(mask_idxs.shape) == 1
        assert len(prompts) == num_prompts and num_prompts == len(mask_idxs)
        
        # Indexing the probing position ([MASK] tokens for mask LMs, -1 for casual LMs)
        token_idxs = mask_idxs.unsqueeze(1).unsqueeze(2) ## (num_prompts --> (num_prompts, 1, 1))
        token_idxs = token_idxs.expand(num_prompts, 1, num_vocabs)
        if "bert" in self.model_type:
            logits = torch.gather(input=outputs.logits, dim=1, index=token_idxs) ## (num_prompts, 1, num_vocabs)
        elif self.model_type in ["llama2", "llama3", "qwen"]:
            logits = outputs.logits[:, -1, :].unsqueeze(1)
        else:
            raise NotImplementedError()
        probs = F.softmax(logits, dim=-1) ## (num_prompts, 1, num_vocabs)
        
        # Indexing the probing tokens
        gt_probs = probs[torch.arange(len(prompts)), 0, target_labels] ## (num_prompts, )
        raw_grads = torch.autograd.grad(
            outputs=gt_probs, 
            inputs=self.baseline_activations, 
            grad_outputs=torch.ones_like(gt_probs), 
            retain_graph=True) # (num_prompts, num_layers, num_tokens, num_neurons)
        
        raw_grads = self._move_to_same_device(raw_grads) 
        raw_grads = torch.stack(raw_grads).transpose(0, 1)
        
        # Index activation and gradients tensors with mask_idx at 3rd dimension    
        raw_grads = raw_grads[torch.arange(num_prompts).unsqueeze(1), torch.arange(self.n_layers()), mask_idxs.cpu().unsqueeze(1), :]
        acts = self._move_to_same_device(self.baseline_activations)
        acts = torch.stack(acts).transpose(0, 1) # (num_prompts, num_layers, num_tokens, num_neurons)
        acts = acts[torch.arange(num_prompts).unsqueeze(1), torch.arange(self.n_layers()), mask_idxs.cpu().unsqueeze(1), :]

        if return_cg:
            grads = raw_grads
        else:                
            grads = self.calculate_scores(acts, raw_grads, act_threshold)
        
        for handle in handles:
            handle.remove()
        self.baseline_activations = []
        
        return probs.squeeze(1).detach().cpu(), grads.cpu(), acts.detach().cpu()
        
    def get_negs_multilabels(
            self, 
            prompts: List[str], 
            multi_labels: List[List[str]], 
            act_threshold: float=0):
        self.baseline_activations = []
        encoded_inputs = self.encode_prompt(prompts)
        encoded_inputs, mask_idxs = self.tokenize_prompts(prompts, encoded_inputs)
        target_labels = torch.stack([self.tokenize_labels(labels) for labels in multi_labels])
        target_labels = target_labels.reshape(target_labels.size(0), len(multi_labels[0]))
        assert target_labels.size(0) == len(prompts)

        def get_activations(model, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations.append(acts)

            return register_hook_by_mask(
                model,
                layer_idxs=list(range(self.n_layers())),
                mask_idxs=mask_idxs,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.ff_attr
            )

        handles = get_activations(self.model, mask_idx=mask_idxs)
        outputs = self.model(**encoded_inputs, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        
        # Define shapes
        num_prompts, num_tokens, num_vocabs = outputs.logits.shape
        assert len(mask_idxs.shape) == 1
        assert len(prompts) == num_prompts and num_prompts == len(mask_idxs), f"num_prompts: {num_prompts}, len(prompts): {len(prompts)}, len(mask_idxs): {len(mask_idxs)}"
        
        # Indexing the probing position ([MASK] tokens for mask LMs, -1 for casual LMs)
        token_idxs = mask_idxs.unsqueeze(1).unsqueeze(2) ## (num_prompts --> (num_prompts, 1, 1))
        token_idxs = token_idxs.expand(num_prompts, 1, num_vocabs)
        if "bert" in self.model_type:
            logits = torch.gather(input=outputs.logits, dim=1, index=token_idxs) ## (num_prompts, 1, num_vocabs)
        elif self.model_type in ["llama2", "llama3", "qwen"]:
            logits = outputs.logits[:, -1, :].unsqueeze(1)
        probs = F.softmax(logits, dim=-1) ## (num_prompts, 1, num_vocabs)
        
        # Indexing the probing tokens
        gt_probs = probs[torch.arange(len(prompts)).unsqueeze(1), 0, target_labels] ## (num_prompts, num_labels)

        raw_multi_grads = []
        for i in range(gt_probs.size(1)):
            grads = torch.autograd.grad(
                outputs=gt_probs[:, i], 
                inputs=self.baseline_activations, 
                grad_outputs=torch.ones_like(gt_probs[:, i]), 
                retain_graph=True) # (num_prompts, num_layers, num_tokens, num_neurons)
            grads = self._move_to_same_device(grads) 
            grads = torch.stack(grads).transpose(0, 1)
            indexed_grads = grads[torch.arange(num_prompts).unsqueeze(1), torch.arange(self.n_layers()), mask_idxs.cpu().unsqueeze(1), :]
            raw_multi_grads.append(indexed_grads)
        raw_multi_grads = torch.stack(raw_multi_grads)

        # Index activation tensors with mask_idx at 3rd dimension
        acts = self._move_to_same_device(self.baseline_activations)
        acts = torch.stack(acts).transpose(0, 1) # (num_prompts, num_layers, num_tokens, num_neurons)
        acts = acts[torch.arange(num_prompts).unsqueeze(1), torch.arange(self.n_layers()), mask_idxs.cpu().unsqueeze(1), :]

        multi_grads = self.calculate_scores(acts, raw_multi_grads, act_threshold).transpose(0, 1)
                
        for handle in handles:
            handle.remove()
        self.baseline_activations = []
        return probs.squeeze(1).detach().cpu(), multi_grads.cpu(), acts.detach().cpu()
        
    def get_generation_info(self, prompts, labels, return_prob: bool=False):
        results = []
        encoded_inputs, mask_idxs = self.tokenize_prompts(prompts)
        token_idxs = self.tokenize_labels(labels)

        assert len(token_idxs) == len(mask_idxs), \
            f"We only support len(target_label)==1, but get label: {labels}"
        
        data_indexs = torch.arange(len(prompts)).unsqueeze(1) # Used for indexing
        outputs = self.model(**encoded_inputs, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        probs = F.softmax(outputs.logits[data_indexs, mask_idxs.unsqueeze (1), :], dim=-1).squeeze(1)
        
        gt_probs = probs[data_indexs, token_idxs.unsqueeze(1)].squeeze(1).detach().cpu()
        argmax_probs, argmax_ids = probs.max(dim=1)
        argmax_probs = argmax_probs.detach().cpu()
        argmax_strs = [self.tokenizer.decode([token_id]) for token_id in argmax_ids]
        probs = probs.detach().cpu()

        for prompt, gt, gt_prob, argmax_prob, argmax_str, prob in zip(prompts, labels, gt_probs, argmax_probs, argmax_strs, probs):
            results.append({
                "prompt": prompt,
                "gt": gt,
                "gt_prob": gt_prob.item(),
                "argmax_completion": argmax_str,
                "argmax_prob": argmax_prob.item(),
                "probs": prob if return_prob else None
            }
            )
        return results

    def modify_activations(
        self,
        prompts: List[str],
        labels: List[str],
        pos_neurons: List[List[int]],
        neg_neurons: List[List[int]],
        pos_changes: torch.Tensor = None,
        neg_changes: torch.Tensor = None,
        mode: str = "bienhance",
        change_ratio: float = None,
        change_volume: float = None,
        return_prob: bool = None,
    ) -> Tuple[dict, Callable]:
        _, mask_idx, = self.tokenize_prompts(prompts)
        assert len(prompts) == len(labels) and len(labels) == len(mask_idx)
        old_results = self.get_generation_info(prompts, labels, return_prob)
        
        all_layers = set([n[0] for n in pos_neurons]).union([n[0] for n in neg_neurons])
        
        if change_volume is not None:
            assert pos_changes is None and neg_changes is None
            pos_changes = (torch.ones(size=(len(pos_neurons), )) * change_volume).to(self.device)
            neg_changes = (torch.ones(size=(len(neg_neurons), )) * change_volume).to(self.device)

        patch_ff_layer_batch(
            self.model,
            mask_idxs=mask_idx.cpu(),
            mode=mode,
            pos_neurons=pos_neurons,
            neg_neurons=neg_neurons,
            pos_changes=pos_changes,
            neg_changes=neg_changes,
            change_ratio=change_ratio,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_results = self.get_generation_info(prompts, labels, return_prob)
        
        results_dicts = []
        for old_result, new_result in zip(old_results, new_results):
            results_dict = {}
            results_dict["before"] = old_result
            results_dict["after"] = new_result
            results_dicts.append(results_dict)

        unpatch_fn = partial(
            unpatch_ff_layers_batch,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.ff_attr,
        )
        
        unpatch_fn()
        unpatch_fn = lambda *args: args

        return results_dicts, unpatch_fn
    
    def suppress_knowledge(
        self,
        prompts: List[str],
        labels: List[str],
        pos_neurons: List[List[int]],
        neg_neurons: List[List[int]] = None,
        change_ratio: float = 1,
        change_volume: float = None,
        return_prob: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `pos_neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        if change_volume:
            assert change_ratio is None, "change_ratio has to explicited to zero when change_volume is not None"
        return self.modify_activations(
            prompts=prompts,
            labels=labels,
            pos_neurons=pos_neurons,
            neg_neurons=neg_neurons,
            change_ratio=change_ratio,
            change_volume=change_volume,
            mode="bisuppress",
            return_prob=return_prob,
        )

    def enhance_knowledge(
        self,
        prompts: List[str],
        labels: List[str],
        pos_neurons: List[List[int]],
        neg_neurons: List[List[int]] = None,
        change_ratio: float = 1,
        change_volume: float = None,
        return_prob: bool = False
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `pos_neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        if change_volume:
            assert change_ratio is None, "change_ratio has to explicited to zero when change_volume is not None"
        
        return self.modify_activations(
            prompts=prompts,
            labels=labels,
            pos_neurons=pos_neurons,
            neg_neurons=neg_neurons,
            change_ratio=change_ratio,
            change_volume=change_volume,
            mode="bienhance",
            return_prob=return_prob,
        )