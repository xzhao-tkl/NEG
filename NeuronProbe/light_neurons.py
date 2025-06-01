import os
import torch
import numpy as np

from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from typing import List
from collections import Counter

from NeuronProbe import LIGHT_NEURONS_CACHE_ROOT
from NeuronProbe.base_dataset import BaseDataset
from NeuronProbe.grad_neurons import SkillNeuronProbe
from NeuronProbe.utils import dump_json, load_json

class LightNeuronProbe(SkillNeuronProbe):
    """
    A class for collecting data for neg-based neuron probing.
    This class is designed to work with the super large LLMs, such as Llama2-70B,
    It is a lightweight version of the NeuronProbe framework."""
    
    def __init__(
            self, 
            dataset: BaseDataset,
            model_name: str,
            model_type: str,
            skip_model: bool=False,
            device: int=-1) -> None:
        super().__init__(dataset, model_name, model_type, skip_model, device)
        self.cache_dir = self.cache_dir
        self.this_cache_dir = os.path.join(
            LIGHT_NEURONS_CACHE_ROOT, 
            self.model_name,
            self.dataset.dataset_name, 
            self.dataset.prompt_type, 
            f"{self.dataset.n_shot}-shot_{self.dataset.balanced}_sampling")
        os.makedirs(self.this_cache_dir, exist_ok=True)
    
    def load_light_ingradients(
            self, 
            prompt_size, 
            batch_size = 8, 
            split = "train", 
            recache: bool=False):
        
        ds = self.dataset.ds[split]
        prompt_size = len(ds) if prompt_size == -1 else min(len(ds), prompt_size)

        gt_fn, prob_fn, grad_polar_fn, grad_value_fn = self.__find_light_cache_files(split, prompt_size)
        if all([os.path.exists(fn) for fn in [gt_fn, prob_fn, grad_polar_fn, grad_value_fn]]) and not recache:
            print(f"Load light neuron ingradients from cache: {gt_fn}")
            gts = load_json(gt_fn)[:prompt_size]
            probs = torch.load(prob_fn, weights_only=True)[:prompt_size]
            polars = torch.load(grad_polar_fn, weights_only=True)[:prompt_size]
            values = torch.load(grad_value_fn, weights_only=True)[:prompt_size]
        else:
            ds = ds.select(np.arange(prompt_size))
            gts = ds['_answer']

            probs_batches = []
            labels, label_idxs = self.get_label_idxs()
            polars = torch.zeros((self.n_layers, self.intermediate_size))
            values = torch.zeros((self.n_layers, self.intermediate_size))
            for j in tqdm(range(0, prompt_size, batch_size), total=int(prompt_size/batch_size)):
                gts_b = ds['_answer'][j:j+batch_size]
                prompts = ds['_prompt'][j:j+batch_size]
                try:    
                    probs_b, grads_b, _ = self.ngrad.get_gradients_multilabels(
                        prompts=prompts, 
                        multi_labels=[labels] * len(prompts),
                        act_threshold=0)
                    probs_batches.append(probs_b[:, torch.tensor(label_idxs)].squeeze(2))
                    
                    ## Calculate polars
                    grad_polars = torch.sign(grads_b)
                    gt_polars = self.get_gt_scores(grad_polars, gts_b)
                    polars += (2 * gt_polars - grad_polars.sum(1)).sum(dim=0)

                    ## Calculate values
                    gt_scores = self.get_gt_scores(grads_b, gts_b)
                    _values = torch.sign(gt_scores.unsqueeze(1) - grads_b).sum(dim=1)
                    values += _values.sum(dim=0)
                finally:
                    if self.ngrad is not None:
                        self.ngrad.baseline_activations = []
                        torch.cuda.empty_cache()
                
            probs = torch.vstack(probs_batches)
            dump_json(gts, gt_fn)
            torch.save(probs, prob_fn)
            torch.save(polars, grad_polar_fn)
            torch.save(values, grad_value_fn)
        return gts, probs, polars, values

    def rank_neurons_by_grads_polar(
            self, 
            neuron_polars: torch.Tensor, 
            topk: int, 
            use_neg: bool):
        
        neuron_polars = neuron_polars.flatten()
        if use_neg:
            topk_neurons = neuron_polars.abs().topk(topk).indices
            neuron_polarity = torch.sign(neuron_polars[topk_neurons])
        else:
            topk_neurons = neuron_polars.topk(topk).indices
            neuron_polarity = torch.ones_like(topk_neurons)
        return topk_neurons, neuron_polarity
    
    def rank_neurons_by_grads_volume(
            self,
            larger_neurons: torch.Tensor, 
            topk: int, 
            use_neg: bool):
        larger_neurons = larger_neurons.flatten()
        
        if use_neg:
            topk_neurons = larger_neurons.abs().topk(topk).indices
            neuron_polarity = torch.sign(larger_neurons[topk_neurons])
        else:
            topk_neurons = larger_neurons.topk(topk).indices
            neuron_polarity = torch.ones_like(topk_neurons)
        return topk_neurons, neuron_polarity
    

    def get_skill_neurons(
            self,
            gts: List[str], 
            probs: torch.Tensor, 
            grads: torch.Tensor, 
            acts: torch.Tensor,
            topk: int=10,
            score_type: str="grad",
            rank_by: str="grad_polar",
            verbose: bool=True,
            use_neg: bool=True):
        
        assert score_type in ["grad", "act", "gact"]

        if score_type == "grad":
            scores_tensor = grads
        elif score_type == "act":
            scores_tensor = acts
        elif score_type == "gact":
            scores_tensor = grads * acts.unsqueeze(1)
        
        if rank_by == "grad_polar":
            (
                skill_neuron_pred_idx, 
                topk_neurons, 
                neuron_polarity
            ) = self.rank_neurons_by_grads_polar(
                scores_tensor, gts, topk, use_neg
            )
        elif rank_by == "grad_value":
            (
                skill_neuron_pred_idx, 
                topk_neurons, 
                neuron_polarity
            ) = self.rank_neurons_by_grads_volume(
                scores_tensor, gts, topk, use_neg
            )
        else:
            raise NotImplementedError
                
        if verbose:
            prob_pred_idx = probs.argmax(dim=1)
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            self.logger.info("----------------Train set--------------------")
            layers = (topk_neurons % self.n_layers).tolist()
            self.logger.info(f"Top Neuron distribution : {Counter(layers)}")
            self.logger.info(f"Accuracy by output prob : {(gt_idx == prob_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Accuracy by skill neuron: {(gt_idx == skill_neuron_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Ground-Truth answer distri: {Counter(gt_idx.tolist())}")
            self.logger.info(f"Prob-based   answer distri: {Counter(prob_pred_idx.tolist())}")
            self.logger.info(f"Neuron-based answer distri: {Counter(skill_neuron_pred_idx.tolist())}")

        return skill_neuron_pred_idx, topk_neurons, neuron_polarity

    
    def __find_light_cache_files(self, split, prompt_size):
        cand_fns = []
        for fn in os.listdir(self.this_cache_dir):
            if fn.startswith(f"{split}_"):
                this_size = int(fn.split('_')[1])
                if this_size > prompt_size:
                    cand_fns.append(fn)
        gt_fn = os.path.join(self.this_cache_dir, f"{split}_{prompt_size}_groud_truth.json")
        prob_fn = os.path.join(self.this_cache_dir, f"{split}_{prompt_size}_probs.json")
        grad_polar_fn = os.path.join(self.this_cache_dir, f"{split}_{prompt_size}_grad_polar.pt")
        grad_value_fn = os.path.join(self.this_cache_dir, f"{split}_{prompt_size}_grad_value.pt")
        return gt_fn, prob_fn, grad_polar_fn, grad_value_fn