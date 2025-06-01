import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm as std_tqdm
from typing import List
from collections import Counter
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from NeuronProbe import GRANEUR_NEURONS_CACHE_ROOT
from NeuronProbe.base_dataset import BaseDataset
from NeuronProbe.utils import dump_json, load_json

from NeurGrad.neurgrad import NeurGrad
from NeurGrad import INTERMIDIATE_SIZE, MODEL_LAYERS, initialize_model_and_tokenizer


tqdm = partial(std_tqdm, dynamic_ncols=True)

def chunked_argmax(tensor):
    chunk_size = 100
    results = []

    for i in tqdm(range(0, tensor.size(0), chunk_size)):
        chunk = tensor[i:i + chunk_size]
        chunk_result = torch.argmax(chunk, dim=1)
        results.append(chunk_result)

    result = torch.cat(results, dim=0)
    return result

def dtree_classification(train_x, train_y, test_x, test_y, n_trees=100, n_layers=None):
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, max_depth=n_layers)
    rf.fit(train_x, train_y)

    final_predictions = np.array(rf.predict(test_x))
    
    accuracy = accuracy_score(test_y, final_predictions)
    print(f"n_trees: {n_trees}, n_layers: {n_layers}, Accuracy: {accuracy}")
    return rf

class SkillNeuronProbe:
    def __init__(
            self, 
            dataset: BaseDataset,
            model_name: str,
            model_type: str,
            skip_model: bool=False,
            device: int=-1) -> None:
        """
        Initalize the class for collecting data for gradident-based neuron probing. 
        `dataset`: BaseDataset
            The dataset instance for probing. 
        `model_name`: str
            The LLMs for probing
        `model_type`: str
            To specify the tokenizer type. GraNeur restricts the probed generation to a token size of 1.
            This binds the probing procedure to the tokenizer type.
        `skip_model`: bool
        """
        self.dataset = dataset
        self.labels = dataset.labels
        self.model_type = model_type
        self.model_name = model_name
        if model_type.startswith("bert"):
            assert device >= 0
        
        if skip_model:
            _, tokenizer = initialize_model_and_tokenizer(model_name, gpu=device, only_tokenizer=True)
            self.n_layers = MODEL_LAYERS[model_name]
            self.intermediate_size = INTERMIDIATE_SIZE[model_name]
            self.ngrad = None
        else:
            model, tokenizer = initialize_model_and_tokenizer(model_name, gpu=device)
            self.ngrad = NeurGrad(model, tokenizer, device=device, model_type=model_type)
            self.n_layers = self.ngrad.n_layers()
            self.intermediate_size = self.ngrad.intermediate_size()
        
        self.tokenizer = tokenizer
        self.cache_dir = os.path.join(
            GRANEUR_NEURONS_CACHE_ROOT, 
            self.model_name,
            self.dataset.dataset_name, 
            self.dataset.prompt_type, 
            f"{self.dataset.n_shot}-shot_{self.dataset.balanced}_sampling")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = self.dataset.logger
    
    def get_label_idxs(self):
        assert self.labels is not None
        label_idxs = [self.ngrad.encode_token(label) for label in self.labels]
        return self.labels, label_idxs
    
    def measure_accuracy(self, pred_idx):
        assert self.labels is not None
        _, label_idxs = self.ngrad.tokenize_labels(self.labels)
        return ((label_idxs == pred_idx).type(torch.int8).sum() / len(pred_idx)).item()

    def parse_indices_to_neurons(self, indices): 
        """
        Given a list of flatten indice of neurons, parse them into (row, col) style. 
        """
        neurons = []
        for indice in indices:
            neurons.append((indice // self.ngrad.intermediate_size(), indice % self.ngrad.intermediate_size()))
        return neurons
    
    def get_gt_idx(self, gts: List[str]):
        """
        Given the ground-truth labels, return an array of index among all labels.
        """
        return torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
    
    def get_gt_scores(self, grads: torch.Tensor, gts: List[str]):
        """ 
        Given the gradients tensor for all labels, return the gradients for only the ground-truth label
        Inputs:
            - grads: torch.Tensor, with shape [num_labels, num_prompts, -1]
            - gts: a list of ground-truth labels
        Output:
            - The gradient tensor of ground-truth labels with the shape [num_prompts, -1]
        """
        gt_idx = self.get_gt_idx(gts)
        return grads[torch.arange(gt_idx.size(0)).unsqueeze(1), gt_idx.unsqueeze(1)].squeeze(1)
    
    def generate(self, prompts, max_length=10):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.ngrad.model.generation_config.temperature=None
        self.ngrad.model.generation_config.top_p=None
        
        input_ids = self.tokenizer(
            prompts, 
            padding=True, truncation=True,
            return_tensors="pt",
            return_attention_mask=True).input_ids.to(self.ngrad.model.device)
        generated_ids = self.ngrad.model.generate(
            input_ids, 
            max_new_tokens=max_length, 
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        new_generated_texts = [gen[len(prompt):].strip() for gen, prompt in zip(generated_texts, prompts)]
        return new_generated_texts
        
    def identify_neurons_polarity(self, neuron_indices, grads, gts):
        """
        Given the flatten neuron indices and the gradient value for each indice, 
        Return the neurons in [row, col] style and classify the positive/negative neurons.
        `neuron_indices`: torch.Tensor
            [num_neurons, ]
        """
        gt_grads = self.get_gt_scores(grads, gts)
        gt_grads = gt_grads.reshape(gt_grads.size(0), -1)
        topk_grads = gt_grads[:, neuron_indices]
        pos_cnt = (topk_grads > 0).sum(dim=0)
        neg_cnt = (topk_grads < 0).sum(dim=0)
        is_pos_list = (pos_cnt > neg_cnt).tolist()

        pos_neurons, neg_neurons = [], []
        for indice, is_pos in zip(neuron_indices.tolist(), is_pos_list):
            if is_pos:
                pos_neurons.append(self.parse_indices_to_neurons([indice])[0])
            else:
                neg_neurons.append(self.parse_indices_to_neurons([indice])[0])
        return pos_neurons, neg_neurons
    
    def get_generations(
            self, 
            prompt_size, 
            split = "train", 
            recache: bool=False,
            skip_load: bool=False):
        
        ds = self.dataset.ds[split]
        prompt_size = len(ds) if prompt_size == -1 else min(len(ds), prompt_size)
        ds = ds.select(np.arange(prompt_size))
        prompts = ds['_prompt']
        generations = []
        for prompt in prompts:
            generation = self.generate(prompt)
            generations.append(generation)
        return generations

    def load_generation(
            self, 
            prompt_size, 
            batch_size = 8, 
            split = "train", 
            recache: bool=False,
            skip_load: bool=False):
        
        ds = self.dataset.ds[split]
        prompt_size = len(ds) if prompt_size == -1 else min(len(ds), prompt_size)

        _, gen_fn, _, _, _ = self._find_cache_files(split, prompt_size)
        if os.path.exists(gen_fn) and not recache:
            if not skip_load:
                all_generations = load_json(gen_fn)[:prompt_size]
            else:
                all_generations = None
        else:
            ds = ds.select(np.arange(prompt_size))
            gts = ds['_answer']
            all_generations = []
            for j in tqdm(range(0, prompt_size, batch_size), total=int(prompt_size/batch_size)):
                prompts = ds['_prompt'][j:j+batch_size]
                all_generations.extend(self.generate(prompts))
            dump_json(all_generations, gen_fn)
        return all_generations
            
    def load_neurons_ingradients(
            self, 
            prompt_size, 
            batch_size = 8, 
            split = "train", 
            recache: bool=False,
            skip_load: bool=False):
        
        ds = self.dataset.ds[split]
        prompt_size = len(ds) if prompt_size == -1 else min(len(ds), prompt_size)

        gt_fn, gen_fn, grads_fn, probs_fn, acts_fn = self._find_cache_files(split, prompt_size)
        if all([os.path.exists(fn) for fn in [gt_fn, grads_fn, probs_fn, acts_fn]]) and not recache:
            print(f"Load neuron ingradients from cache: {gt_fn}")
            if not skip_load:
                gts = load_json(gt_fn)[:prompt_size]
                if gen_fn is not None and os.path.exists(gen_fn):
                    all_generations = load_json(gen_fn)[:prompt_size]
                else:
                    all_generations = None
                if probs_fn is not None and os.path.exists(probs_fn):
                    probs = torch.load(probs_fn, weights_only=True)[:prompt_size]
                else:
                    probs = None
                grads = torch.load(grads_fn, weights_only=True)[:prompt_size]
                acts = torch.load(acts_fn, weights_only=True)[:prompt_size]
            else:
                gts, all_generations, probs, grads, acts = None, None, None, None, None
        else:
            ds = ds.select(np.arange(prompt_size))
            gts = ds['_answer']
            if gts[0].isdigit():
                key_map = {'1': "a", '2': "b", '3': "c", '4': "d", '5': "e"}
                gts = [key_map[gt] for gt in gts]

            labels, label_idxs = self.get_label_idxs()
            grads_batches, probs_batches, acts_batches = [], [], []
            all_generations = []
            for j in tqdm(range(0, prompt_size, batch_size), total=int(prompt_size/batch_size)):
                prompts = ds['_prompt'][j:j+batch_size]
                all_generations.extend(self.generate(prompts))
                try:    
                    probs_b, grads_b, acts_b = self.ngrad.get_negs_multilabels(
                        prompts=prompts, 
                        multi_labels=[labels] * len(prompts),
                        act_threshold=0)
                    probs_batches.append(probs_b[:, torch.tensor(label_idxs)].squeeze(2))
                    grads_batches.append(grads_b)
                    acts_batches.append(acts_b)
                finally:
                    self.ngrad.baseline_activations = []
                    torch.cuda.empty_cache()
            
            probs = torch.vstack(probs_batches)
            grads = torch.vstack(grads_batches)
            acts = torch.vstack(acts_batches)
            torch.save(probs, probs_fn)
            torch.save(grads, grads_fn)
            torch.save(acts, acts_fn)
            dump_json(all_generations, gen_fn)
        return gts, all_generations, probs, grads, acts

    def rank_neurons_by_grads_polar(
            self, 
            scores: torch.Tensor, 
            gts: List[str], 
            topk: int, 
            use_neg: bool):
        """
        Rank neurons' importance to reflect the task knowledge based on the polarity of scores tensor. 
        Motivation: 
            if a neuron appears more as positive/negative neurons to ground-truth label, 
            then the polarity of this neuron could be as indicator of knowledge
        `scores`: torch.Tensor
            The scores could be either `gradients`, `gradients x activations`. 
            It should contains the score of prompts to all the labels, with the shape [num_prompts, num_labels, -1]
        `gts`: List[str]
            The array of ground-truth labels
        `topk`: int
            Specify how many neurons to return
        `use_neg`: bool
            Decide whether consider neurons with negative scores

        Return: 
        `pred_idx`: torch.Tensor
            The array of prediction based on the identified skill neurons' value. 
        `topk_neurons`: torch.Tensor
            A one-dimension array of topk neurons' indices.
        `neuron_polarity`: torch.Tensor
            A one-dimension array of topk neurons' polarity. 
            The value is either 1 (positive neuron) or -1 (negative neuron)
        """
        score_polars = torch.sign(scores).reshape(scores.size(0), scores.size(1), -1)
        gt_polars = self.get_gt_scores(score_polars, gts)

        residue_polars = (2 * gt_polars - score_polars.sum(1)).sum(dim=0)
        if use_neg:
            topk_neurons = residue_polars.abs().topk(topk).indices
            neuron_polarity = torch.sign(residue_polars[topk_neurons])
            adjusted_polars = score_polars[:, :, topk_neurons] * neuron_polarity.unsqueeze(0).unsqueeze(0)
            pred_idx = adjusted_polars.sum(dim=2).argmax(dim=1)
        else:
            topk_neurons = residue_polars.topk(topk).indices
            pred_idx = score_polars[:, :, topk_neurons].sum(dim=2).argmax(dim=1)
            neuron_polarity = torch.ones_like(topk_neurons)
        return pred_idx, topk_neurons, neuron_polarity
    
    def rank_neurons_by_grads_volume(
            self,
            scores: torch.Tensor, 
            gts: List[str], 
            topk: int, 
            use_neg: bool):
        """
        Rank neurons' importance to reflect the task knowledge based on the volume of scores. 
        Motivation: 
            if a neuron's grads to ground-truth label are larger/smaller than others,
            then the volume of this neuron could be as indicator of knowledge

        `scores`: torch.Tensor
            The scores could be either `gradients`, `activations` or `gradients x activations`. 
            It should contains the score of prompts to all the labels, with the shape [num_prompts, num_labels, -1]
        `gts`: List[str]
            The array of ground-truth labels
        `topk`: int
            Specify how many neurons to return
        `use_neg`: bool
            Decide whether consider neurons with negative scores

        Return: 
        `pred_idx`: torch.Tensor
            The array of prediction based on the identified skill neurons' value. 
        `topk_neurons`: torch.Tensor
            A one-dimension array of topk neurons' indices.
        `neuron_polarity`: torch.Tensor
            A one-dimension array of topk neurons' polarity. 
            The value is either 1 (positive neuron) or -1 (negative neuron)
        """
        gt_scores = self.get_gt_scores(scores, gts)
        gtscore_larger_than_others = torch.sign(gt_scores.unsqueeze(1) - scores).sum(dim=1).reshape(gt_scores.size(0), -1)
        gtscore_larger_than_others = gtscore_larger_than_others.sum(dim=0)
        
        if use_neg:
            topk_neurons = gtscore_larger_than_others.abs().topk(topk).indices
            neuron_polarity = torch.sign(gtscore_larger_than_others[topk_neurons])
        else:
            topk_neurons = gtscore_larger_than_others.topk(topk).indices
            neuron_polarity = torch.ones_like(topk_neurons)
        scores_flatten = scores.reshape(scores.size(0), scores.size(1), -1)
        adjusted_grads = scores_flatten[:, :, topk_neurons] * neuron_polarity.unsqueeze(0).unsqueeze(0)
        pred_idx = adjusted_grads.sum(dim=2).argmax(dim=1)
        return pred_idx, topk_neurons, neuron_polarity
    
    def rank_neuron_by_acts_values(
            self,
            acts: torch.Tensor, 
            gts: List[str], 
            topk: int,
            gold_labels: list[str]):
        
        label_counts = Counter(gts)
        num_classes = len(label_counts)

        # Binary classification
        if num_classes == 2:
            gts = self.get_gt_idx(gts)
            return self._binary_rank_neuron_by_acts_values(acts, gts, topk)
        # Multi-class classification
        else:
            pred_strs, topk_neurons_dt, neuron_polarity_dt, neuron_thres_dt = self._multiclass_rank_neuron_by_acts_values(acts, gts, topk)
            pred_idx = self.convert_pred_to_tensor(pred_strs, gold_labels)
            return pred_idx, topk_neurons_dt, neuron_polarity_dt, neuron_thres_dt

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
        
        neuron_thres = None
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
        elif rank_by == "act_value":
            (
                skill_neuron_pred_idx, 
                topk_neurons, 
                neuron_polarity,
                neuron_thres
            ) = self.rank_neuron_by_acts_values(
                acts, gts, topk, self.labels
            )
        else:
            raise NotImplementedError
        
        if isinstance(topk_neurons, dict):
            topk_neurons_tensor = []
            for neurons in topk_neurons.values():
                topk_neurons_tensor.extend(neurons)
            topk_neurons_tensor = torch.tensor(topk_neurons_tensor)
        else:
            topk_neurons_tensor = topk_neurons

        if verbose:
            prob_pred_idx = probs.argmax(dim=1)
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            self.logger.info("----------------Train set--------------------")
            layers = (topk_neurons_tensor % self.n_layers).tolist()
            self.logger.info(f"Top Neuron distribution : {Counter(layers)}")
            self.logger.info(f"Accuracy by output prob : {(gt_idx == prob_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Accuracy by skill neuron: {(gt_idx == skill_neuron_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Ground-Truth answer distri: {Counter(gt_idx.tolist())}")
            self.logger.info(f"Prob-based   answer distri: {Counter(prob_pred_idx.tolist())}")
            self.logger.info(f"Neuron-based answer distri: {Counter(skill_neuron_pred_idx.tolist())}")

        return skill_neuron_pred_idx, topk_neurons, neuron_polarity, neuron_thres

    def test_skill_neurons(
            self, 
            gts: List[str],
            probs: torch.Tensor, 
            grads: torch.Tensor, 
            acts: torch.Tensor,
            topk_neurons: torch.Tensor,
            neuron_thres: torch.Tensor,
            neuron_polarity: torch.Tensor,
            score_type: str="grad",
            rank_by: str="grad_polar",
            major_vote: bool=True,
            roc_plot: bool=True,
            verbose: bool=True):

        if score_type == "grad":
            scores = grads
        elif score_type == "act":
            scores = acts
        elif score_type == "gact":
            scores = grads * acts.unsqueeze(1).shape
                    
        if rank_by == 'grad_polar':
            score_polars = torch.sign(scores).reshape(scores.size(0), scores.size(1), -1)
            adjusted_scores = score_polars[:, :, topk_neurons] * neuron_polarity.unsqueeze(0).unsqueeze(0)
            pred_idx = adjusted_scores.sum(dim=2).argmax(dim=1)
        elif rank_by == 'grad_value':
            scores_flatten = scores.reshape(scores.size(0), scores.size(1), -1)
            adjusted_scores = scores_flatten[:, :, topk_neurons] * neuron_polarity.unsqueeze(0).unsqueeze(0)
            if major_vote:
                adjusted_scores = (adjusted_scores - adjusted_scores.max(dim=1).values.unsqueeze(1)) == 0
            pred_idx = adjusted_scores.sum(dim=2).argmax(dim=1)
        elif rank_by == 'act_value':
            pred_idx, adjusted_scores = self.predict_idx(gts, scores, topk_neurons, neuron_thres, neuron_polarity, self.labels)
        else:
            raise NotImplementedError
        
        if isinstance(topk_neurons, dict):
            topk_neurons_tensor = []
            for neurons in topk_neurons.values():
                topk_neurons_tensor.extend(neurons)
            topk_neurons_tensor = torch.tensor(topk_neurons_tensor)
        else:
            topk_neurons_tensor = topk_neurons

        if verbose:
            prob_pred_idx = probs.argmax(dim=1)
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            self.logger.info("----------------Test set--------------------")
            self.logger.info(f"Accuracy by output prob : {(gt_idx == prob_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Accuracy by skill neuron: {(gt_idx == pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Ground-Truth answer distri: {Counter(gt_idx.tolist())}")
            self.logger.info(f"Prob-based   answer distri: {Counter(prob_pred_idx.tolist())}")
            self.logger.info(f"Neuron-based answer distri: {Counter(pred_idx.tolist())}")

        if roc_plot:
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            self.roc_skill_neurons(
                gt_idx,
                adjusted_scores,
                title=f'ROC Curve of {self.dataset.dataset_name} (dataset) by {topk_neurons.size(0)} skill neurons')
        return pred_idx
    
    def get_skill_randomforest(
            self,
            gts: List[str], 
            probs: torch.Tensor, 
            grads: torch.Tensor, 
            acts: torch.Tensor,
            n_estimators: int=100,
            use_act: bool=False,
            verbose: bool=True):
        if use_act:
            dtree_cache_fn = os.path.join(self.cache_dir, f"dtree_acts_{n_estimators}.pkl")
        else:
            dtree_cache_fn = os.path.join(self.cache_dir, f"dtree_grads_{n_estimators}.pkl")
        
        x_features, y_labels = self.generate_dtree_x_y(acts, grads, gts, use_act=use_act)
        if not os.path.exists(dtree_cache_fn):
            self.logger.info(f"Training dtree to cache {dtree_cache_fn}")
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=None)
            rf.fit(x_features, y_labels)
            with open(dtree_cache_fn, 'wb') as f:
                pickle.dump(rf, f)
        else:
            self.logger.info(f"Loading dtree from cache {dtree_cache_fn}")
            with open(dtree_cache_fn, 'rb') as f:
                rf = pickle.load(f)

        self.logger.info(f"Predicting using dtree")
        final_predictions = np.array(rf.predict(x_features))
        max_accuracy = accuracy_score(y_labels, final_predictions)
        
        # import_indexes = None
        # if not (n_features is None and n_estimators == 100 and n_layers is None):
        #     importances = rf.feature_importances_
        #     import_indexes = np.argsort(importances)[-n_features:]
        #     x_features_partial = x_features[:, import_indexes]
        
        #     rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=n_layers)
        #     rf.fit(x_features_partial, y_labels)
        
        #     final_predictions = np.array(rf.predict(x_features_partial))
        #     accuracy = accuracy_score(x_features_partial, final_predictions)
            
        if verbose:
            prob_pred_idx = probs.argmax(dim=1)
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            source = "activations" if use_act else "gradients"
            self.logger.info(f"----------------Train set using {source}--------------------")
            self.logger.info(f"Accuracy by output prob : {(gt_idx == prob_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Accuracy by full random-forest: {max_accuracy}")
            # if not (n_features is None and n_estimators == 100 and n_layers is None):
            #     self.logger.info(f"Accuracy by partial random-forest: {accuracy}")
        return rf, max_accuracy

    def test_skill_randomforest(
            self,
            rf: RandomForestClassifier,
            gts: List[str], 
            probs: torch.Tensor, 
            grads: torch.Tensor, 
            acts: torch.Tensor,
            import_indexes: List=None,
            use_act: bool=False,
            verbose: bool=True):
        
        x_features, y_labels = self.generate_dtree_x_y(acts, grads, gts, use_act=use_act)        
        if import_indexes is not None:
            x_features = x_features[:, import_indexes]
        final_predictions = np.array(rf.predict(x_features))
        accuracy = accuracy_score(y_labels, final_predictions)
        if verbose:
            prob_pred_idx = probs.argmax(dim=1)
            gt_idx = torch.tensor([self.labels.index(gt) for gt in gts], dtype=torch.int64)
            source = "activations" if use_act else "gradients"
            self.logger.info(f"----------------Test set using {source}--------------------")
            self.logger.info(f"Accuracy by output prob : {(gt_idx == prob_pred_idx).sum()/len(gt_idx):.4f}")
            self.logger.info(f"Accuracy by random-forest: {accuracy}")
        return accuracy
    
    def roc_skill_neurons(
            self, 
            gt_idxs: torch.Tensor,
            adjusted_scores: torch.Tensor, 
            title: str=None):
        true_signs, false_signs = [], []
        for i, gt_idx in enumerate(gt_idxs):
            for j in range(len(self.labels)):
                if gt_idx == j:
                    true_signs.append(adjusted_scores[i][j])
                else:
                    false_signs.append(adjusted_scores[i][j])
        true_signs = torch.stack(true_signs)
        false_signs = torch.stack(false_signs)
        true_ranks = (true_signs==1).sum(dim=1)
        false_ranks = (false_signs==1).sum(dim=1)
        
        true_labels, pred_scores = [], []
        pred_scores.extend(true_ranks.tolist())
        pred_scores.extend(false_ranks.tolist())
        true_labels.extend([1] * true_ranks.size(0))
        true_labels.extend([0] * false_ranks.size(0))

        fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.show()
    
    def generate_dtree_x_y(self, acts, grads, gts, use_act=False):
        if use_act:
            x_features = acts.reshape(acts.shape[0], -1)
            
        if len(self.dataset.labels) == 2:
            if not use_act:
                flatten_grads = grads.reshape(grads.shape[0], 2, -1)
                x_features = (flatten_grads[:, 0] > flatten_grads[:, 1]).type(torch.int8).numpy()
            y_labels = np.array([int(gt == self.dataset.labels[0]) for gt in gts])
        else:
            if not use_act:
                flatten_grads = grads.reshape(grads.shape[0], grads.shape[1], -1)
                x_features = torch.argmax(flatten_grads, dim=1)
            y_labels = np.array([self.dataset.labels.index(gt) for gt in gts])
        return x_features, y_labels

    def binary_encode(self, items, prefix=""):
        '''
        Encode multi-choices with 0-1 strings.
        Example: 
        >>> items = ['a', 'b', 'c', 'd']
        >>> codes = binary_encode(items)
        >>> print(codes)
        {'a': '00', 'b': '01', 'c': '10', 'd': '11'}
        '''
        n = len(items)
        
        # Base case: if only one item, assign the current prefix as its code
        if n == 1:
            return {items[0]: prefix}

        # Recursive case: split items into two halves
        mid = n // 2
        left_items = items[:mid]
        right_items = items[mid:]
        
        # Recursively encode left and right halves
        codes = {}
        codes.update(self.binary_encode(left_items, prefix + "0"))  # Left half: prefix + "0"
        codes.update(self.binary_encode(right_items, prefix + "1"))  # Right half: prefix + "1"

        return codes

    def convert_pred_to_tensor(self, pred_strs, labels) -> torch.tensor:
        codes = self.binary_encode(labels)
        inverted_dict = {v: k for k, v in codes.items()}
        mapping_dict = inverted_dict
        vectorized_map = np.vectorize(mapping_dict.get)
        mapped_arr = vectorized_map(pred_strs)
        
        if labels[0].isdigit():
            invalid_value = -1
            x = np.array(mapped_arr)
            mapped_x = np.array([int(i)-1 if i.isdigit() else invalid_value for i in x])
            tensor_x = torch.tensor(mapped_x)

        else:
            mapping_dict = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
            invalid_value = -1
            x = np.array(mapped_arr)
            mapped_x = np.array([mapping_dict.get(char, invalid_value) for char in x])
            tensor_x = torch.tensor(mapped_x)
        return tensor_x
    
    def _find_cache_files(self, split, prompt_size):
        cand_fns = []
        for fn in os.listdir(self.cache_dir):
            if fn.startswith(f"{split}_"):
                this_size = int(fn.split('_')[1])
                if this_size > prompt_size:
                    cand_fns.append(fn)
        # assert len(cand_fns) == 5 or len(cand_fns) == 0

        probs_fn = None
        if len(cand_fns) == 5:
            for fn in cand_fns:
                if fn.endswith('_groud_truth.json'):
                    gt_fn = os.path.join(self.cache_dir, fn)
                elif fn.endswith('_grads.pt'):
                    grads_fn = os.path.join(self.cache_dir, fn)
                elif fn.endswith('_probs.pt'):
                    probs_fn = os.path.join(self.cache_dir, fn)
                elif fn.endswith('_acts.pt'):
                    acts_fn = os.path.join(self.cache_dir, fn)
                elif fn.endswith('_generations.pt'):
                    acts_fn = os.path.join(self.cache_dir, fn)
        else:
            gt_fn = os.path.join(self.cache_dir, f"{split}_{prompt_size}_groud_truth.json")
            generation_fn = os.path.join(self.cache_dir, f"{split}_{prompt_size}_generations.json")
            grads_fn = os.path.join(self.cache_dir, f"{split}_{prompt_size}_grads.pt")
            probs_fn = os.path.join(self.cache_dir, f"{split}_{prompt_size}_probs.pt")
            acts_fn = os.path.join(self.cache_dir, f"{split}_{prompt_size}_acts.pt")
            
        return gt_fn, generation_fn, grads_fn, probs_fn, acts_fn

    def _binary_rank_neuron_by_acts_values(
            self,
            acts: torch.Tensor, 
            gts: List[str], 
            topk: int):
        """
        Original rank_neuron_by_acts_values function.
        """
        
        gt_idx = torch.tensor(gts)
        acts = acts.reshape(acts.size(0), -1)
        bsl_acts = acts.mean(dim=0)
        largerthan_bsl = torch.sign((acts - bsl_acts.unsqueeze(0)) > 0)
        match_gt_label = (largerthan_bsl.int() - gt_idx.unsqueeze(1)) == 0
        neuron_acc = match_gt_label.sum(dim=0)/largerthan_bsl.size(0)
        relative_neuron_acc = neuron_acc - 0.5

        topk_neurons = relative_neuron_acc.abs().topk(topk).indices
        neuron_polarity = (relative_neuron_acc[topk_neurons] > 0).int() * 2 - 1
        neuron_thres = bsl_acts[topk_neurons]
        
        skill_neuron_val = acts[:, topk_neurons].reshape(acts.size(0), -1)
        indictor = ((skill_neuron_val - neuron_thres.unsqueeze(0)) > 0)
        indictor = indictor.int() * 2 - 1
        indictor = indictor * neuron_polarity.unsqueeze(0)
        pos_indicator = (indictor == 1).sum(dim=1)
        neg_indicator = (indictor == -1).sum(dim=1)
        pred_idx = (pos_indicator > neg_indicator).int()
        return pred_idx, topk_neurons, neuron_polarity, neuron_thres

    def _multiclass_rank_neuron_by_acts_values(
            self,
            acts: torch.Tensor, 
            gts: List[str], 
            topk: int):
        """
        Support Multi-class classification.
        """
    
        topk_neurons_dt = {}
        neuron_polarity_dt = {}
        neuron_thres_dt = {}

        unique_labels = tuple(sorted(list(set(gts))))
        mid_point = len(unique_labels) // 2
        group1 = unique_labels[:mid_point]
        group2 = unique_labels[mid_point:]
        
        # Convert to binary case
        gts_binary = [0 if label in group1 else 1 for label in gts]
        pred_idx, topk_neurons, neuron_polarity, neuron_thres = self._binary_rank_neuron_by_acts_values(acts, gts_binary, topk)
        # Add to dicts
        topk_neurons_dt[tuple(unique_labels)] = topk_neurons
        neuron_polarity_dt[tuple(unique_labels)] = neuron_polarity
        neuron_thres_dt[tuple(unique_labels)] = neuron_thres

        pred_idx_str = pred_idx.numpy().astype(str)

        # Recursion, until done
        if len(group1) > 1:
            selected_idx = (torch.tensor(gts_binary) == 0)
            
            pred_idx1_str, topk_neurons1, neuron_polarity1, neuron_thres1 = self._multiclass_rank_neuron_by_acts_values(
                acts[selected_idx], [gt for gt in gts if gt in group1], topk)
            pred_idx_str[selected_idx] = np.char.add(pred_idx_str[selected_idx], pred_idx1_str)
            topk_neurons_dt.update(topk_neurons1)
            neuron_polarity_dt.update(neuron_polarity1)
            neuron_thres_dt.update(neuron_thres1)
        else:
            pass
        
        if len(group2) > 1:
            selected_idx = (torch.tensor(gts_binary) == 1)
            pred_idx2_str, topk_neurons2, neuron_polarity2, neuron_thres2 = self._multiclass_rank_neuron_by_acts_values(
                acts[selected_idx], [gt for gt in gts if gt in group2], topk)
            pred_idx_str[selected_idx] = np.char.add(pred_idx_str[selected_idx], pred_idx2_str)

            topk_neurons_dt.update(topk_neurons2)
            neuron_polarity_dt.update(neuron_polarity2)
            neuron_thres_dt.update(neuron_thres2)
        else:
            pass
        
        return pred_idx_str, topk_neurons_dt, neuron_polarity_dt, neuron_thres_dt
    
    def binary_pred(self,
                scores: torch.tensor,
                topk_neurons: torch.tensor,
                neuron_thres: torch.tensor,
                neuron_polarity: torch.tensor):
    
        scores = scores.reshape(scores.size(0), -1)
        skill_neuron_val = scores[:, topk_neurons].reshape(scores.size(0), -1)
        indictor = ((skill_neuron_val - neuron_thres.unsqueeze(0)) > 0)
        indictor = indictor.int() * 2 - 1
        indictor = indictor * neuron_polarity.unsqueeze(0)
        pos_indicator = (indictor == 1).sum(dim=1)
        neg_indicator = (indictor == -1).sum(dim=1)
        pred_idx = (pos_indicator > neg_indicator).int()
        adjusted_scores = torch.stack([indictor==-1, indictor==1]).transpose(0, 1)
        return pred_idx, adjusted_scores

    def multi_pred(self, 
                choices: list[str],
                scores: torch.tensor,
                topk_neurons,
                neuron_thres,
                neuron_polarity):
        
        unique_labels = tuple(sorted(list(set(choices))))
        mid_point = len(unique_labels) // 2
        group1 = unique_labels[:mid_point]
        group2 = unique_labels[mid_point:]

        # Convert to binary case
        pred_idx, adjusted_scores = self.binary_pred(scores, topk_neurons=topk_neurons[unique_labels], neuron_thres=neuron_thres[unique_labels], neuron_polarity=neuron_polarity[unique_labels])

        pred_idx_str = pred_idx.numpy().astype(str)

        # Recursion, until done
        if len(group1) > 1:
            # selected_idx = (torch.tensor(gts_binary) == 0)
            selected_idx = (pred_idx == 0)
            pred_idx1_str, adjusted_scores1 = self.multi_pred(group1, 
                scores[selected_idx], topk_neurons, neuron_thres, neuron_polarity)
            pred_idx_str[selected_idx] = np.char.add(pred_idx_str[selected_idx], pred_idx1_str)

            # adjusted scores
            adjusted_scores[selected_idx] = adjusted_scores[selected_idx] & adjusted_scores1
        else:
            pass
            
        if len(group2) > 1:
            selected_idx = (pred_idx == 1)
            pred_idx2_str, adjusted_scores2 = self.multi_pred(group2, 
                scores[selected_idx], topk_neurons, neuron_thres, neuron_polarity)
            pred_idx_str[selected_idx] = np.char.add(pred_idx_str[selected_idx], pred_idx2_str)

            # adjusted scores
            adjusted_scores[selected_idx] = adjusted_scores[selected_idx] & adjusted_scores2
        else:
            pass
            
        return pred_idx_str, adjusted_scores

    def predict_idx(self, 
                    gts: list[str],
                    scores: torch.tensor,
                    topk_neurons: torch.tensor,
                    neuron_thres: torch.tensor,
                    neuron_polarity: torch.tensor,
                    gold_labels: list[str]):
        if len(set(gts)) == 2:
            pred_idx, adjusted_scores = self.binary_pred(scores, topk_neurons, neuron_thres, neuron_polarity)
        else:
            pred_idx_str, adjusted_scores = self.multi_pred(gts, scores, topk_neurons,neuron_thres, neuron_polarity)
            pred_idx = self.convert_pred_to_tensor(pred_idx_str, gold_labels)
        return pred_idx, adjusted_scores