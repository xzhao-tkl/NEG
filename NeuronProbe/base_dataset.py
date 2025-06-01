import os
import random
import logging
import datetime
from typing import List
import datasets
from collections import Counter
from abc import ABC, abstractmethod

from NeuronProbe import GRANEUR_DATASET_ROOT, GRANEUR_LOG_ROOT

def down_sampling(dataset):
    label_counts = dataset['_answer']
    unique_labels = set(label_counts)
    label_count_dict = {label: label_counts.count(label) for label in unique_labels}

    min_count = min(label_count_dict.values())
    downsampled_data = []
    for label in unique_labels:
        label_examples = [example for example in dataset if example['_answer'] == label]
        random.seed(42)
        downsampled_examples = random.sample(label_examples, min_count)
        downsampled_data.extend(downsampled_examples)
    downsampled_dataset = dataset.from_list(downsampled_data)
    downsampled_dataset_dict = datasets.DatasetDict({'sampling': downsampled_dataset})
    return downsampled_dataset_dict["sampling"]

def up_sampling(dataset):
    label_counts = dataset['_answer']
    unique_labels = set(label_counts)
    label_count_dict = {label: label_counts.count(label) for label in unique_labels}

    max_count = max(label_count_dict.values())
    upsampled_data = []
    for label in unique_labels:
        label_examples = [example for example in dataset if example['_answer'] == label]
        num_to_add = max_count - len(label_examples)
        
        if num_to_add > 0:
            random.seed(42)
            upsampled_examples = random.choices(label_examples, k=num_to_add)
            label_examples.extend(upsampled_examples)
        upsampled_data.extend(label_examples)
    upsampled_dataset = dataset.from_list(upsampled_data)
    upsampled_dataset_dict = datasets.DatasetDict({'sampling': upsampled_dataset})
    return upsampled_dataset_dict["sampling"]

class BaseDataset(ABC):
    def __init__(
            self, 
            dataset_name: str, 
            ds_name: str,
            n_shot: int,
            prompt_type: str,
            balanced: str='no',
            seed: int=42,
            log_to_file: bool=True, 
            recache: bool=False, 
            verbose: bool=False) -> None:
        """
        `dataset_name`: str
            The dataset name associating with the prompt type. 
            One huggingface dataset could have multiple GraNeur dataset instances due to different prompt types.
            The design of prompt should follows the one-token generation style, because currently only the one-token neuron probing is supported.
        `n_shot`: str
            The number of demonstrations in few-shot learning. 
            The default `-1` means using the same number of demonstrations as the number of labels. 
        """
        self.seed = seed
        self.ds_name = ds_name
        self.prompt_type = prompt_type
        self.dataset_name = dataset_name
        
        n_shot = n_shot if n_shot != -1 else len(self.labels)
        if prompt_type.endswith("-shot"):
            _n_shot = int(prompt_type.split('-')[0])
            # assert _n_shot == n_shot, f'Inconsistent n-shot number between params prompt_type: {prompt_type} and n_shot: {n_shot}'
            n_shot = _n_shot
        self.n_shot = n_shot if n_shot != -1 else len(self.labels)
        assert balanced in ["no", "up", "down"], f'`balanced` has to be one from ["no", "up", "down"] but got {balanced}' 
        self.balanced = balanced

        self.cache_root = os.path.join(GRANEUR_DATASET_ROOT, dataset_name, prompt_type, f"{self.n_shot}-shot_{balanced}_sampling")
        self.logger = self._setup_logger(log_to_file=log_to_file)
        self.ds = self.load_dataset(recache=recache, verbose=verbose)

    @property 
    def ds_path(self):
        """
        `ds_path`: str
        """
        return "iszhaoxin/MCEval8K"

    @property 
    @abstractmethod
    def instruction(self):
        """
        `instruction`: str
            The instruction for zero/few-shot learning. 
        """
        raise NotImplementedError("Subclass must implement `instruction`")

    @property 
    @abstractmethod
    def labels(self):
        """
        `labels`: List[str]
            An array of labels including all possible answers for the prompt-style. 
            All labels must be a one token word. 
        """
        raise NotImplementedError("Subclass must implement `labels`")

    @abstractmethod
    def make_demo(item, demo=True):
        """
        Generate a prompt given a data item. 
        This function can be used to either generate demonstartion (demo=True) and question prompt (demo=False)
        """
        pass
    
    def make_prompt(self, item, demos=""):
        """
        Define the rules for constructing the final prompt. 
        The demo is a MUST argument and will be automatically generated if n_shot > 0. 
        The generated prompt need to be set in `_prompt` column in `DatasetDict`. 
        """
        question = self.make_demo(item, demo=False)
        if demos == "":
            item['_prompt'] = f"{self.instruction.strip()}\n\n{question.strip()}"
        else:
            item['_prompt'] = f"{self.instruction.strip()}\n\n{demos.strip()}\n\n{question.strip()}"
        return item

    def split_dataset(self, dataset):
        """
        Split the original dataset to contain three subset `train`, `valid` and `test`. 
        """
        return dataset['train'], dataset['validation'], dataset['test']

    def select_demos(self, dataset: datasets.DatasetDict):
        assert self.labels is not None, f"`self.labels` need to specified in child class"
        
        if self.n_shot != len(self.labels):
            raise ValueError(f"The default `select_demos` function requires the n-shot to be the same as labels size")
        
        sampled_examples = []
        for answer in self.labels:
            filtered_data = dataset.filter(lambda x: x['_answer'] == answer, load_from_cache_file=False)
            if len(filtered_data) > 0:
                random.seed(self.seed)
                random_example = random.choice(filtered_data)
                sampled_examples.append(random_example)
            else:
                raise ValueError(f"The size of filtered_data shouldn't be zero, but got {len(filtered_data)} for answer: {answer}")
        return sampled_examples
    
    def create_answers(self, item):
        """
        Generate the `_answer` column. 
        Each token in `_answer` column must set as one token for correctly probed.
        """
        if item['label'] == -1:
            return item
        item['_answer'] = self.labels[item['label']]
        return item

    def generate_demos(self, dataset: datasets.DatasetDict):
        demos = ""
        if self.n_shot > 0:
            examples = self.select_demos(dataset['train'])
            assert len(examples) == self.n_shot, f"The number of demos has to be same as n_shot {self.n_shot}, but got {len(examples)}"
            demos = "\n\n".join([self.make_demo(item) for item in examples])    
        return demos
    
    def download_dataset(self):
        if self.ds_name is not None:
            dataset = datasets.load_dataset(self.ds_path, self.ds_name, trust_remote_code=True).shuffle(seed=self.seed)
        else:
            dataset = datasets.load_dataset(self.ds_path, trust_remote_code=True).shuffle(seed=self.seed)
        return dataset
        
    def load_dataset(self, recache: bool, verbose: bool):
        if self.ds_path == None:
            raise ValueError("`ds_path` must be provided in the child class")
        
        if self.n_shot == -1:
            self.n_shot = len(self.labels)

        if os.path.exists(self.cache_root) and not recache:
            dataset = datasets.DatasetDict.load_from_disk(self.cache_root)
            self.logger.info(f"Load dataset from {self.cache_root}")
        else:
            dataset = self.download_dataset()
            trainset, validset, testset = self.split_dataset(dataset)
            dataset = datasets.DatasetDict({
                'train': trainset,
                'validation': validset,
                'test': testset,
            })
            
            dataset = dataset.map(self.create_answers, load_from_cache_file=False)
            params = {}
            if self.n_shot > 0:
                params.update({'demos': self.generate_demos(dataset)})
            dataset = dataset.map(self.make_prompt, fn_kwargs=params, load_from_cache_file=False)
            dataset.save_to_disk(self.cache_root)
            
        if self.balanced == "down":
            dataset['train'] = down_sampling(dataset['train']).shuffle(self.seed)
            dataset['validation'] = down_sampling(dataset['validation']).shuffle(self.seed)
        elif self.balanced == "up":
            dataset['train'] = up_sampling(dataset['train']).shuffle(self.seed)
            dataset['validation'] = up_sampling(dataset['validation']).shuffle(self.seed)
        dataset['test'] = dataset['test'].shuffle(self.seed)
        
        if verbose:
            self.logger.info(f"--------------------Dataset information-----------------------")
            self.logger.info(f"Train dataset: {Counter(dataset['train']['_answer'])}")
            self.logger.info(f"Valid dataset: {Counter(dataset['validation']['_answer'])}")
            self.logger.info(f"Test dataset: {Counter(dataset['test']['_answer'])}")
            self.logger.info(f"Example of the processed prompts:\n{dataset['train']['_prompt'][0]}\n")
        return dataset
    
    def _setup_logger(self, log_filename: str=None, log_to_file: bool=True):
        if log_filename is None:
            log_dir = os.path.join(GRANEUR_LOG_ROOT, self.dataset_name, self.prompt_type)
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(log_dir, f"{self.n_shot}-shot_{self.balanced}_sampling_{timestamp}.log")
        
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(console_handler)

        if log_to_file:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

class MultilingualDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name, 
            ds_name, 
            n_shot, 
            prompt_type, 
            langs: List[str],
            balanced = 'no', 
            seed = 42, 
            log_to_file = True, 
            recache = False, 
            verbose = False):
        self.langs = langs
        self.lang2texts = {'en': 'Text', 'fr': 'Texte', 'de': 'Text', 'zh': '文本', 'es': 'Texto',}
        self.lang2options = {'en': 'Options', 'fr': 'Options', 'de': 'Optionen', 'zh': '选项', 'es': 'Opciones',}
        self.lang2questions = {'en': 'Question', 'fr': 'Question', 'de': 'Frage', 'zh': '答案', 'es': 'Respuesta',}
        self.lang2answers = {'en': 'Answers', 'fr': 'Réponse', 'de': 'Antwort', 'zh': '问题', 'es': 'Pregunta',}
        self.lang2targets = {'en': 'Target word', 'fr': 'Mot cible', 'de': 'Zielwort', 'zh': '目标词', 'es': 'Palabra objetivo',}
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

    def load_dataset(self, recache: bool, verbose: bool):
        if self.ds_path == None:
            raise ValueError("`ds_path` must be provided in the child class")
        
        if self.n_shot == -1:
            self.n_shot = len(self.labels)

        if os.path.exists(self.cache_root) and not recache:
            dataset = datasets.DatasetDict.load_from_disk(self.cache_root)
            self.logger.info(f"Load dataset from {self.cache_root}")
        else:
            dataset = self.download_dataset()
            trainset, validset, testset = self.split_dataset(dataset)
            dataset = datasets.DatasetDict({
                'train': trainset,
                'validation': validset,
                'test': testset,
            })
            
            dataset = dataset.map(self.create_answers, load_from_cache_file=False)
            params = {}
            if self.n_shot > 0:
                params.update({'lang2demos': self.generate_demos(dataset)})
            dataset = dataset.map(self.make_prompt, fn_kwargs=params, load_from_cache_file=False)
            dataset.save_to_disk(self.cache_root)
            
        if self.balanced == "down":
            dataset['train'] = down_sampling(dataset['train']).shuffle(self.seed)
            dataset['validation'] = down_sampling(dataset['validation']).shuffle(self.seed)
        elif self.balanced == "up":
            dataset['train'] = up_sampling(dataset['train']).shuffle(self.seed)
            dataset['validation'] = up_sampling(dataset['validation']).shuffle(self.seed)
        dataset['test'] = dataset['test'].shuffle(self.seed)
        
        if verbose:
            self.logger.info(f"--------------------Dataset information-----------------------")
            self.logger.info(f"Train dataset: {Counter(dataset['train']['_answer'])}")
            self.logger.info(f"Valid dataset: {Counter(dataset['validation']['_answer'])}")
            self.logger.info(f"Test dataset: {Counter(dataset['test']['_answer'])}")
            for lang in self.langs:
                ds_lang = dataset['train'].filter(lambda x: x['lang'] == lang)
                self.logger.info(f"Example of the processed prompts in language: {lang}: \n{ds_lang['_prompt'][0]}\n")
        return dataset
    
    @property
    def instruction(self):
        raise ValueError("Variable self.instruction shouldn't be referred in MultilingualDataset.")
    
    @property
    def lang2instructions(self):
        raise NotImplementedError("Subclass must implement `lang2instructions`")

    def make_prompt(self, item, lang2demos=None):
        instruction = self.lang2instructions[item['lang']]
        if lang2demos is None:
            question = self.make_demo(item, demo=False)
            item['_prompt'] = f"{instruction.strip()}\n\n{question.strip()}"
        else:
            assert item['lang'] in lang2demos
            demos = lang2demos[item['lang']]
            question = self.make_demo(item, demo=False)
            item['_prompt'] = f"{instruction.strip()}\n\n{demos.strip()}\n\n{question.strip()}"
        return item

    def generate_demos(self, dataset: datasets.DatasetDict):
        demos = ""
        if self.n_shot > 0:
            lang2demos = {}
            for lang in self.langs:
                ds_lang = dataset['train'].filter(lambda x: x['lang']==lang)
                examples = self.select_demos(ds_lang)
                assert len(examples) == self.n_shot, f"The number of demos has to be same as n_shot {self.n_shot}, but got {len(examples)}"
                demos = "\n\n".join([self.make_demo(item) for item in examples])
                lang2demos[lang] = demos
        return lang2demos
    