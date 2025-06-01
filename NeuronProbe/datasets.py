import random
from typing import List
import datasets
from NeuronProbe.base_dataset import BaseDataset, MultilingualDataset

class AgnewsDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="agnews", 
            ds_name: str="agnews",
            n_shot: int = -1, 
            prompt_type: str="default",
            balanced: str ='no', 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)
        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Determine the genre of the news article. "
        instruction += "Please choose from the following options: a.World b.Sports c.Business d.science. "
        instruction += "Select the letter corresponding to the most appropriate genre."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]
        
    def make_demo(self, item, demo=True):
        demonstration = f"### Text:{item['text']}\n### Genres:\na.World\nb.Sports\nc.Business\nd.Science\n### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class AmazonReviewsDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="amazon-reviews", 
            ds_name: str="amazon-reviews",
            n_shot: int = -1, 
            prompt_type: str="default",
            balanced: str ='no', 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)
        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Analyze the sentiment of the given Amazon review and assign a score from 1 (very negative) to 5 (very positive) based on the review. Output only the score."
        return instruction

    @property
    def labels(self):
        return ["1", "2", "3", "4", "5"]
        
    def make_demo(self, item, demo=True):
        demonstration = f"### Input Review:{item['text']}\n### Output Score:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        """
        Generate the `_answer` column. 
        Each token in `_answer` column must set as one token for correctly probed.
        """
        if item['label'] == -1:
            return item
        assert item['label'] in [1, 2, 3, 4, 5]
        item['_answer'] = self.labels[int(item['label']-1)]
        return item

class BBQDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str, 
            ds_name: str=None,
            n_shot: int = -1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "heegyu/bbq"

    @property
    def instruction(self):
        instruction ="### Instruction: Given the context, answer the question by selecting the most appropriate option."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c"]

    def select_demos(self, dataset):
        stopwords = ["n't", "unknown", "cannot", "not", "undetermined"]
        def _find_stop_word(item):
            answer = random_example[f"ans{random_example['label']}"].lower()
            for word in stopwords:
                if word in answer:
                    return True
            return False

        sampled_examples = []
        for answer in self.labels:
            filtered_data = dataset.filter(lambda x: x['_answer'] == answer)
            if len(filtered_data) > 0:
                random.seed(42)
                random_example = random.choice(filtered_data)
                i = 0
                if answer != "a":
                    while _find_stop_word(random_example):
                        random.seed(i)
                        random_example = random.choice(filtered_data)
                        i += 1
                elif answer == 'a':
                    while not _find_stop_word(random_example):
                        random.seed(i)
                        random_example = random.choice(filtered_data)
                        i += 1
                else:
                    raise ValueError(f"get {answer}")

                sampled_examples.append(random_example)
        return sampled_examples

    def make_demo(self, item, demo=True):
        demonstration = f"### Context:{item['context']}\n"""
        demonstration += f"### Question:{item['question']}\n"""
        demonstration += f"### Options:\na.{item['ans0']}\nb.{item['ans1']}\nc.{item['ans2']}\n### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_test_set = dataset['test'].train_test_split(test_size=0.1)
        testset = train_test_set["test"]
        trainset = train_test_set["train"]

        valid_size = min(int(len(trainset) * 0.2), 500)
        train_valid_set = trainset.train_test_split(test_size=valid_size)
        validset = train_valid_set["test"]
        trainset = train_valid_set["train"]
        return trainset, validset, testset

class IMDBDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str='imdb',
            ds_name: str='imdb',
            n_shot: int = -1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int = 42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        return "### Instruction: Based the review, is the movie good or bad?"

    @property
    def labels(self):
        return ["bad", "good"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Review:{item['text']}\n### Answer:"
        demonstration = demonstration.replace("\n\n", "\n")
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class MRDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str='mr', 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "mattymchen/mr"

    @property
    def instruction(self):
        return "### Instruction: Based the review, is the movie good or bad?"

    @property
    def labels(self):
        return ["bad", "good"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Review:{item['text']}\n### Answer:"
        demonstration = demonstration.replace("\n\n", "\n")
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_valid_split = dataset['test'].train_test_split(0.2)
        trainset = train_valid_split["train"]
        testset = train_valid_split["test"]
        valid_test_split = trainset.train_test_split(test_size=1000)
        validset = valid_test_split['train']
        testset = valid_test_split['test']
        return trainset, validset, testset
    
class MNLIDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="mnli", 
            ds_name: str="mnli",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        return "### Instruction: Given a premise and a hypothesis, determine the relationship."

    @property
    def labels(self):
        return ["a", "b", "c"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Premise:{item['premise']}\n" + \
                        f"### Hypothesis:{item['hypothesis']}\n" + \
                        f"### Options:\na.Entailment\nb.Neutral\nc.Contradiction\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class QNLIDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="qnli", 
            ds_name: str="qnli",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/glue"

    @property
    def instruction(self):
        return "### Instruction: Whether the context sentence contains the answer to the question."

    @property
    def labels(self):
        return ["a", "b"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Question:{item['question']}\n" + \
                        f"### Context:{item['sentence']}\n" + \
                        f"### Options:\na.Contained\nb.Not contained\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_valid_set = dataset['train'].train_test_split(test_size=1000)
        validset = train_valid_set["test"]
        trainset = train_valid_set["train"]
        return trainset, validset, dataset['validation']

class SNLIDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="snli", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "stanfordnlp/snli"

    @property
    def instruction(self):
        return "### Instruction: Given a premise and a hypothesis, determine the relationship."

    @property
    def labels(self):
        return ['a', 'b', 'c']

    def make_demo(self, item, demo=True):
        demonstration = f"### Premise:{item['premise']}\n" + \
                        f"### Hypothesis:{item['hypothesis']}\n" + \
                        f"### Question: What is the relationship between the two sentences?.\n" + \
                        f"### Options:\na.Entailment\nb.Contradiction\nc.Neutral\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        labels = ["a", "b", "c"]
        try:
            item['_answer'] = labels[item['label']]
        except Exception as e:
            pass
        return item

class WNLIDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="wnli", 
            ds_name: str="wnli",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/glue"

    @property
    def instruction(self):
        return "### Instruction: Given a premise and a hypothesis, determine the relationship."

    @property
    def labels(self):
        return ["a", "b"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Premise:{item['sentence1']}\n" + \
                        f"### Hypothesis:{item['sentence2']}\n" + \
                        f"### Question: What is the relationship between the two sentences?.\n" + \
                        f"### Options:\na.Not entailment\nb.Entailment\n" + \
                        f"### Answer:"
        
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_valid_set = dataset['train'].train_test_split(test_size=100)
        validset = train_valid_set["test"]
        trainset = train_valid_set["train"]
        return trainset, validset, dataset['validation']
    
class FEVERDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="fever",
            ds_name: str="fever",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Given the factual claim, evaluate its validity and respond with either 'True' or 'False' only.\n"
        instruction += "Do not provide any additional information or explanation."
        return instruction

    @property
    def labels(self):
        return ["False", "True"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Claim:{item['claim']}\n### Validity (True or False):"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class COLADataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="cola", 
            ds_name: str="cola",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/glue"

    @property
    def instruction(self):
        instruction = "### Instruction: Is the sentence grammatically acceptable or not?"
        return instruction

    @property
    def labels(self):
        return ["a", "b"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Sentence:{item['sentence']}\n### Options:\na.Not acceptable\nb.Acceptable\n### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_valid_set = dataset['train'].train_test_split(test_size=1000)
        validset = train_valid_set["test"]
        trainset = train_valid_set["train"]
        return trainset, validset, dataset['validation']
    
class MyriadLAMADataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="myriadlama", 
            ds_name: str="myriadlama",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Predict the [MASK] in the sentence from the options. "
        instruction += "Do not provide any additional information or explanation."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        prompt = item['text'].replace('[Y]', '[MASK]')
        options = f"a.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### Question:{prompt}\n" + \
                        f"### Options:\n{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class TempLAMADataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="templama", 
            ds_name: str="templama",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Select the correct year from the provided options that match the temporal fact in the sentence. "
        instruction += "Output the index of the correct year."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"a.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### Question:{item['text']}\n" + \
                        f"### Options:\n{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class HotpotqaDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="hotpotqa", 
            ds_name: str="hotpotqa",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        import datasets
        
        ds = datasets.load_dataset(self.ds_path, ds_name, trust_remote_code=True).shuffle(seed=42)
        self.answers = set([output[0]['answer'] for output in ds['train']['output']])
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)
        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "facebook/kilt_tasks"

    @property
    def instruction(self):
        instruction = "### Instruction: Answer the question from the options.\n"
        instruction += "Do not provide any additional information or explanation."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Question:{item['input']}\n" + \
                        f"### Options:\n{item['options']}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_test_split = dataset['train'].train_test_split(test_size=1000)
        trainset = train_test_split["train"].select(range(10000))
        testset = train_test_split["test"]
        return trainset, dataset['validation'], testset, 

    def create_answers(self, item):
        options = random.sample(self.answers, 4)
        if item['output'][0]['answer'] not in options:
            options[0] = item['output'][0]['answer']
        random.shuffle(options)
        item['options'] = f"a.{options[0]}\nb.{options[1]}\nc.{options[2]}\nd.{options[3]}"
        idx = options.index(item['output'][0]['answer'])
        item['_answer'] = ['a', 'b', 'c', 'd'][idx]
        return item
    
class MMLUDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="mmlu", 
            ds_name: str="all",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)
        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "lighteval/mmlu"

    @property
    def instruction(self):
        instruction = "### Instruction: Answer the question from the options.\n"
        instruction += "Do not provide any additional information or explanation."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Question:{item['question']}\n" + \
                        f"### Options:\na.{item['choices'][0]}\nb.{item['choices'][1]}\nc.{item['choices'][2]}\nd.{item['choices'][3]}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        return dataset['auxiliary_train'], dataset['validation'], dataset['test']

    def create_answers(self, item):
        item['_answer'] = self.labels[item['answer']]
        return item
    
    def select_demos(self, dataset: datasets.DatasetDict):
        assert self.labels is not None, f"`self.labels` need to specified in child class"
        
        if self.n_shot != len(self.labels):
            raise ValueError(f"The default `select_demos` function requires the n-shot to be the same as labels size")
        
        sampled_examples = []
        for answer in self.labels:
            filtered_data = dataset.filter(lambda x: x['_answer'] == answer, load_from_cache_file=False)
            filtered_data = filtered_data.filter(lambda x: len(x['article']) < 200, load_from_cache_file=False)
            if len(filtered_data) > 0:
                random.seed(self.seed)
                random_example = random.choice(filtered_data)
                sampled_examples.append(random_example)
            else:
                raise ValueError(f"The size of {len(filtered_data)} shouldn't be zero, but got {len(filtered_data)}")
        return sampled_examples

class MRPCDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="mrpc", 
            ds_name: str="mrpc",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/glue"

    @property
    def instruction(self):
        instruction = "### Instruction: Is the second sentence a paraphrase of the first? Answer exactly 'yes' or 'no'."
        return instruction

    @property
    def labels(self):
        return ["no", "yes"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                        f"### Sentence 2:{item['sentence2']}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class PAWSDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="paws", 
            ds_name: str="paws",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        if prompt_type.endswith("-shot"):
            n_shot = int(prompt_type.split('-')[0])
        elif prompt_type.startswith("sample"):
            seed = int(prompt_type[6:])
            assert seed != 42
        elif prompt_type.startswith("instruction") or prompt_type == 'default':
            pass
        else:
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        
    @property
    def instruction(self):
        if self.prompt_type in ["default"] or self.prompt_type.startswith("sample") or self.prompt_type.endswith('-shot'):
            instruction = "### Instruction: Is the second sentence a paraphrase of the first? Answer exactly 'yes' or 'no'."
        elif self.prompt_type == "instruction1":
            instruction = "### Instruction: Given two sentences, determine if they are paraphrases of each other."
        elif self.prompt_type == "instruction2":
            instruction = "### Instruction: Review the two given sentences and decide if they express the same idea in different words."
        elif self.prompt_type == "instruction3":
            instruction = "### Instruction: Examine the two sentences provided. Determine if the second sentence is a valid paraphrase of the first sentence."        
        elif self.prompt_type == "instruction4":
            instruction = "### Instruction: You are provided with two sentences. Identify whether they convey identical ideas or differ in meaning."
        else:
            raise NotImplementedError
        
        return instruction

    @property
    def labels(self):
        if self.prompt_type in ["default"] or self.prompt_type.endswith('-shot') or self.prompt_type.startswith('sample'):
            return ["no", "yes"]
        elif self.prompt_type.startswith("instruction"):
            return ["a", "b"]
        else:
            raise NotImplementedError

    def make_demo(self, item, demo=True):
        if self.prompt_type == "default" or self.prompt_type.endswith('-shot') or self.prompt_type.startswith('sample'):
            demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                            f"### Sentence 2:{item['sentence2']}\n" + \
                            f"### Answer:"
        elif self.prompt_type == "instruction1":
             demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                             f"### Sentence 2:{item['sentence2']}\n" + \
                             f"### Options:\na.not paraphrase\nb.paraphrase\n" \
                             f"### Answer:"
        elif self.prompt_type == "instruction2":
             demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                             f"### Sentence 2:{item['sentence2']}\n" + \
                             f"### Options:\na.non-equivalent\nb.equivalent\n" \
                             f"### Answer:"
        elif self.prompt_type == "instruction3":
             demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                             f"### Sentence 2:{item['sentence2']}\n" + \
                             f"### Options:\na.different\nb.similar\n" \
                             f"### Answer:"
        elif self.prompt_type == "instruction4":
             demonstration = f"### Sentence 1:{item['sentence1']}\n" + \
                             f"### Sentence 2:{item['sentence2']}\n" + \
                             f"### Options:\na.The sentences convey different idea.\nb.The sentences convey the same ideas.\n" \
                             f"### Answer:"
        else:
            raise NotImplementedError
        
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def select_demos(self, dataset):
        if self.n_shot == 2:
            return super().select_demos(dataset)
        else:
            examples = dataset.shuffle(seed=42).select(range(self.n_shot))
            return examples

class QQPDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="qqp", 
            ds_name: str="qqp",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/glue"

    @property
    def instruction(self):
        instruction = "### Instruction: Is the second sentence a paraphrase of the first? Answer exactly 'yes' or 'no'."
        return instruction

    @property
    def labels(self):
        return ["no", "yes"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Sentence 1:{item['question1']}\n" + \
                        f"### Sentence 2:{item['question2']}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def split_dataset(self, dataset):
        train_valid_set = dataset['train'].train_test_split(test_size=1000)
        validset = train_valid_set["test"]
        trainset = train_valid_set["train"]
        return trainset, validset, dataset['validation']

class HaluEvalDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="halueval", 
            ds_name: str="halueval",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Given the knowledge context, dialogue histroy and response, determine if any hallucination is present.\n"
        instruction += "Provide a response of either 'yes' or 'no' only."
        return instruction

    @property
    def labels(self):
        return ["no", "yes"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Context:{item['knowledge']}\n" + \
                        f"### Dialogue history:{item['dialogue']}\n" + \
                        f"### Response:{item['response']}\n" + \
                        f"### Hallucination (yes or no):"
        demonstration = demonstration.replace("\n\n", "\n")
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        item['label'] = item['hallucination']
        item['_answer'] = self.labels[item['hallucination']]
        return item
    
class ToxicDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="toxicity", 
            ds_name: str="toxicity",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Determine if the provided text contains toxic content. "
        instruction += "Provide a response of either 'yes' or 'no' only."
        return instruction

    @property
    def labels(self):
        return ["no", "yes"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Text:{item['text']}\n" + \
                        f"### Toxicity (yes or no):"
        demonstration = demonstration.replace("\n\n", "\n")
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        item['label'] = item['toxic']
        item['_answer'] = self.labels[item['toxic']]
        return item

class StereosetDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="stereoset", 
            ds_name: str="stereoset",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Given the context, identify and select the sentence that does not convey the stereotype related to the context."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Context:{item['context']}\n" + \
                        f"### Options:\na.{item['sentences'][0]}\nb.{item['sentences'][1]}\nc.{item['sentences'][2]}\n" + \
                        f"### Answer:"
        demonstration = demonstration.replace("\n\n", "\n")
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        item['label'] = self.labels[item['non-stereotype']]
        item['_answer'] = self.labels[item['non-stereotype']]
        return item

class SWAGDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="swag", 
            ds_name: str="swag",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Given the context, select the most likely completion from the following choices. Please exactly answer the label."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Context:{item['text']}\n" + \
                        f"### Options:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class CommonsenseQADataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="commonsenseqa", 
            ds_name: str="commonsenseqa",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def instruction(self):
        instruction = "### Instruction: Please select the most accurate and relevant answer based on the context."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d", "e"]

    def make_demo(self, item, demo=True):
        choices = item["options"]
        options = f"### Options:\na.{choices[0]}\nb.{choices[1]}\nc.{choices[2]}\nd.{choices[3]}\ne.{choices[4]}"
        demonstration = f'### Question:{item["text"]}\n{options}\n### Answer:'
        
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class RACEDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="race", 
            ds_name: str="all",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "ehovy/race"

    @property
    def instruction(self):
        instruction = "### Instruction: Please select the best answer based on the given passage and question."
        return instruction

    @property
    def labels(self):
        return ['a', 'b', 'c', 'd']

    def make_demo(self, item, demo=True):
        choices = item['options']
        options = "\n".join([f"{chr(97 + i)}. {choice}" for i, choice in enumerate(choices)])
        demonstration = f"### Passage:{item['article']}\n" + \
                        f"### Question:{item['question']}\n" + \
                        f"### Options: \n{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
    def create_answers(self, item):
        keymap = {"A": "a", "B": "b", "C": "c", "D": "d"}
        if item["answer"] != "":
            item['_answer'] = keymap[item["answer"]]
        return item

class TRECDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="trec", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "CogComp/trec"

    @property
    def instruction(self):
        instruction = "### Instruction: You are a experienced classifier. Classify the following question into the appropriate category."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d", "e", "f"]

    def make_demo(self, item, demo=True):
        question = item["text"]
        options = "### Categories:\n1. Abbreviation\n2. Entity\n3. Description\n4. Human\n5. Location\n6. Numeric"
        demonstration = f'### Question: {question}\n{options}\n### Answer:'
        
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        item['_answer'] = self.labels[item["coarse_label"]]
        return item

    def split_dataset(self, dataset):
        test_valid_split = dataset['train'].train_test_split(test_size=0.1, seed=self.seed)
        trainset = test_valid_split['train']
        validset = test_valid_split['test']
        testset = dataset['test']
        return trainset, validset, testset

class HellaSwagDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="hellaswag", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "Rowan/hellaswag"

    @property
    def instruction(self):
        instruction = "### Instruction: Select the most appropriate continuation for the given context."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        context = item["ctx"]
        choices = item["endings"]
        options = f"### Options:\na. {choices[0]}\nb. {choices[1]}\nc. {choices[2]}\nd. {choices[3]}"
        demonstration = f'### Context: {context}\n{options}\n### Answer:'

        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        keymap = {'0': "a", '1': "b", '2': "c", '3': "d"}
        if item["label"] != '':
            item['_answer'] = keymap[item["label"]]
        return item

    def split_dataset(self, dataset):
        test_valid_split = dataset['validation'].train_test_split(test_size=1000)
        testset = test_valid_split['test']
        validset = test_valid_split['train']
        return dataset['train'], validset, testset
    
class SST5Dataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="sst5", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return 'SetFit/sst5'

    @property
    def instruction(self):
        instruction = "### Instruction: Determine the sentiment of the given sentence."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d", "e"]

    def make_demo(self, item, demo=True):
        sentence = item["text"]
        options = f"### Options:\na. very negative\nb. negative\nc. neutral\nd. positive\ne. very positive"
        demonstration = f'### Sentence: {sentence}\n{options}\n### Answer:'

        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        keymap = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        if item["label"] != -1:
            item['_answer'] = keymap[item["label"]]
        return item

    def split_dataset(self, dataset):
        test_valid_split = dataset['validation'].train_test_split(test_size=1000)
        testset = test_valid_split['test']
        validset = test_valid_split['train']
        return dataset['train'], validset, testset

class BoolQDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="boolq", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "google/boolq"

    @property
    def instruction(self):
        instruction = "### Instruction: Based on the given passage, please answer the question with 'yes' or 'no'."
        return instruction

    @property
    def labels(self):
        return ["yes", "no"]

    def make_demo(self, item, demo=True):
        passage = item["passage"]
        question = item["question"]
        options = "### Options:\n yes\n no"
        demonstration = f'### Passage: {passage}\n### Question: {question}?\n{options}\n### Answer:'

        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        keymap = {True: "yes", False: "no"}
        if item["answer"] is not None:
            item['_answer'] = keymap[item["answer"]]
        return item

    def split_dataset(self, dataset):
        test_valid_split = dataset['validation'].train_test_split(test_size=1000)
        testset = test_valid_split['test']
        validset = test_valid_split['train']
        return dataset['train'], validset, testset

class PhraseChunkingDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="chunking", 
            ds_name: str="chunking",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Identify the chunk type for the specified target phrase in the sentence and select the correct label from the provided options. "
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"### Options:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### Input text:{item['text']}\n" + \
                        f"### Target phrase:\'{' '.join(item['target_chunk'])}\'\n" + \
                        f"{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class NERDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="ner", 
            ds_name: str="ner",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Identify the named entity type for the specified target phrase in the give text. "
        instruction += "Choose the correct type from the provided options"
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"### Options:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### Input text:{item['text']}\n" + \
                        f"### Target phrase:\'{' '.join(item['target_chunk'])}\'\n" + \
                        f"{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class POSDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="postag", 
            ds_name: str="postag",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Determine the part-of-speech (POS) tag for the highlighted target word in the given text. "
        instruction += "Choose the correct tag from the provided options."
        return instruction

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"### Options:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### Input text:{item['text']}\n" + \
                        f"### Target word:\'{item['target_token']}\'\n" + \
                        f"{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class GEDDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="clang8", 
            ds_name: str="clang8",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Which of the sentence below is linguistically acceptable?"
        return instruction

    @property
    def labels(self):
        return ["a", "b"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Sentences:\na.{item['sentence1']}\nb.{item['sentence2']}\n### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
    def create_answers(self, item):
        item['label'] = item['acceptable']
        item['_answer'] = self.labels[item['acceptable']]
        return item

class LTIDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="lti", 
            ds_name: str="lti",
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        self.langs = ['en', 'fr', "de", 'zh', 'es']
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    @property
    def instruction(self):
        instruction = "### Instruction: Identify the language of the given sentence."
        return instruction

    @property
    def labels(self):
        # return ["English", "French", "German", "Chinese", "Spanish"]
        return ["a", "b", "c", "d", "e"]

    def make_demo(self, item, demo=True):
        demonstration = f"### Text:{item['text']}\n"
        demonstration += "### Options:\na.English\nb.French\na.German\na.Chinese\na.Spanish\n"
        demonstration += "### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
    def create_answers(self, item):
        item['label'] = self.langs.index(item['lang'])
        item['_answer'] = self.labels[item['label']]
        return item
    
class MultilingualPOSDataset(MultilingualDataset):
    def __init__(
            self, 
            dataset_name: str="mpostag", 
            ds_name: str="mpostag",
            langs: List[str]=["en", "fr", "de", "zh", "es"],
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, langs, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    
    @property
    def lang2instructions(self):
        lang2instructions = {
            'en': "### Instruction: Determine the part-of-speech tag for the highlighted target word in the given text. Choose the correct tag from the provided options.",
            'fr': "### Instruction: Déterminez la catégorie grammaticale (POS) du mot cible mis en évidence dans le texte donné. Choisissez la catégorie correcte parmi les options fournies.",
            'de': "### Anweisung: Bestimmen Sie die Wortart des hervorgehobenen Zielworts im gegebenen Text. Wählen Sie die richtige Wortart aus den bereitgestellten Optionen aus.",
            'zh': "### 指令：确定给定文本中高亮目标词的词性。从提供的选项中选择正确的词性标签。",
            'es': "### Instrucción: Determine la categoría gramatical de la palabra objetivo resaltada en el texto dado. Elija la etiqueta correcta de las opciones proporcionadas."
        }
        return lang2instructions

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"### {self.lang2options[item['lang']]}:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### {self.lang2texts[item['lang']]}:{item['text']}\n" + \
                        f"### {self.lang2targets[item['lang']]}:\'{item['target_token']}\'\n" + \
                        f"{options}\n" + \
                        f"### {self.lang2answers[item['lang']]}:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class MultilingualPOSDataset(MultilingualDataset):
    def __init__(
            self, 
            dataset_name: str="mpostag", 
            ds_name: str="mpostag",
            langs: List[str]=["en", "fr", "de", "zh", "es"],
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, langs, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    
    @property
    def lang2instructions(self):
        lang2instructions = {
            'en': "### Instruction: Determine the part-of-speech tag for the highlighted target word in the given text. Choose the correct tag from the provided options.",
            'fr': "### Instruction: Déterminez la catégorie grammaticale (POS) du mot cible mis en évidence dans le texte donné. Choisissez la catégorie correcte parmi les options fournies.",
            'de': "### Anweisung: Bestimmen Sie die Wortart des hervorgehobenen Zielworts im gegebenen Text. Wählen Sie die richtige Wortart aus den bereitgestellten Optionen aus.",
            'zh': "### 指令：确定给定文本中高亮目标词的词性。从提供的选项中选择正确的词性标签。",
            'es': "### Instrucción: Determine la categoría gramatical de la palabra objetivo resaltada en el texto dado. Elija la etiqueta correcta de las opciones proporcionadas."
        }
        return lang2instructions

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        options = f"### {self.lang2options[item['lang']]}:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### {self.lang2texts[item['lang']]}:{item['text']}\n" + \
                        f"### {self.lang2targets[item['lang']]}:\'{item['target_token']}\'\n" + \
                        f"{options}\n" + \
                        f"### {self.lang2answers[item['lang']]}:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class MLAMADataset(MultilingualDataset):
    def __init__(
            self, 
            dataset_name: str="mlama", 
            ds_name: str="mlama",
            langs: List[str]=["en", "fr", "de", "zh", "es"],
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, langs, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    
    @property
    def lang2instructions(self):
        lang2instructions = {
            'en': "### Instruction: Predict the [MASK] in the sentence from the options. Do not provide any additional information or explanation.",
            'fr': "### Instruction: Prédisez le [MASK] dans la phrase parmi les options. Ne fournissez aucune information ou explication supplémentaire.",
            'de': "### Anweisung: Sagen Sie das [MASK] im Satz anhand der Optionen voraus. Geben Sie keine zusätzlichen Informationen oder Erklärungen an.",
            'zh': "### 指令：根据选项预测句子中的 [MASK]。不要提供任何额外的信息或解释。",
            'es': "### Instrucción: Prediga el [MASK] en la oración a partir de las opciones. No proporcione información ni explicaciones adicionales."
        }
        return lang2instructions

    @property
    def labels(self):
        return ["a", "b", "c", "d"]

    def make_demo(self, item, demo=True):
        prompt = item['text'].replace('[Y]', '[MASK]')
        options = f"### {self.lang2options[item['lang']]}:\na.{item['options'][0]}\nb.{item['options'][1]}\nc.{item['options'][2]}\nd.{item['options'][3]}"
        demonstration = f"### {self.lang2questions[item['lang']]}:{prompt}\n" + \
                        f"{options}\n" + \
                        f"### {self.lang2answers[item['lang']]}:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
class XNLIDataset(MultilingualDataset):
    def __init__(
            self, 
            dataset_name: str="xnli", 
            ds_name: str="xnli",
            langs: List[str]=["en", "fr", "de", "zh", "es"],
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        self.lang2premise = {'en': "Premise", 'fr': "Prémisse", 'de': "Prämisse", 'zh': "前提", 'es': "Premisa"}
        self.lang2hypothesis = {'en': "Hypothesis", 'fr': "Hypothèse", 'de': "Hypothese", 'zh': "假设", 'es': "Hipótesis"}
        self.lang2entailment = {'en': "Entailment", 'fr': "Implication", 'de': "Implikation", 'zh': "蕴涵", 'es': "Implicación"}
        self.lang2neutral = {'en': "Neutral", 'fr': "Neutre", 'de': "Neutral", 'zh': "中立", 'es': "Neutral"}
        self.lang2contradiction = {'en': "Contradiction", 'fr': "Contradiction", 'de': "Widerspruch", 'zh': "矛盾", 'es': "Contradicción"}

        super().__init__(dataset_name, ds_name, n_shot, prompt_type, langs, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    
    @property
    def lang2instructions(self):
        lang2instructions = {
            'en': "### Instruction: Given a premise and a hypothesis, determine the relationship.",
            'fr': "### Instruction: Étant donné une prémisse et une hypothèse, déterminez la relation.",
            'de': "### Anweisung: Gegeben eine Prämisse und eine Hypothese, bestimmen Sie die Beziehung.",
            'zh': "### 指令：给定一个前提和一个假设，确定它们之间的关系。",
            'es': "### Instrucción: Dada una premisa y una hipótesis, determine la relación."
        }
        return lang2instructions

    @property
    def labels(self):
        return ["a", "b", "c"]

    def make_demo(self, item, demo=True):
        if item['lang'] == 'zh':
            item['premise'] = ''.join(item['premise'].split(' '))
            item['hypothesis'] = ''.join(item['hypothesis'].split(' '))
        
        demonstration = f"### {self.lang2premise[item['lang']]}:{item['premise']}\n" + \
                        f"### {self.lang2hypothesis[item['lang']]}:{item['hypothesis']}\n" + \
                        f"### {self.lang2options[item['lang']]}:\na.{self.lang2entailment[item['lang']]}\nb.{self.lang2neutral[item['lang']]}\nc.{self.lang2contradiction[item['lang']]}\n" + \
                        f"### {self.lang2answers[item['lang']]}:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

class MultilingualAmazonReviewsDataset(MultilingualDataset):
    def __init__(
            self, 
            dataset_name: str="amazon-review-multi", 
            ds_name: str="amazon-review-multi",
            langs: List[str]=["en", "fr", "de", "zh", "es"],
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=100,
            log_to_file: bool = True,
            recache: bool=False, 
            verbose: bool=True) -> None:
        
        self.lang2inputkeys = {'en': "Input Review", 'fr': "Avis d'entrée", 'de': "Eingabebewertung", 'zh': "输入评论", 'es': "Reseña de entrada"}
        self.lang2outputkeys = {'en': "Output Score", 'fr': "Note de sortie", 'de': "Ausgabewertung", 'zh': "输出评分", 'es': "Puntuación de salida"}
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, langs, balanced, seed, log_to_file, recache, verbose)        
        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")
    
    @property
    def lang2instructions(self):
        lang2instructions = {
            'en': "### Instruction: Analyze the sentiment of the given Amazon review and assign a score from 1 (very negative) to 5 (very positive) based on the review. Output only the score.",
            'fr': "### Instruction: Analysez le sentiment de l'avis Amazon donné et attribuez une note de 1 (très négatif) à 5 (très positif) en fonction de l'avis. Indiquez uniquement la note.",
            'de': "### Anweisung: Analysieren Sie die Stimmung der gegebenen Amazon-Bewertung und vergeben Sie eine Punktzahl von 1 (sehr negativ) bis.",
            'zh': "### 指令：分析给定的亚马逊评论的情感，并根据评论分配一个从1（非常负面）到5（非常正面）的评分。只输出评分。",
            'es': "### Instrucción: Analiza el sentimiento de la reseña de Amazon proporcionada y asigna una puntuación del 1 (muy negativo) al 5 (muy positivo) según la reseña. Solo devuelve la puntuación."
        }
        return lang2instructions
    
    @property
    def labels(self):
        return ["1", "2", "3", "4", "5"]
        
    def make_demo(self, item, demo=True):
        demonstration = f"### {self.lang2inputkeys[item['lang']]}:{item['review']}\n### {self.lang2outputkeys[item['lang']]}:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration

    def create_answers(self, item):
        assert item['label'] in [1, 2, 3, 4, 5]
        item['_answer'] = self.labels[int(item['label']-1)]
        return item

class BLiMPDataset(BaseDataset):
    def __init__(
            self, 
            dataset_name: str="blimp", 
            ds_name: str=None,
            n_shot: int=-1, 
            prompt_type: str="default",
            balanced: str = "no", 
            seed: int=42,
            log_to_file: bool = True,
            recache: bool=False) -> None:
        
        self.term2options = {
            'island_effects': 'Invalid syntactic structure blocks connections between sentence parts.',
            'anaphor_agreement': 'Incorrect pronouns in gender or number',
            'argument_structure': 'Inappropriate pairing of verbs with their arguments.',
            'determiner_noun_agreement': 'Mismatch between demonstrative and noun number.',
            'subject_verb_agreement': 'Verb doesn’t match the subject’s number/person.',
            'ellipsis': 'Missing words create confusion.',
            'control_raising': 'Incorrect subject for predicates.',
            'quantifiers': 'Incorrect quantity words.',
            'irregular_forms': 'Incorrect verb/noun forms used.',
            'npi_licensing': 'Incorrect negative words used.',
            'binding': 'Pronouns or reflexives don’t link to the correct noun.',
            'filler_gap_dependency': 'Incorrect wh-question structure causes confusion.',
        }
        self.correct_message = "No error in the sentence."
        super().__init__(dataset_name, ds_name, n_shot, prompt_type, balanced, seed, log_to_file, recache)

        if prompt_type != "default" and not prompt_type.endswith('-shot'):
            raise NotImplementedError(f"Prompt type: {self.prompt_type} is not supported yet")

    @property
    def ds_path(self):
        return "nyu-mll/blimp"

    @property
    def instruction(self):
        instruction = "### Instruction: For each sentence, carefully analyze its grammaticality and select the correct option that describes it."
        return instruction

    @property
    def labels(self):
        return ['a', 'b', 'c', 'd']

    def download_dataset(self):
        from tqdm import tqdm
        from datasets import get_dataset_config_names, concatenate_datasets

        all_datasets = []
        linguistic_terms2config = {}
        config_names = get_dataset_config_names("nyu-mll/blimp")
        random.shuffle(config_names)
        for config in tqdm(config_names[:6]):
            dataset = datasets.load_dataset("nyu-mll/blimp", name=config)
            linguistic_terms2config.setdefault(dataset['train']['linguistics_term'][0], []).append(config)
            if config == 's-selection':
                dataset['train'] = dataset['train'].map(lambda x: {'linguistics_term': 'argument_structure'})
            all_datasets.append(dataset)
        # linguistic_terms2config['argument_structure'].extend(linguistic_terms2config['s-selection'])
        all_dataset = concatenate_datasets([dataset['train'] for dataset in all_datasets])
        dataset = datasets.DatasetDict({'train': all_dataset})
        return dataset
    
    def make_demo(self, item, demo=True):
        options = []
        for i, term in enumerate(item['terms']):
            if term == 'correct':
                options.append(f"{chr(97 + i)}. {self.correct_message}")
            else:
                options.append(f"{chr(97 + i)}. {self.term2options[term]}")
        options = "\n".join(options)
        demonstration = f"### Sentence:{item['_prompt']}\n" + \
                        f"### Options: \n{options}\n" + \
                        f"### Answer:"
        if demo:
            return f'{demonstration}{item["_answer"]}'
        return demonstration
    
    def create_answers(self, item):
        all_terms = list(self.term2options.keys())
        if item['linguistics_term'] not in all_terms:
            print(all_terms)
            print(item['linguistics_term'])
        all_terms.remove(item['linguistics_term'])
        random.shuffle(all_terms)
        terms = []
        terms.extend([item['linguistics_term'], 'correct'])
        terms.extend(all_terms[:2])
        random.shuffle(terms)
        item['terms'] = terms

        if random.random() <= 0.25:
            item['_prompt'] = item['sentence_good']
            term = 'correct'
        else:
            item['_prompt'] = item['sentence_bad']
            term = item['linguistics_term']

        item['_answer'] = ['a', 'b', 'c', 'd'][terms.index(term)]
        return item
    
    def split_dataset(self, dataset):
        train_valid_split = dataset['train'].train_test_split(test_size=1000)
        validset = train_valid_split["test"]
        train_test_split = train_valid_split['train'].train_test_split(test_size=1000)
        testset = train_test_split["test"]
        trainset = train_test_split["train"]
        return trainset, validset, testset