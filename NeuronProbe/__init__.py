import os
import pathlib

DATA_CACHE_ROOT = "<path/to/your/data/cache/root>"  # Change this to your actual data cache root path
GRANEUR_LOG_ROOT = pathlib.Path(__file__).parent.absolute() / "../logs"
GRANEUR_DATASET_ROOT = os.path.join(DATA_CACHE_ROOT, "datasets")
GRANEUR_NEURONS_CACHE_ROOT = os.path.join(DATA_CACHE_ROOT, "grad_neurons")
LIGHT_NEURONS_CACHE_ROOT = os.path.join(DATA_CACHE_ROOT, "light_neurons")
GRANEUR_MODELFT_CACHE_ROOT = os.path.join(DATA_CACHE_ROOT, "grad_ft")

os.makedirs(DATA_CACHE_ROOT, exist_ok=True)
os.makedirs(GRANEUR_LOG_ROOT, exist_ok=True)
os.makedirs(GRANEUR_DATASET_ROOT, exist_ok=True)
os.makedirs(GRANEUR_NEURONS_CACHE_ROOT, exist_ok=True)

ALL_TASKS = [
    "clang8",
    "postag",
    "chunking",
    "ner",
    "agnews",
    "amazon-reviews",
    "imdb",
    "myriadlama",
    "fever",
    "commonsenseqa",
    "templama",
    "paws",
    "mnli",
    "swag",
    "halueval",
    "toxicity",
    "stereoset",
    "amazon-review-multi",
    "lti",
    "mlama",
    'xnli',
    'mpostag'
]

TASKS_OPTIONS = {
    "clang8": 2, "postag": 4, "chunking": 4, "ner": 4,
    "agnews": 4, "amazon-reviews": 5, "imdb": 2,
    "myriadlama": 4, "fever": 2, "commonsenseqa": 4, "templama": 4,
    "paws": 2, "mnli": 3, "swag": 4,
    "halueval": 2, "toxicity": 2, "stereoset": 3,
    "amazon-review-multi": 5, "lti": 5, "mlama": 4, 'xnli': 3, 'mpostag': 4
}

TASK2NAMES = {
    "clang8": "GED", "postag": "POS", "chunking": "CHUNK", "ner": "NER",
    "agnews": "Agnews", "amazon-reviews": "Amazon", "imdb": "IMDB", 
    "myriadlama": "MyriadLAMA", "fever": "FEVER", "commonsenseqa": "CSQA", "templama": "TempLAMA",
    "paws": "PAWS", "mnli": "MNLI", "swag": "SWAG", 
    "halueval": "HaluEval", "toxicity": "Toxic", "stereoset": "Stereoset",
    "amazon-review-multi": "M-Amazon", "lti": "LTI", "mlama": "mLAMA", 'xnli': "XNLI", 'mpostag': "M-POS"
}

TASK_PER_GENRE = {
    'linguistic': ["clang8", "postag", "chunking", "ner"],
    'classification': ["agnews", "amazon-reviews", "imdb"],
    'factuality': ["myriadlama", "fever", "commonsenseqa", "templama"],
    'nli': ["paws", "mnli", "swag"],
    'self-reflection': ["halueval", "toxicity", "stereoset"],
    'multilinguality': ["lti", "amazon-review-multi", "mlama", 'xnli', 'mpostag'],
}

SIX_TASKS = ["ner", "agnews", "paws", "commonsenseqa", "halueval", "mlama"]

def load_dataset(
        name :str, 
        prompt_type='default', 
        balanced='no', 
        recache: bool=False, 
        verbose:bool=True):
    from NeuronProbe.datasets import (
        AgnewsDataset, 
        CommonsenseQADataset, 
        FEVERDataset, 
        MyriadLAMADataset,
        HaluEvalDataset, 
        IMDBDataset, 
        MNLIDataset, 
        PAWSDataset, 
        SWAGDataset, 
        StereosetDataset,
        ToxicDataset,
        GEDDataset,
        AmazonReviewsDataset, 
        MLAMADataset, 
        MultilingualAmazonReviewsDataset, 
        MultilingualPOSDataset, 
        NERDataset, 
        POSDataset, 
        PhraseChunkingDataset, 
        TempLAMADataset, 
        XNLIDataset,
        LTIDataset
    )
    if name == "clang8":
        return GEDDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "postag":
        return POSDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "chunking":
        return PhraseChunkingDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "ner":
        return NERDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    
    elif name == "agnews":
        return AgnewsDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "imdb":
        return IMDBDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "amazon-reviews":
        return AmazonReviewsDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    
    elif name == "mnli":
        return MNLIDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "paws":
        return PAWSDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "swag":
        return SWAGDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    
    elif name == "fever":
        return FEVERDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "myriadlama":
        return MyriadLAMADataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "commonsenseqa":
        return CommonsenseQADataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "templama":
        return TempLAMADataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    
    elif name == "halueval":
        return HaluEvalDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "toxicity":
        return ToxicDataset(prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "stereoset":
        return StereosetDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    
    elif name == "amazon-review-multi":
        return MultilingualAmazonReviewsDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "lti":
        return LTIDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "mlama":
        return MLAMADataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "xnli":
        return XNLIDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    elif name == "mpostag":
        return MultilingualPOSDataset(name, prompt_type=prompt_type, balanced=balanced, recache=recache, verbose=verbose)
    else:
        raise NotImplementedError(f"Task {name} is not supported yet.")
    