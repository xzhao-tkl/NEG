import os
import torch
from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer
)

HF_MODEL_HUB_ROOT = "/net/tokyo100-10g/data/str01_01/xzhao/huggingface/hub"

BERT_MODELS = {
    "bert-base-uncased": "bert-base-uncased", 
    "bert-large-uncased": "bert-large-uncased", 
    "bert-large-uncased-whole-word-masking": "bert-large-uncased-whole-word-masking", 
    "bert-base-multilingual-uncased": "bert-base-multilingual-uncased",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}

LLAMA_MODELS = [
    "llama2-7b", 
    "llama2-13b", 
    "llama2-70b",
    "llama2-7b-it", 
    "llama2-13b-it", 
    "llama2-70b-it",
    
    "llama3-8b", 
    "llama3-70b",
    "llama3.1-8b", 
    "llama3.1-70b",
    "llama3.2-1b", 
    "llama3.2-3b", 
    
    "llama3-8b-it", 
    "llama3-70b-it",
    "llama3.1-8b-it", 
    "llama3.1-70b-it",
    "llama3.2-1b-it", 
    "llama3.2-3b-it", 
]

ALL_MODELS = list(BERT_MODELS.keys()) + LLAMA_MODELS

MODEL_LAYERS = {
    "bert-base-uncased": 12,
    "bert-large-uncased": 24,
    "bert-large-uncased-whole-word-masking": 24,
    
    "modernbert-base": 22,
    "modernbert-large": 28,
    
    "llama2-7b": 32, 
    "llama2-7b-it": 32, 
    "llama2-13b": 40, 
    "llama2-13b-it": 40, 
    "llama2-70b": 80, 
    "llama2-70b-it": 80, 
    
    "llama3-8b": 32, 
    "llama3-8b-it": 32, 
    "llama3-70b": 80, 
    "llama3-70b-it": 80, 
    
    "llama3.1-8b": 32, 
    "llama3.1-8b-it": 32, 
    "llama3.1-70b": 80, 
    "llama3.1-70b-it": 80, 
    
    "llama3.2-1b": 16, 
    "llama3.2-1b-it": 16, 
    "llama3.2-3b": 28, 
    "llama3.2-3b-it": 28, 

    "qwen2.5-7b-it": 28 
}

INTERMIDIATE_SIZE = {
    "bert-base-uncased": 3072,
    "bert-large-uncased": 4096,
    "bert-large-uncased-whole-word-masking": 4096,
    
    "llama2-7b": 11008, 
    "llama2-7b-it": 11008, 
    "llama2-13b": 13824, 
    "llama2-13b-it": 13824, 
    "llama2-70b": 28672, 
    "llama2-70b-it": 28672, 
    
    "llama3-8b": 14336, 
    "llama3-8b-it": 14336, 
    "llama3-70b": 11008, 
    "llama3-70b-it": 11008, 

    "llama3.1-8b": 14336, 
    "llama3.1-8b-it": 14336, 
    "llama3.1-70b": 11008, 
    "llama3.1-70b-it": 11008, 

    "llama3.2-1b": 8192, 
    "llama3.2-1b-it": 8192, 
    "llama3.2-3b": 11008, 
    "llama3.2-3b-it": 11008,

    "qwen2.5-7b-it": 18944 
}

MODEL_PATH_ROOT = "/net/tokyo100-10g/data/str01_01/xzhao/models/llama_hf/"
LLAMA_MODEL_PATHS = {
    "llama2-7b": os.path.join(MODEL_PATH_ROOT, "llama2_7b"),
    "llama2-7b-it": os.path.join(MODEL_PATH_ROOT, "llama2_7b_it"),
    "llama2-13b": os.path.join(MODEL_PATH_ROOT, "llama2_13b"),
    "llama2-13b-it": os.path.join(MODEL_PATH_ROOT, "llama2_13b_it"),
    "llama2-70b": os.path.join(MODEL_PATH_ROOT, "llama2_70b"),
    "llama2-70b-it": os.path.join(MODEL_PATH_ROOT, "llama2_70b_it"),

    "llama3-8b": os.path.join(MODEL_PATH_ROOT, "llama3_8b"),
    "llama3-8b-it": os.path.join(MODEL_PATH_ROOT, "llama3_8b_it"),
    "llama3-70b": os.path.join(MODEL_PATH_ROOT, "llama3_70b"),
    "llama3-70b-it": os.path.join(MODEL_PATH_ROOT, "llama3_70b_it"),
    "llama3.1-8b": os.path.join(MODEL_PATH_ROOT, "llama3.1_8b"),
    "llama3.1-8b-it": os.path.join(MODEL_PATH_ROOT, "llama3.1_8b_it"),
    "llama3.1-70b": os.path.join(MODEL_PATH_ROOT, "llama3.1_70b"),
    "llama3.1-70b-it": os.path.join(MODEL_PATH_ROOT, "llama3.1_70b_it"),
    "llama3.2-1b": os.path.join(MODEL_PATH_ROOT, "llama3.2_1b"),
    "llama3.2-1b-it": os.path.join(MODEL_PATH_ROOT, "llama3.2_1b_it"),
    "llama3.2-3b": os.path.join(MODEL_PATH_ROOT, "llama3.2_3b"),
    "llama3.2-3b-it": os.path.join(MODEL_PATH_ROOT, "llama3.2_3b_it"),
}

QWEN_MODEL_PATHS = {
    "qwen2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
}

def get_model_type(model_name: str):
    if "bert" in model_name:
        model_type = "bert"  
    elif model_name.startswith("llama2"):
        model_type = "llama2"
    elif model_name.startswith("llama3"):
        model_type = "llama3"
    elif model_name.startswith("qwen"):
        model_type = "qwen"
    elif model_name.startswith("phi3"):
        model_type = "phi3"
    return model_type


def initialize_model_and_tokenizer(model_name: str, gpu: int=0, only_tokenizer: bool=False):
    model = None
    if model_name in BERT_MODELS:
        device = "cuda:0" if gpu < 0 else f'cuda:{gpu}'
        tokenizer = BertTokenizer.from_pretrained(BERT_MODELS[model_name], device_map=device)
        if not only_tokenizer:
            model = BertLMHeadModel.from_pretrained(BERT_MODELS[model_name], device_map=device)
    elif model_name in LLAMA_MODEL_PATHS or model_name or model_name in QWEN_MODEL_PATHS:
        if model_name in LLAMA_MODEL_PATHS:
            model_path = LLAMA_MODEL_PATHS[model_name] 
        elif model_name in QWEN_MODEL_PATHS:
            model_path = QWEN_MODEL_PATHS[model_name] 
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        if not only_tokenizer:
            if gpu == -1:
                device = "balanced_low_0" 
            elif gpu == -2:
                device = "auto" 
            else:
                device = f"cuda:{gpu}"
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float32, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Model {model_name} not supported")

    if not only_tokenizer:
        model.eval()

    return model, tokenizer
