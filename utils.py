import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed
from collections import defaultdict
import os
import json

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)


def load_metadata(cfg):
    val_prompts = defaultdict(list)
    
    prompt_path = cfg.benchmark.prompts
    if cfg.benchmark.name=="dpgbench":
        prompt_lists = sorted(os.listdir(prompt_path))
        for p in prompt_lists:
            full_path = os.path.join(prompt_path, p)
            with open(full_path, 'r') as f:
                line = f.read().splitlines()[0]
            val_prompts["name"].append(p.replace("txt", "png"))
            val_prompts["prompts"].append(line)
        
    elif cfg.benchmark.name=="geneval":
        with open(prompt_path) as f:
            metadatas = [json.loads(line) for line in f]
        val_prompts["prompts"] = [metadata['prompt'] for metadata in metadatas]
        val_prompts["name"] = [f"{idx:0>5}" for idx in range(len(val_prompts["prompts"]))]
        
    elif cfg.benchmark.name=="mjhq":
        with open(prompt_path, "r") as f:
            metadatas = json.load(f)
        file_names = sorted(list(metadatas.keys()))
        
        val_prompts["name"] = [file_name + ".png" for file_name in file_names]
        val_prompts["prompts"] = [metadatas[filename]["prompt"] for filename in file_names]
        val_prompts["categories"] = [metadatas[filename]["category"] for filename in file_names]
        
        
    else:
        raise NotImplementedError(f"Unknown benchmark name: {cfg.benchmark.name}")
    return val_prompts
