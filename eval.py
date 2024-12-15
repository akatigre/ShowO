# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import time
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer
from torchvision.utils import make_grid

import torch
import hydra
import logging
from omegaconf import DictConfig
from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console

from models import Showo
from training.prompting_utils import UniversalPrompting
from tqdm import trange
from utils import set_seed, load_metadata, get_vq_model_class
from models import MAGVITv2
from generate import generate
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(
        console=Console(theme=Theme({"logging.level.success": "green"}))
    )]
)

log = logging.getLogger("rich")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config: {cfg}")
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    model_params = cfg.model_params
    assert model_params.model_name == "Show-o", "Model name should be Show-O"
    # Create folder to save images
    if cfg.teacher_force:
        folder_name = f"reconstructed/{cfg.model_params.model_name}/{cfg.decode}{cfg.cfg_scale}_tf{cfg.teacher_force_upto*100}"
    else:
        folder_name = f"generated/{cfg.model_params.model_name}/{cfg.decode}{cfg.cfg_scale}"

    if cfg.nonmyopic:
        folder_name += "_nonmyopic"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_params.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=model_params.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=model_params.training.cond_dropout_prob)

    vq_model: MAGVITv2 = get_vq_model_class(model_params.model.vq_model.type)
    vq_model = vq_model.from_pretrained(model_params.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model: Showo = Showo.from_pretrained(model_params.model.showo.pretrained_model_path).to(device)
    mask_token_id = model.config.mask_token_id
    model.eval()
    
    # Prepare prompts and metadata
    val_prompts, metadatas = load_metadata(cfg)
    categories = val_prompts.get("categories", None)
    N = len(val_prompts['prompts'])
    
    batch_size = 8
    per_prompt_images = []
    for start_idx in trange(0, N, batch_size):
        gt_path = None
        if cfg.benchmark.name=="geneval":
            prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
            names = val_prompts['name'][start_idx: start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]
            metas = metadatas[start_idx: start_idx + batch_size]
            for save, metadata in zip(save_path[::4], metas[::4]):
                os.makedirs(save.parent, exist_ok=True)
                with open(os.path.join(save.parent, "metadata.jsonl"), "w") as fp:
                    json.dump(metadata, fp)

        elif cfg.benchmark.name=="dpgbench":
            prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
            names = val_prompts['name'][start_idx: start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]

        elif cfg.benchmark.name=="mjhq":
            cats = categories[start_idx: start_idx + batch_size] if categories is not None else None
            gt_path = [Path(cfg.benchmark.outdirs).parent / 'root' / cat / name for cat, name in zip(cats, names)]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / cat / name for cat, name in zip(cats, names) if not (Path(cfg.benchmark.outdirs) / folder_name / cat / name).exists()]
            for save in save_path:
                os.makedirs(save.parent, exist_ok=True)
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        if not len(save_path):
            continue
        
        max_seq_length = model_params.dataset.preprocessing.max_seq_length # maximum number of text tokens.
        num_vq_tokens = model_params.model.showo.num_vq_tokens # number of image tokens
        num_new_special_tokens = model_params.model.showo.num_new_special_tokens # number of new special tokens
        llm_vocab_size = model_params.model.showo.llm_vocab_size # size of the language model vocabulary
        generation_timesteps = model_params.generation_timesteps # number of timesteps for generation
        generation_temperature = model_params.generation_temperature # temperature for generation
        codebook_size = model_params.model.showo.codebook_size

        # generate image from text
        start_time = time.time()
        images = generate(
                cfg,
                model,
                vq_model,
                uni_prompting,
                device,
                mask_token_id = mask_token_id,
                prompts = prompts,
                image_path = gt_path,
                force_teacher = cfg.teacher_force,
                teacher_force_upto = cfg.teacher_force_upto,
                max_seq_length = max_seq_length,
                num_vq_tokens = num_vq_tokens,
                num_new_special_tokens = num_new_special_tokens,
                llm_vocab_size = llm_vocab_size,
                generation_timesteps = generation_timesteps,
                generation_temperature = generation_temperature,
                codebook_size = codebook_size,
        )
        end_time = time.time()
        
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
        if cfg.benchmark.name=="dpgbench":
            per_prompt_images.extend([image for image in images])
            for img_idx in range(0, len(per_prompt_images), cfg.benchmark.batch):
                images = make_grid(per_prompt_images[img_idx: img_idx + cfg.benchmark.batch], nrow=2)
                images = images.permute(1, 2, 0).cpu().numpy().astype('uint8')
                images = Image.fromarray(images)
                save_path[img_idx].parent.mkdir(parents=True, exist_ok=True)
                images.save(save_path[img_idx])
            per_prompt_images = []
        else:
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            for save_at, image in zip(save_path, pil_images):
                save_at.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_at)

            
        log.info(f"Total time taken: {end_time - start_time} generating {prompts} with {cfg.decode}")
    
if __name__=="__main__":
    main()