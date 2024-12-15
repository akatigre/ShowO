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

import time
from PIL import Image
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer

import torch
import hydra
import logging
from omegaconf import DictConfig
from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console

from models import Showo
from training.prompting_utils import UniversalPrompting
import utils
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
    
    cfg.benchmark.name = 'mjhq'
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
    prompt_idx = cfg.prompt_idx
    prompt = val_prompts['prompts'][prompt_idx]
    name = val_prompts['name'][prompt_idx]              
    cat = categories[prompt_idx] if categories is not None else None
    gt_path = Path(cfg.benchmark.outdirs).parent / 'root' / cat / name
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
            prompt = prompt,
            image_path = gt_path,
            batch_size = cfg.benchmark.batch,
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
    log.info(f"Total time taken: {end_time - start_time} generating {prompt} with {cfg.decode}")
    
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
    images = images.squeeze(dim=0).detach().cpu().permute(1, 2, 0).numpy().astype("uint8")
    images = Image.fromarray(images)
    w, h = images.size
    gt_images = Image.open(str(gt_path)).resize((w, h))
    image = utils.save_image(images, gt_images, prompt, cfg, cfg.teacher_force_upto)
    img_name = f"test{prompt_idx}"
    decode = cfg.decode if cfg.decode == '_vanilla' else f"_{cfg.decode}_{cfg.cfg_scale}"
    if cfg.teacher_force:
        folder_path = f"./outputs/reconstructed"
        if cfg.teacher_force_upto < 1.0:
            img_name += f"{decode}_teach{cfg.teacher_force_upto * 100:.0f}"
        else:
            img_name += "_full_recon"
    else:
        folder_path = f"./outputs/generated"
        img_name += decode

    img_name = f"{img_name}_nonmyopic" if cfg.nonmyopic else img_name

    save_path = f"{folder_path}/{img_name}.jpg"
    image.save(save_path)
    log.info(f"Save into {save_path}")
    
if __name__=="__main__":
    main()