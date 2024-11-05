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
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import numpy as np
import torch
from tqdm import trange, tqdm

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from transformers import AutoTokenizer
from change_showo import change_showo_forward, change_phi_forward, change_phi_decoder_layer_forward, cfg_pag_forward

import hydra
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from run_image_generation import inpaint, extrapolate
from models.sampling import mask_by_random_topk
from change_showo import change_phi_forward, change_phi_decoder_layer_forward, cfg_pag_forward
from utils import set_seed, load_metadata
import logging
from rich.logging import RichHandler
os.system('export PYTHONPATH="${PYTHONPATH}:/home/server08/yoonjeon_workspace/MMAR"')

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    model_params = cfg.model_params
    assert model_params.model_name == "Show-o", "Model name should be Show-o"
    
    set_seed(seed=cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    enable_pag = cfg.pag_scale > 0.0
    enable_cfg = cfg.cfg_scale > 1.0
    enable_cd = cfg.cd_beta < 1.0
    log.info(f"Enable PAG: {enable_pag}, Enable CFG: {enable_cfg}, Enable CD: {enable_cd}")
    
    # Create folder to save images
    output_path = str(Path(cfg.benchmark.outdirs) / cfg.model_params.model_name)
    log.info(f"Output path: {output_path}")
    folder_name = "generated"
    if enable_pag: 
        folder_name += f"_pag:{cfg.pag_scale}_layer:{cfg.layer_types}"
        output_path += f"_PAG_{cfg.pag_scale}"
    if enable_cfg: 
        folder_name += f"_cfg{cfg.cfg_scale}"
        output_path += f"_CFG_{cfg.cfg_scale}"
    if enable_cd:
        folder_name += f"_cd{cfg.cd_beta}"
        output_path += f"_CD_{cfg.cd_beta}"

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_params.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=model_params.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=model_params.training.cond_dropout_prob)

    vq_model = get_vq_model_class(model_params.model.vq_model.type)
    vq_model = vq_model.from_pretrained(model_params.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(model_params.model.showo.pretrained_model_path).to(device)
    model.eval()

    #! Change layer forward functions to support PAG
    decoder_layers = model.showo.model.layers
    batch_size = cfg.benchmark.batch
    if cfg.layer_types=="all":
        layer_idxs = range(len(decoder_layers))
    elif cfg.layer_types=="early":
        layer_idxs = range(len(decoder_layers) // 3)
    elif cfg.layer_types=="middle":
        layer_idxs = range(len(decoder_layers) // 3, 2 * len(decoder_layers) // 3)
    elif cfg.layer_types=="late":
        layer_idxs = range(2 * len(decoder_layers) // 3, len(decoder_layers))
    log.info(f"Total layers : {len(decoder_layers)}, Changing layers: {layer_idxs}")


    model.showo = change_showo_forward(model.showo)
    model.showo.model = change_phi_forward(model.showo.model)
    for idx, layer in enumerate(decoder_layers):
        layer = change_phi_decoder_layer_forward(layer)
        layer.self_attn = cfg_pag_forward(layer.self_attn)
        if idx in layer_idxs:
            layer.self_attn.pag_layer = True
        else:
            layer.self_attn.pag_layer = False
     
    log.info("Layer structure changed successfully")
    mask_token_id = model.config.mask_token_id
 
    val_prompts = load_metadata(cfg)
    categories = val_prompts.get("categories", None)
    # load from users passed arguments
    for idx, (prompt, name) in tqdm(enumerate(zip(val_prompts['prompts'], val_prompts['name'])), desc="Generating images"):
        
        cat = categories[idx] if categories is not None else None
        outpath = Path(output_path) / name if cat is None else Path(output_path) / cat / name
        
        if cfg.benchmark.name=="geneval":
            os.makedirs(outpath, exist_ok=True)
            if len(os.listdir(outpath)) > batch_size:
                continue
        elif cfg.benchmark.name=="mjhq" or cfg.benchmark.name=="dpgbench":
            os.makedirs(outpath.parent, exist_ok=True)
            if os.path.exists(outpath):
                continue
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        log.info(f"With Prompt '{prompt}' generating {batch_size} images")
        
        if cfg.model_params.mode == "inpainting":
            images = inpaint(cfg, model, vq_model, uni_prompting, device, mask_token_id)
        elif cfg.model_params.mode == 'extrapolation':
            images = extrapolate(cfg, model, vq_model, uni_prompting, device, mask_token_id)
        elif cfg.model_params.mode == 't2i':
            image_tokens = torch.ones((batch_size, model_params.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting(([prompt] * batch_size, image_tokens), 't2i_gen')

            total_input_ids = []
            total_input_ids.append(input_ids) # cond | cond, cfg | cond, pag | cond, cfg, pag

            if cfg.cfg_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * batch_size, image_tokens), 't2i_gen')
                total_input_ids.append(uncond_input_ids)
            else:
                uncond_input_ids = None

            if cfg.pag_scale > 0:
                pag_input_ids, _ = uni_prompting(([prompt] * batch_size, image_tokens), 't2i_gen')
                total_input_ids.append(pag_input_ids)

            else:
                pag_input_ids = None

            total_input_ids = torch.cat(total_input_ids, dim=0)
            attention_mask = create_attention_mask_predict_next(total_input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)

            if model_params.get("mask_schedule", None) is not None:
                schedule = model_params.mask_schedule.schedule
                args = model_params.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(model_params.training.get("mask_schedule", "cosine"))
            
            
            cfg_scale = cfg.cfg_scale
            pag_scale = cfg.pag_scale
            temperature = model_params.generation_temperature
            timesteps = model_params.generation_timesteps
            noise_schedule = mask_schedule
            # begin with all image token ids masked
            num_vq_tokens = model_params.model.showo.num_vq_tokens
            num_new_special_tokens = model_params.model.showo.num_new_special_tokens

            input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
            input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                        mask_token_id,
                                                        input_ids_minus_lm_vocab_size - model_params.model.showo.llm_vocab_size - num_new_special_tokens)

            # for classifier-free guidance
            if uncond_input_ids is not None:
                uncond_prefix = uncond_input_ids[:, :model_params.dataset.preprocessing.max_seq_length + 1]

            if pag_input_ids is not None:
                pag_prefix = pag_input_ids[:, :model_params.dataset.preprocessing.max_seq_length + 1]
            
            enable_cfg = uncond_input_ids is not None and cfg_scale > 0
            enable_pag = pag_input_ids is not None and pag_scale > 0

            extras = defaultdict(list)
            for step in range(timesteps): # cond | cond, uncond | cond, pag | cond, uncond, pag
                if enable_cfg and not enable_pag:
                    uncond_input_ids = torch.cat(
                        [uncond_prefix, input_ids[:, model_params.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    model_input = torch.cat([input_ids, uncond_input_ids])
                    with torch.no_grad():
                        cond_logits, uncond_logits = model.showo(
                            input_ids=model_input, 
                            attention_mask=attention_mask,
                            enable_cfg=enable_cfg,
                            enable_pag=enable_pag,
                            use_cache=False,
                            prefix_len=len(input_ids),
                            )['logits'].chunk(2)
                        
                    logits = (1 + cfg_scale) * cond_logits - cfg_scale * uncond_logits
                elif enable_pag and not enable_cfg:
                    pag_input_ids = torch.cat(
                        [pag_prefix, input_ids[:, model_params.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    model_input = torch.cat([input_ids, pag_input_ids])
                    with torch.no_grad():
                        cond_logits, pag_logits = model.showo(
                            input_ids=model_input, 
                            attention_mask=attention_mask,
                            enable_cfg=enable_cfg,
                            enable_pag=enable_pag,
                            use_cache=False,
                            prefix_len=len(input_ids),
                            )['logits'].chunk(2)
                    logits = (1 + pag_scale) * cond_logits - pag_scale * pag_logits

                elif enable_pag and enable_cfg:
                    uncond_input_ids = torch.cat(
                        [uncond_prefix, input_ids[:, model_params.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    pag_input_ids = torch.cat(
                        [pag_prefix, input_ids[:, model_params.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                    model_input = torch.cat([input_ids, uncond_input_ids, pag_input_ids])
                    with torch.no_grad():
                        # cond_logits, uncond_logits, pag_logits = model(model_input, attention_mask=attention_mask).chunk(3)
                        cond_logits, uncond_logits, pag_logits = model.showo(
                            input_ids=model_input, 
                            attention_mask=attention_mask,
                            enable_cfg=enable_cfg,
                            enable_pag=enable_pag,
                            use_cache=False,
                            prefix_len=len(input_ids),
                            )['logits'].chunk(3)
                    logits = (1 + cfg_scale - pag_scale) * cond_logits - cfg_scale * uncond_logits + pag_scale * pag_logits
                    
                else:
                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                    
                logits = logits[:, -(num_vq_tokens + 1):-1, model_params.model.showo.llm_vocab_size + num_new_special_tokens:-1]
                topk_logits = logits.topk(100, dim=-1)
                extras["logit_hist_vals"].append(topk_logits.values)
                extras["logit_hist_inds"].append(topk_logits.indices)
                probs = logits.softmax(dim=-1)
                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=None)[:, 0].view(*logits.shape[:-1])

                unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                # Defines the mask ratio for the next round. The number to mask out is
                # determined by mask_ratio * unknown_number_in_the_beginning.
                ratio = 1.0 * (step + 1) / timesteps
                mask_ratio = noise_schedule(torch.tensor(ratio))
                # Computes the probabilities of each selected tokens.
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)

                # Ignores the tokens given in the input by overwriting their confidence.
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                # Gets mask lens for each sample in the batch according to the mask ratio.
                mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
                # Keeps at least one of prediction in this round and also masks out at least
                # one and for the next iteration
                mask_len = torch.max(
                    torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
                )
                # Adds noise for randomness
                temperature = temperature * (1.0 - ratio)
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=None)
                # Masks tokens with lower confidence.
                input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                            sampled_ids + model_params.model.showo.llm_vocab_size
                                                            + num_new_special_tokens)
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

            sampled_ids = torch.clamp(sampled_ids, max=model_params.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(sampled_ids)
        else:
            raise ValueError(f"mode {model_params.mode} not supported.")
        
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        
        if cfg.benchmark.name=="geneval":
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]
            sample_count = 0
            for sample in pil_images:
                sample.save(os.path.join(outpath, f"{sample_count:05}.png"))
                sample_count += 1
            (Path(outpath) / str(sample_count)).mkdir(exist_ok=True)
            torch.save(extras, Path(outpath) / str(sample_count) / "extras.pt")
            torch.save(sampled_ids, Path(outpath) / str(sample_count)/ "generated_tokens.pt")
        elif cfg.benchmark.name=="dpgbench":
            from torchvision.utils import make_grid
            images = make_grid(images, nrow=2)
            images = images.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            images = Image.fromarray(images)
            images.save(outpath)
            (Path(outpath.parent) / outpath.stem).mkdir(exist_ok=True)
            torch.save(extras, Path(outpath.parent) / outpath.stem / "extras.pt")
            torch.save(sampled_ids, Path(outpath.parent) / outpath.stem / "generated_tokens.pt")
        elif cfg.benchmark.name=="mjhq":
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            images = Image.fromarray(images[0])
            images.save(outpath)
            (Path(outpath.parent) / outpath.stem).mkdir(exist_ok=True)
            torch.save(extras, Path(outpath.parent) / outpath.stem / "extras.pt")
            torch.save(sampled_ids, Path(outpath.parent) / outpath.stem / "generated_tokens.pt")
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        log.info(f"Generated image saved at {outpath}")

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    
if __name__ == '__main__':
    try:
        main()
    except:
        log.exception("Error!")