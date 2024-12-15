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

import torch
from run_image_generation import inpaint, extrapolate
from tqdm import trange
from decode import dola_greedy_decode, pag_decode, gt_decode
import logging
from rich.logging import RichHandler
from tokenize_showo import tokenize_text, tokenize_image
from sample import sample_with_mask
from models import MAGVITv2, Showo
from models.phi import PhiForCausalLM
from mask import create_attn_mask
from typing import List
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)

def generate(
        cfg,
        model,
        vq_model,
        uni_prompting,
        device,
        mask_token_id,
        prompts: List[str],
        image_path: List,
        force_teacher: bool,
        teacher_force_upto: float,
        max_seq_length: int,
        num_vq_tokens,
        num_new_special_tokens,
        llm_vocab_size,
        generation_timesteps,
        generation_temperature,
        codebook_size,
    ):
    
    if cfg.model_params.mode == "inpainting":
        images = inpaint(cfg, model, vq_model, uni_prompting, device)
    elif cfg.model_params.mode == 'extrapolation':
        images = extrapolate(cfg, model, vq_model, uni_prompting, device)
    elif cfg.model_params.mode == 't2i':
        images = t2i(cfg, vq_model, model, uni_prompting, device, 
                    prompt=prompts, 
                    image_path = image_path, 
                    mask_token_id = mask_token_id, 
                    force_teacher=force_teacher, 
                    teacher_force_upto=teacher_force_upto,
                    max_seq_length = max_seq_length,
                    num_vq_tokens = num_vq_tokens,
                    num_new_special_tokens = num_new_special_tokens,
                    generation_timesteps = generation_timesteps,
                    llm_vocab_size = llm_vocab_size,
                    generation_temperature = generation_temperature,
                    codebook_size = codebook_size,
                    nonmyopic = cfg.nonmyopic,
                    )
    
    return images

def t2i(
    cfg,
    vq_model: MAGVITv2,
    model: Showo,
    uni_prompting,
    device,
    prompt: List[str],
    image_path,
    mask_token_id,
    max_seq_length,
    num_vq_tokens,
    num_new_special_tokens,
    llm_vocab_size,
    generation_timesteps,
    generation_temperature,
    codebook_size,
    generator = None,
    force_teacher = False,
    teacher_force_upto = 1.0,
    nonmyopic = False,
):
    input_ids_cond = tokenize_text(prompt, cfg, device, mask_token_id, uni_prompting) # [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
    input_ids_uncond = tokenize_text([""] * len(prompt), cfg, device, mask_token_id, uni_prompting) # [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
    if force_teacher:
        assert image_path is not None
        input_ids_gt = tokenize_image(image_path, device=device, vq_model=vq_model, uni_prompting=uni_prompting) # [soi] [image_tokens] [eoi] # added llm text vocab size

    image_start = max_seq_length + 1 # add one for task specific token
    
    image_tokens = input_ids_cond[:, -(num_vq_tokens + 1): -1].clone()
    input_ids_minus_lm_vocab_size = torch.where(
        image_tokens == mask_token_id,
        mask_token_id, # mask all image tokens at start
        image_tokens - llm_vocab_size - num_new_special_tokens # subtract vocab size on text tokens
        )
    
    pad_id, soi_id, eoi_id = [int(uni_prompting.sptids_dict[f'<|{tok}|>']) for tok in ['pad', 'soi', 'eoi']]
    # total_input_ids =  torch.cat([input_ids_cond, input_ids_uncond, input_ids_cond], dim=0)
    total_input_ids = torch.cat([input_ids_cond, input_ids_uncond], dim=0)
    attention_mask = create_attn_mask(total_input_ids, rm_pad_in_image = True, pad_id = pad_id, soi_id = soi_id, eoi_id = eoi_id) # makes causal mask with text and image tokens
    # attention_mask[-1, :, image_start : , image_start : ] = torch.diag(torch.ones(attention_mask.shape[-1] - image_start, dtype=torch.bool)).unsqueeze(dim=0).unsqueeze(dim=0) # mask the image tokens in the pag step
    
    for step in range(generation_timesteps): # cond, uncond, pag
        #! Add mask to predicted tokens
        logits = single_step(
            model = model,
            attention_mask = attention_mask,
            input_ids = total_input_ids, # prefix | image tokens
            mask_token_id = mask_token_id,
            decode = cfg.decode,
            cfg = cfg,
        )
        
        logits_image = logits[:, -(num_vq_tokens + 1) : -1 , llm_vocab_size + num_new_special_tokens:-1]
        probs = logits_image.softmax(dim=-1)
        sampled = probs.reshape(-1, logits_image.size(-1))
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits_image.shape[:-1])
        # begin with all image token ids masked
        
        unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
        sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]) # Computes the probabilities of each selected tokens.
        selected_probs = selected_probs.squeeze(-1)
        masking = sample_with_mask(
            selected_probs = selected_probs, # 1, n_tokens
            unknown_map = unknown_map,
            num_vq_tokens = num_vq_tokens,
            temperature = generation_temperature,
            ratio = 1.0 * (step + 1) / generation_timesteps,
        )
        if nonmyopic:
            input_ids_cond_ = input_ids_cond.clone()
            input_ids_cond_[:, -(num_vq_tokens + 1):-1] = sampled_ids + llm_vocab_size + num_new_special_tokens
            with torch.no_grad():
                outputs = model.showo(
                            input_ids=input_ids_cond_, 
                            attention_mask=attention_mask[0:1], # (batch, 1, tgt_len, src_len) where padding elements are indicated by very large negative values.
                            use_cache = False,
                            output_hidden_states = False,
                            )
            logits_ = outputs['logits']
            logits = logits * torch.exp(logits_)
            logits_image = logits[:, -(num_vq_tokens + 1) : -1 , llm_vocab_size + num_new_special_tokens:-1]
            probs = logits_image.softmax(dim=-1)
            sampled = probs.reshape(-1, logits_image.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits_image.shape[:-1])
            
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]) # Computes the probabilities of each selected tokens.
            selected_probs = selected_probs.squeeze(-1)
            masking = sample_with_mask(
                selected_probs = selected_probs, # 1, n_tokens
                unknown_map = unknown_map,
                num_vq_tokens = num_vq_tokens,
                temperature = generation_temperature,
                ratio = 1.0 * (step + 1) / generation_timesteps,
            )
        input_ids_cond[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id, sampled_ids + llm_vocab_size + num_new_special_tokens)
        input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
        
        if force_teacher and step < generation_timesteps * teacher_force_upto:
            
            input_ids_cond[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id, input_ids_gt[:, 1:-1])
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, input_ids_gt[:, 1:-1] - llm_vocab_size - num_new_special_tokens)
                
        input_ids_uncond = torch.cat([input_ids_uncond[:, : image_start], input_ids_cond[:, image_start : ]], dim=1) # update image tokens
        # total_input_ids =  torch.cat([input_ids_cond, input_ids_uncond, input_ids_cond], dim=0)
        total_input_ids = torch.cat([input_ids_cond, input_ids_uncond], dim=0)
        
    sampled_ids = torch.clamp(sampled_ids, max=codebook_size - 1, min=0)
    images = vq_model.decode_code(sampled_ids)
    return images

def single_step(
    model,
    attention_mask,
    input_ids,
    mask_token_id,
    decode="dola",
    cfg=None,
    ):
    
    showo: PhiForCausalLM = model.showo
    
    with torch.no_grad():
        outputs = showo(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, # (batch, 1, tgt_len, src_len) where padding elements are indicated by very large negative values.
                    use_cache=False,
                    output_hidden_states = True if decode == "dola" else False,
                    )
        logits = outputs['logits']
    
    if decode == "dola":
        layer_selected = [3, 6, 9, 12, 15, 18, 21, 24]
        premature_logits = torch.stack([model.showo.lm_head(prem_state) for idx, prem_state in enumerate(outputs['hidden_states']) if idx in layer_selected], dim=0) 
        # n_layers 25, n_channels 3, n_tokens 1155, n_vocab 58498
        mature_logits = logits
        logits = dola_greedy_decode(
            mature_logits = mature_logits,
            premature_logits = premature_logits,
            layer_logit_dist = None,
            relative_top = 0.1
            )
    elif decode in ["cfg", "pag", "cd"]:
        # logit_cond, logit_uncond, logit_ptb_cond = logits.chunk(3)
        logit_cond, logit_uncond = logits.chunk(2)
        gamma = cfg.cfg_scale
        logits = (1 - gamma) * logit_uncond + gamma * logit_cond
    else:
        # logit_cond, logit_uncond, logit_ptb_cond = logits.chunk(3)
        logit_cond, logit_uncond = logits.chunk(2)
        logits = logit_cond
        
    return logits