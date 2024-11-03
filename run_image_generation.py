import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F

from models import get_mask_chedule
from training.utils import image_transform
from training.prompting_utils import create_attention_mask_predict_next


def extrapolate(cfg, model, vq_model, uni_prompting, device, mask_token_id):
    prompt = [p for p in cfg.prompt.split(" *** ") if len(p) != 0]
    extra_direction = [d for d in cfg.extra_direction.split(" *** ") if len(d) != 0]
    print(prompt, extra_direction)
    W = cfg.dataset.params.resolution // 16
    for id, (prt, direction) in enumerate(zip(prompt, extra_direction)):
        prt = [prt] * cfg.training.batch_size
        if id == 0:
            extrapolation_image = Image.open(cfg.image_path).convert("RGB")
            extrapolation_image = image_transform(extrapolation_image,
                                                    resolution=cfg.dataset.params.resolution).to(device)

            B, _, _ = extrapolation_image.shape
            extrapolation_image = extrapolation_image.unsqueeze(0)
            extrapolation_image_tokens = vq_model.get_code(extrapolation_image) + len(uni_prompting.text_tokenizer)
            extrapolation_image_tokens = extrapolation_image_tokens.reshape(1,
                                                                            cfg.dataset.params.resolution // 16,
                                                                            cfg.dataset.params.resolution // 16)
            extrapolation_image_tokens = extrapolation_image_tokens.repeat(cfg.training.batch_size, 1, 1)
        else:


            extrapolation_image_tokens = gen_token_ids + len(uni_prompting.text_tokenizer)

        image_left_part = extrapolation_image_tokens[:, :, :-(W//2-cfg.offset)] - len(uni_prompting.text_tokenizer)
        image_right_part = extrapolation_image_tokens[:, :, W//2-cfg.offset:] - len(uni_prompting.text_tokenizer)
        image_up_part = extrapolation_image_tokens[:, :-(W//2-cfg.offset), :] - len(uni_prompting.text_tokenizer)
        image_down_part = extrapolation_image_tokens[:, W//2-cfg.offset:, :] - len(uni_prompting.text_tokenizer)

        if direction in ['left', 'right']:
            extrapolation_mask = torch.zeros((cfg.training.batch_size,
                                                cfg.dataset.params.resolution // 16,
                                                cfg.dataset.params.resolution // 16 // 2 + cfg.offset),
                                                dtype=torch.int64, device=device) + mask_token_id
        else:
            extrapolation_mask = torch.zeros((cfg.training.batch_size,
                                                cfg.dataset.params.resolution // 16 // 2 + cfg.offset,
                                                cfg.dataset.params.resolution // 16),
                                                dtype=torch.int64, device=device) + mask_token_id

        if direction == 'left':
            extrapolation_image_tokens = torch.cat(
                [extrapolation_mask, extrapolation_image_tokens[:, :, :W//2-cfg.offset]], dim=-1)
        elif direction == 'right':
            extrapolation_image_tokens = torch.cat(
                [extrapolation_image_tokens[:, :, -(W//2-cfg.offset):], extrapolation_mask], dim=-1)
        elif direction == 'up':
            extrapolation_image_tokens = torch.cat(
                [extrapolation_mask, extrapolation_image_tokens[:, :W // 2 - cfg.offset, :]], dim=-2)
        else:
            extrapolation_image_tokens = torch.cat(
                [extrapolation_image_tokens[:, -(W // 2 - cfg.offset):, :], extrapolation_mask], dim=-2)

        extrapolation_image_tokens = extrapolation_image_tokens.reshape(cfg.training.batch_size, -1)

        input_ids, _ = uni_prompting((prt, extrapolation_image_tokens), 't2i_gen')

        if cfg.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prt), extrapolation_image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if cfg.get("mask_schedule", None) is not None:
            schedule = cfg.mask_schedule.schedule
            args = cfg.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(cfg.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=cfg.training.guidance_scale,
                temperature=cfg.training.get("generation_temperature", 1.0),
                timesteps=cfg.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=cfg.training.get("noise_type", "mask"),
                seq_len=cfg.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=cfg,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=cfg.model.showo.codebook_size - 1, min=0)
        gen_token_ids = gen_token_ids.reshape(cfg.training.batch_size,
                                                cfg.dataset.params.resolution // 16,
                                                cfg.dataset.params.resolution // 16)
        if direction == 'left':
            gen_token_ids = torch.cat([gen_token_ids, image_right_part], dim=-1)
        elif direction == 'right':
            gen_token_ids = torch.cat([image_left_part, gen_token_ids], dim=-1)
        elif direction == 'up':
            gen_token_ids = torch.cat([gen_token_ids, image_down_part], dim=-2)
        else:
            gen_token_ids = torch.cat([image_left_part, gen_token_ids], dim=-2)

    _, h, w = gen_token_ids.shape
    gen_token_ids = gen_token_ids.reshape(cfg.training.batch_size, -1)
    images = vq_model.decode_code(gen_token_ids, shape=(h, w))

    return images


def inpaint(cfg, model, vq_model, uni_prompting, device, mask_token_id):
    prompt = [cfg.prompt] * cfg.batch_size
    inpainting_image = Image.open(cfg.image_path).convert("RGB")
    inpainting_mask = Image.open(cfg.inpainting_mask_path).convert("L")

    inpainting_image = image_transform(inpainting_image, resolution=cfg.dataset.params.resolution).to(device)
    inpainting_mask = image_transform(inpainting_mask, resolution=cfg.dataset.params.resolution, normalize=False)

    # record original image and inpainting mask
    images = torch.clamp(
        (torch.stack([inpainting_image, inpainting_mask.repeat(3, 1, 1).to(device)], dim=0) + 1.0) / 2.0,
        min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    labels = ['original image', 'inpainting mask']

    inpainting_image = inpainting_image.unsqueeze(0).repeat(cfg.training.batch_size, 1, 1, 1)

    inpainting_mask = inpainting_mask.unsqueeze(0).to(device)
    inpainting_mask = F.interpolate(inpainting_mask, size=cfg.dataset.params.resolution // 16, mode='bicubic')
    inpainting_mask = inpainting_mask.repeat(cfg.training.batch_size, 1, 1, 1)

    inpainting_mask[inpainting_mask < 0.5] = 0
    inpainting_mask[inpainting_mask >= 0.5] = 1

    inpainting_mask = inpainting_mask.reshape(cfg.training.batch_size, -1)
    inpainting_mask = inpainting_mask.to(torch.bool)

    inpainting_image_tokens = vq_model.get_code(inpainting_image) + len(uni_prompting.text_tokenizer)
    inpainting_image_tokens[inpainting_mask] = mask_token_id

    input_ids, _ = uni_prompting((prompt, inpainting_image_tokens), 't2i_gen')

    if cfg.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(prompt), inpainting_image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True)
    else:
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True)
        uncond_input_ids = None

    if cfg.get("mask_schedule", None) is not None:
        schedule = cfg.mask_schedule.schedule
        args = cfg.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(cfg.training.get("mask_schedule", "cosine"))

    with torch.no_grad():
        gen_token_ids = model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=cfg.training.guidance_scale,
            temperature=cfg.training.get("generation_temperature", 1.0),
            timesteps=cfg.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=cfg.training.get("noise_type", "mask"),
            seq_len=cfg.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=cfg,
        )

    gen_token_ids = torch.clamp(gen_token_ids, max=cfg.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)
    return images