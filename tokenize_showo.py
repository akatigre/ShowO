import torch
from training.utils import image_transform
from PIL import Image
from models import Showo, MAGVITv2
from training.prompting_utils import UniversalPrompting
from typing import List

def tokenize_image(
    image_path: List[str],
    device,
    vq_model: MAGVITv2,
    uni_prompting: UniversalPrompting,
    ):
    imgs = []
    for img in image_path:
        image_ori = Image.open(img).convert("RGB")
        image = image_transform(image_ori, resolution=512).to(device)
        imgs.append(image)
    image = torch.stack(imgs)

    image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer) # 1, 1024

    input_ids = torch.cat([
        (torch.ones(image_tokens.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
        image_tokens,
        (torch.ones(image_tokens.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
    ], dim=1).long() # 1, 1026
    
    return input_ids # [soi] [image tokens] [eoi]


def tokenize_text(
    prompt: List[str],
    cfg,
    device: str,
    mask_token_id,
    uni_prompting: UniversalPrompting,
    ):
    model_params = cfg.model_params
    batch_size = len(prompt)
    image_ids = torch.ones((batch_size, model_params.model.showo.num_vq_tokens),
                        dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting((prompt, image_ids), 't2i_gen')
    return input_ids # [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]