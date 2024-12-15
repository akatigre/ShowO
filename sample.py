import torch
from models.sampling import mask_by_random_topk
from models import get_mask_chedule

def sample_with_mask(
    selected_probs,
    unknown_map,
    num_vq_tokens,
    temperature,
    ratio,
):
    """
    Args:
        probs (torch.Tensor): 1, n_tokens, n_vocab
        sampled_ids (torch.Tensor): 1, n_tokens
        step (int): current step
        mask_token_id (int): mask token id
        model_params (_type_): model params
        input_ids_minus_lm_vocab_size (_type_): input ids minus lm vocab size
    Returns:
        sampled_ids (torch.Tensor): 1, n_tokens
        masking (torch.Tensor): 1, n_tokens
    """
    
    mask_schedule = get_mask_chedule("cosine")
    temperature = temperature * (1.0 - ratio) # Adds noise for randomness
    mask_ratio = mask_schedule(torch.tensor(ratio))
    
    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max) # Ignores the tokens given in the input by overwriting their confidence.
    mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(selected_probs.device) # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = torch.max(
        torch.tensor([1], device=selected_probs.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    )
    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=None) # Masks tokens with lower confidence.
    
    return masking