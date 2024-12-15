import math
import torch
import numpy as np
import torch
from torch.nn import functional as F

def dola_greedy_decode(
    mature_logits,
    premature_logits,
    layer_logit_dist = None,
    relative_top: float = 0.1,
    
) :
    n_layers = premature_logits.shape[0]
    candidate_premature_layers = list(range(n_layers))
    if layer_logit_dist is None:
        layer_logit_dist = np.zeros(n_layers)
        
    softmax_mature_layer = F.softmax(mature_logits, dim=-1)  # shape: (batch_size, num_features)
    softmax_premature_layers = F.softmax(premature_logits, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)
    M = [softmax_mature_layer + softmax_premature_layers[idx] for idx in range(n_layers)]
    M = 0.5 * torch.stack(M)  # shape: (num_premature_layers, batch_size, num_features)

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(mature_logits, dim=-1)  # shape: (batch_size, num_features)
    log_softmax_premature_layers = F.log_softmax(premature_logits, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    # 6. Reduce the batchmean
    maximum_jsd_index = int(js_divs.mean(-1).argmax().cpu().item()) # shape: (num_premature_layers,)
    premature_layer = candidate_premature_layers[maximum_jsd_index]
    
    base_logits = premature_logits[premature_layer]
    final_logits = mature_logits
    if relative_top > 0.0:
        def _relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
            scores_normalized = scores.log_softmax(dim=-1) 
            sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
            min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
            probs_max = torch.max(scores_normalized, dim=-1).values
            probs_thresh = probs_max + np.log(relative_top)
            probs_thresh = torch.min(min_thresh, probs_thresh)
            probs_thresh = probs_thresh.unsqueeze(-1)
            scores_normalized[scores_normalized < probs_thresh] = filter_value
            return scores_normalized
        
        final_logits = _relative_top_filter(mature_logits, relative_top=relative_top)
        base_logits = base_logits.log_softmax(dim=-1)
        mask = final_logits < -1e3
        base_logits[mask] = -1e3
    
    logits = final_logits - base_logits
    
    return logits, js_divs

def pag_decode(
    logit_cond,
    logit_uncond,
    logit_ptb_cond = None,
    gamma = None,
    omega = 0.0,
    cd_alpha = 0.0,
    cd_beta = 0.0,
    ):
    enable_cd = cd_beta > 0.0
    enable_cfg = gamma > 1.0
    enable_pag = omega > 0.0
    if enable_cd:
        logits_E = gamma * logit_cond + (1 - gamma) * logit_uncond
        cutoff = math.log(cd_alpha) + logits_E.max(dim=-1, keepdim=True).values
        logits_A = logit_ptb_cond
        diffs = (1 + cd_beta) * logits_E - cd_beta * logits_A
        logits = diffs.masked_fill(logits_E < cutoff, -float('inf'))
    elif enable_cfg and enable_pag:
        logits = (gamma + omega) * logit_cond + (1 - gamma) * logit_uncond + (1 - omega) * logit_ptb_cond
    elif enable_cfg:
        logits = (1 - gamma) * logit_uncond + gamma * logit_cond
    elif enable_pag:
        logits = (1 - omega) * logit_ptb_cond + omega * logit_cond # cfg = (1 - gamma) / gamma
    
    return logits


def gt_decode(
    input_ids_cond,
    mask_token_id,
    probs,
    input_ids_gt,
    llm_vocab_size,
    num_new_special_tokens,
    image_start
):
    pred_mask = input_ids_cond[:, image_start : ] != mask_token_id # includes soi, eoi token
    predicted_logits = probs[pred_mask[:, 1:-1]] # image vocabs are only 8192 | text vocabs should be subtracted (llm_vocab_size + num_new_special_tokens)
    true_ids = input_ids_gt[pred_mask][1:-1] - (llm_vocab_size + num_new_special_tokens)
    pred_ids = predicted_logits.argmax(dim=-1)
    true_logits_predicted = torch.gather(predicted_logits, 1, true_ids.unsqueeze(1)).squeeze(1) # teacher forcing logits
    actual_logits_predicted = torch.gather(predicted_logits, 1, pred_ids.unsqueeze(1)).squeeze(1) # actual logits from text context
    diff = true_logits_predicted - actual_logits_predicted
    input_ids_cond[:, image_start : ][pred_mask] = input_ids_gt[pred_mask] # replace predictions with ground truth
    return input_ids_cond, diff


def adaptive_decode(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, ada: float = 0.01, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    assert ada > 0.0005 and ada < 0.01, f"adaptive threshold should be between 0.0005 and 0.01, got {ada}"
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    prob = sorted_logits.softmax(dim=-1)
    cumulative_probs = prob.cumsum(dim=-1)

    vocab_size = cumulative_probs.shape[1]
    up_bound = -np.log(1.0 / vocab_size)
    position = torch.arange(1, vocab_size + 1).repeat(cumulative_probs.shape[0], 1).to(cumulative_probs.device)

    A = prob * torch.log(prob * (vocab_size - position) / (1.0 - cumulative_probs))
    B = (1 - cumulative_probs) / (vocab_size - position)
    C = (1 - cumulative_probs + prob) / (vocab_size + 1 - position)
    delta_conf = (A + (1 - cumulative_probs + prob) * torch.log(B / C)) / up_bound
    delta_conf[torch.isnan(delta_conf)] = 0

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = delta_conf <= ada

    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed
