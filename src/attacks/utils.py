import torch
from typing import List, Tuple, Union, Dict
import numpy as np


def token_gradients(
        model,  # Targeted HuggingFace model
        trigger_slice: slice,  # targeted slice
        inputs: Dict[str, torch.Tensor],  # with input_ids, (batch_size, seq_len)
        input_embedding_layer: torch.nn.Embedding,  # embedding layer
        device: str = 'cuda',

        # Fluency constraints
        flu_model=None,
        flu_alpha=0.0,
        l2_alpha=0.0,
        **kwargs,  # additional kwargs to pass to the model for calculating the loss (e.g., labels)
):
    """
    :returns: the gradients wrp to the input (one-hot embedded) tokens in the `trigger_slice` (i.e. the targeted
             sub-sequence).
             The gradient is calculated on the original loss (e.g., Cross Entropy), and towards decreasing it.
    [Inspired by: https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py#L12C8-L12C8]
    """
    input_ids = inputs['input_ids']

    embed_weights = input_embedding_layer.weight
    one_hot = torch.zeros(
        input_ids.shape[0],  # batch_size
        input_ids[:, trigger_slice].shape[1],  # targeted_sub_seq_len
        embed_weights.shape[0],  # vocab_size
        device=device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        -1,
        input_ids[:, trigger_slice].unsqueeze(-1),
        torch.ones(one_hot.shape[0], one_hot.shape[1], 1, device=device, dtype=embed_weights.dtype)
    )

    one_hot.requires_grad_()
    trigger_embeds = (one_hot @ embed_weights)  # (batch_size, targeted_sub_seq_len, embed_dim)

    # now stitch it together with the rest of the embeddings
    embeds = input_embedding_layer(input_ids).detach()  # (batch_size, seq_len, embed_dim)
    full_embeds = torch.cat(
        [
            embeds[:, :trigger_slice.start, :],
            trigger_embeds,
            embeds[:, trigger_slice.stop:, :]
        ],
        dim=1)

    loss = model.calc_loss_for_grad(inputs_embeds=full_embeds, inputs_attention_mask=inputs['attention_mask'], **kwargs)
    if flu_alpha != 0 and flu_model is not None:
        # Calculate the fluency score
        fluency_score = flu_model.calc_score_for_grad(
            one_hot=one_hot,
            trigger_slice=trigger_slice,
            inputs=inputs,
        )
        # Add the fluency score to the loss
        loss += flu_alpha * fluency_score
    if l2_alpha != 0:
        # Add the L2 norm of the trigger to the loss (we want to minimize the term)
        loss += -l2_alpha * trigger_embeds.norm(dim=-1).sum()

    loss.backward()

    onehot_grads = one_hot.grad.clone().detach()
    onehot_grads /= onehot_grads.norm(dim=-1, keepdim=True)  # TODO helps?

    return onehot_grads


def get_trigger_candidates(
        scores: torch.Tensor,  # scores of all possible tokens, highest-better (targeted_sub_seq_len, vocab_size)
        trigger_slice: slice,  # targeted slice
        k_candidates: int = 5,  # number of candidates to return
        candidate_scheme: str = 'best_per_token',  # 'best_per_token' | 'best_overall'
        token_ids_to_ignore: List[int] = None,  # usually the special tokens (which we don't want to consider)

        flu_scores: torch.Tensor = None,  # fluency scores of all possible tokens, highest-better (seq_len, vocab_size)
        filter_to_n_most_readable: int = 700,
) -> Union[Dict[int, Dict[str, List[Union[int, float]]]], List[Tuple[int, int]]]:
    """
    Translates the scores to the actual tokens to flip to.
    :returns: the `num_candidates` best tokens to flip to in each position in the `trigger_slice`; the list is sorted
              from the best token position to flip, to the rest.
              The returns list has each element of (token_idx_to_flip, token_id_to_flip_to).
    Note! this function is indented for a _single_ sample.
    """
    candidates = {}  # {token_idx -> (token_top_ids, token_top_scores), ...}
    scores = scores.clone()
    scores[:, token_ids_to_ignore] = -np.inf  # ignore the special tokens, set their scores to the lowest possible
    if candidate_scheme == 'top_k_per_token':
        for token_idx_in_trigger, token_idx in enumerate(range(trigger_slice.start, trigger_slice.stop)):
            if flu_scores is not None and filter_to_n_most_readable is not None:
                # ignore tokens with low readability
                indices_with_low_readability = np.argsort(flu_scores[token_idx_in_trigger].cpu())[:-filter_to_n_most_readable]
                scores[token_idx_in_trigger, indices_with_low_readability] = -np.inf
            token_top_k_obj = scores[token_idx_in_trigger].topk(k_candidates)
            token_top_ids = token_top_k_obj.indices.tolist()
            token_top_scores = token_top_k_obj.values.tolist()
            candidates[token_idx] = dict(token_top_ids=token_top_ids, token_top_scores=token_top_scores)
    elif candidate_scheme == 'top_k_overall':  # Disabled
        v, i = torch.topk(scores.flatten(), k_candidates)
        v, i = v.cpu(), i.cpu()
        candidates = np.array(np.unravel_index(i.numpy(), scores.shape)).T
        candidates[:, 0] += trigger_slice.start  # shift indices to the correct position
        candidates = [(token_idx, token_id) for token_idx, token_id in candidates]
    else:
        raise ValueError(f"Unknown candidate scheme: {candidate_scheme}")
    return candidates
