import math
import random
import time
from typing import Tuple, List, Any

import wandb
from tqdm import trange, tqdm
import numpy as np
import torch
import logging

from src.attacks.fluency_scorer import initialize_fluency_model
from src.attacks.rephraser import T5Rephraser
from src.attacks.utils import token_gradients, get_trigger_candidates

logger = logging.getLogger(__name__)


def gaslite_attack(
        model,  # HuggingFace model
        inputs: dict,  # with `input_ids` key, (n_samples, seq_len)
        trigger_slice: slice,  # targeted slice
        n_iter: int = 10,  # number of iterations
        n_flips_to_gen: int = 3,  # of candidates to be chosen for loss calc; used only in older attacks. referred to as 'batch size' in GCG paper
        evaluate_flips_on_strs: bool = True,  # whether to evaluate & filter the flips on the string-level

        attack_variant: str = None,  # 'autoprompt' | 'gcg' | 'arca' | None (ours)    [to support older attacks]
        n_grad: int = 1,  # amount of flips to perform random gradient averaging
        k_candidates: int = 200,  # number of candidates to consider
        backprop_batch_size: int = 90,  # maximal samples to backprop in parallel
        time_limit_in_seconds: int = None,  # time limit for the attack

        # Optimization components
        beam_search_config: dict = None,  # whether to perform beam-search-like search

        # Constraints [Fluency]:
        flu_alpha: float = 0,  # weight of the fluency score (0 means no fluency score)
        flu_alpha_inc: dict = {},  # increase the fluency weight over time
        fluency_model_name: str = 'gpt2',  # model used to evaluate fluency of the trigger, {'bert', 'gpt2'}

        # Constraints [L2]:
        l2_alpha: float = 0,  # loss weight of the L2 norm of the trigger (0 means no L2 norm term)

        # Logging:
        log_to: str = None,  # where to log the results per step; {None, 'wandb', 'liveplotloss'}

        # [UNUSED]
        use_rephraser: bool = False, # [UNUSED] whether to rephrase the trigger after the optimization is (mostly) exhausted
        n_sim_tokens_flips_to_gen: float = 100,  # number of candidates to be chosen by the similarity of the tokens [# POSITIVE -> # of sim-based candidates to add, NEGATIVE -> # of sim-based candidates per-token to consider when filtering the EXISTING grad-based candidates]
        gradual_batch_optimization: bool = False,
        perform_arca_opt: bool = False,  # whether to perform ARCA optimization (instead of the default GCG)

        **kwargs,  # additional kwargs to pass to the model for calculating the loss (e.g., labels)
):
    logger.info(f"params: {locals()}")

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids = input_ids.clone().detach().to('cuda')
    n_samples, seq_len = input_ids.shape
    vocab_size = model.get_input_embeddings().num_embeddings
    best_loss, best_flu = -np.inf, -np.inf
    trigger_len = trigger_slice.stop - trigger_slice.start
    switch_to_gcg_in_step = beam_search_config.get('switch_to_gcg_in_step', None) or math.inf

    if log_to == 'livelossplot':
        from livelossplot import PlotLosses
        plotlosses = PlotLosses()

    if k_candidates is None and beam_search_config['perform']:
        k_candidates = beam_search_config['n_cand'] * 2  # default value for k_candidates

    token_ids_to_ignore = model.get_special_token_ids()
    logger.info(f"Attack ignores {len(token_ids_to_ignore) / vocab_size :.3f}% of the tokens.")

    # >> Initialize the fluency scorer
    flu_model = initialize_fluency_model(fluency_model_name, batch_size=128)  # TODO generalize
    if (
            (flu_alpha != 0 or flu_alpha_inc.get('filter_to_n_most_readable', None))  # if you use fluency
            and model.tokenizer.__class__ != flu_model.tokenizer.__class__  # and we have tokenizer mismatch
    ):
        raise ValueError("Fluency model's tokenizer and the model's tokenizer must match.")

    if use_rephraser:  # [CURRENTLY UNUSED]
        rephraser = T5Rephraser()

    start_time = time.time()  # record the start time

    for i in trange(n_iter, desc="Attacking with GASLITE..."):
        if trigger_len == 0:
            logger.info("Trigger is empty; stopping the attack.")  # we allow this for experiments
            best_input_ids = input_ids
            best_loss = torch.tensor(-1)
            break

        if n_grad > 1:
            # sample `grad_batch_size` random positions to flip
            if attack_variant == 'arca':
                # 'Fix' the token position we are currently optimizing
                # [Option1:_fixed_pos] Deterministic fixed position (like in ARCA's paper)
                _fixed_pos = i % (trigger_slice.stop - trigger_slice.start) + trigger_slice.start  # ARCA's `_fixed_pos`
                # [Option2:_fixed_pos] Random fixed position
                # _fixed_pos = random.choice(list(fixed_positions))
                fixed_positions.discard(_fixed_pos)
                if len(fixed_positions) == 0:
                    fixed_positions = set(range(trigger_slice.start, trigger_slice.stop))
                # place a random token in `_fixed_pos` for each batch
                input_ids_repeated = input_ids.repeat(n_grad, 1)
                input_ids_repeated[:, _fixed_pos] = (
                    torch.randint(0, vocab_size, size=(n_grad,),
                                  device='cuda').repeat_interleave(n_samples)
                )
            else:  # GASLITE's grad-averaging applies on multiple token-positions.
                # [Option2] Multiple random positions followed by GCG's candidate-mechanism
                random_tok_positions = torch.randint(trigger_slice.start, trigger_slice.stop,
                                                 size=(n_grad-1,), device='cuda').repeat_interleave(n_samples)
                # place a random token in a random-position for each batch
                input_ids_repeated = input_ids.repeat(n_grad, 1)
                input_ids_repeated[torch.arange(len(input_ids), len(input_ids_repeated)), random_tok_positions] = (
                    torch.randint(0, vocab_size, size=(n_grad-1,),
                                  device='cuda').repeat_interleave(n_samples)
                )
            grad_inputs = {
                'input_ids': input_ids_repeated,
                'attention_mask': attention_mask.repeat(n_grad, 1),
            }
            grad_kwargs = _hack_to_enlarge_kwargs(kwargs, n_grad)
        else:
            grad_inputs = inputs
            grad_kwargs = kwargs

        # 1. Get the gradients wrp to the input
        onehot_grads = _token_gradients_batch(
            model=model,
            input_embedding_layer=model.get_input_embeddings(),
            trigger_slice=trigger_slice,
            inputs=grad_inputs,
            backprop_batch_size=backprop_batch_size,  # maximal samples to backprop in parallel
            flu_model=flu_model,
            flu_alpha=flu_alpha,
            l2_alpha=l2_alpha,
            **grad_kwargs,
        )

        if n_grad > 1:
            # average on the different gradient batches
            onehot_grads = onehot_grads.view(n_grad, onehot_grads.shape[0]//n_grad, *onehot_grads.shape[1:]).mean(dim=0)

        # 2. Scores are the average of the gradients
        scores = onehot_grads.mean(dim=0)  # (n_samples, targeted_sub_seq_len, vocab_size)

        # 3. Get the candidates for token flips
        filter_to_n_most_readable = flu_alpha_inc.get('filter_to_n_most_readable', None)
        candidates = get_trigger_candidates(
            scores=scores,
            flu_scores=flu_model.calc_fluency_scores(
                inputs={
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                },
                return_logits=True,
            )[0, trigger_slice] if filter_to_n_most_readable is not None else None,
            trigger_slice=trigger_slice,
            k_candidates=k_candidates,
            candidate_scheme='top_k_per_token',
            token_ids_to_ignore=token_ids_to_ignore,
            filter_to_n_most_readable=filter_to_n_most_readable
        )

        # 4. Sample `n_flips_to_gen` possible adv samples, and examine the loss for each

        # 4.1. Select the token position(s) to flip
        flipped_input_ids_lst = []
        flipped_idx_and_id_lst = []  # each element is a tuple of (token_idx, token_id)

        # [AUTOPROMPT] Fix a random token position to evaluate, then evaluate all top-k_candidates for this position
        if attack_variant == 'autoprompt':
            _fixed_pos = np.random.randint(trigger_slice.start, trigger_slice.stop)
            token_indices_to_flip = np.full(k_candidates, fill_value=_fixed_pos)
            token_top_ids_indices = np.arange(k_candidates)

        elif attack_variant == 'arca':
            # takes the fixed position of this iteration (as calculated before taking the gradient)
            token_indices_to_flip = np.full(k_candidates, fill_value=_fixed_pos)
            token_top_ids_indices = np.arange(k_candidates)

        elif attack_variant == 'gcg':
            # [GCG] GCG's choice of positions; sample random positions to flip, and a random candidate for each.
            token_indices_to_flip = np.random.randint(trigger_slice.start, trigger_slice.stop, size=n_flips_to_gen).astype(int)
            token_top_ids_indices = np.random.randint(k_candidates, size=n_flips_to_gen)

        else:  # defaults to our method
            # We select each flip-position multiple times (defined by `n_flips_to_gen`),
            # to spread the flips across the trigger, each token_id is sampled from the top-k_candidates
            token_indices_to_flip = np.arange(trigger_slice.start, trigger_slice.stop,
                                              step=(trigger_slice.stop - trigger_slice.start) / n_flips_to_gen).astype(int)
            token_top_ids_indices = np.random.randint(k_candidates, size=n_flips_to_gen)

        # 4.2. Sample a list of random flips, from the `candidates`
        for token_idx, token_top_ids_idx in zip(token_indices_to_flip, token_top_ids_indices):
            # Get the top-k_candidates token-ids for this token position
            token_top_ids = candidates[token_idx]['token_top_ids']
            # Get the token-id to flip to
            token_id = token_top_ids[token_top_ids_idx]
            # Save the flip
            flipped_idx_and_id_lst.append((token_idx, token_id))
            flipped_input_ids = input_ids.clone().detach()
            flipped_input_ids[:, token_idx] = token_id
            flipped_input_ids_lst.append(flipped_input_ids)

        # 4.3. [Optionally] Decode and re-tokenize, after discard flips that are not properly decoded.
        #      Note: this meant to provide a more realistic attack, which focus on the string and not the token-list.
        if evaluate_flips_on_strs:
            flipped_input_ids_lst, flipped_idx_and_id_lst = _decode_and_retokenize(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                flipped_input_ids_lst=flipped_input_ids_lst,
                flipped_idx_and_id_lst=flipped_idx_and_id_lst,
            )

        # 5.1. Calculate the loss for each generated sample
        n_flipped = len(flipped_input_ids_lst)
        flipped_input_ids_lst = torch.stack(flipped_input_ids_lst, dim=0).cuda()  # (n_flipped, n_samples, seq_len)
        calc_losses_kwargs = dict(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            trigger_slice=trigger_slice,
            flipped_input_ids_lst=flipped_input_ids_lst,  # can be updated for different flips
            flipped_idx_and_id_lst=flipped_idx_and_id_lst,  # can be updated for different flips  # TODO how to generalize to beam?
            flu_model=flu_model,
            flu_alpha=flu_alpha,
            l2_alpha=l2_alpha,
            **kwargs,
        )

        if beam_search_config['perform'] and i < switch_to_gcg_in_step:
            # define parameters for beam-search
            bs_params = dict(
                n_flip=beam_search_config['n_flip'],  # number of positions to flip
                B=beam_search_config.get('B', 1),  # size ("width") of preserved set after its iter ("beam")
                n_cand=beam_search_config['n_cand'],  # number of candidates to evaluate, per position
                n_cand__sample_rate=beam_search_config.get('n_cand__sample_rate', 0.5),  # sample rate for the top-n_cand candidates
                beam_indices=beam_search_config.get('beam_indices', None),  # indices to consider for beam-search
            )
            # Deduce the different n_cand's
            bs_params['n_cand__from_sample'] = int(bs_params['n_cand'] * bs_params['n_cand__sample_rate'])
            bs_params['n_cand__from_top'] = int(bs_params['n_cand'] * (1 - bs_params['n_cand__sample_rate']))
            bs_params['n_flip'] = bs_params['n_flip'] or 20  # default value for n_flip
            if bs_params['n_flip'] > trigger_len:
                bs_params['n_flip'] = int(trigger_len * 0.67)  # make sure n_flips is not bigger than the trigger length

            # Choose the indices to perform beam-search on; there are some option for this.
            # Intuition: this might be better for optimizing the trigger's fluency.
            if bs_params['beam_indices'] == 'random_interval':  # for choosing only contiguous subsets
                beam_interval_start = np.random.choice(np.arange(trigger_slice.start, trigger_slice.stop - bs_params['n_flip']))
                trigger_indices = np.arange(beam_interval_start, beam_interval_start + bs_params['n_flip'])
            elif bs_params['beam_indices'] == 'exhaust_interval':  # for deterministic contiguous subsets.
                beam_interval_start = (i % (trigger_slice.stop - trigger_slice.start - bs_params['n_flip'])) + trigger_slice.start
                trigger_indices = np.arange(beam_interval_start, beam_interval_start + bs_params['n_flip'])
            else:  # default; 'random': for sampling any subset indices
                trigger_indices = sorted(np.random.choice(np.arange(trigger_slice.start, trigger_slice.stop),
                                                          size=bs_params['n_flip'], replace=False))

            beam_pool = input_ids.clone().detach().unsqueeze(0)  # [n_beams, n_samples, seq_len)
            for curr_token_idx in tqdm(trigger_indices, desc="Performing Beam-Search..."):

                # List the possible flips to apply on the beam-pool
                # Option 1: choose the top-n_cand candidates
                token_top_ids_1 = torch.tensor(candidates[curr_token_idx]['token_top_ids']
                                               [:bs_params['n_cand__from_top']]).cuda()
                # Option 2: randomly choose from the top-n_cand candidates
                token_top_ids_2 = torch.tensor(random.choices(candidates[curr_token_idx]['token_top_ids']
                                                              [bs_params['n_cand__from_top']:],
                                                              k=bs_params['n_cand__from_sample']),
                                               dtype=torch.int64).cuda()
                # Combine the two options
                token_top_ids = torch.cat([token_top_ids_1, token_top_ids_2], dim=0)
                # new_flipped_idx_and_id_lst = [(curr_token_idx, token_id) for token_id in token_top_ids]
                # Apply flips on the beam pool
                beam_pool_next = []
                for beam_node in beam_pool:  # for each beam -> expand via possible flips
                    beam_pool_next.append(beam_node)  # We always keep the original beam-node (i.e., no flip)
                    new_beam_nodes = beam_node.clone().detach().repeat(len(token_top_ids), 1, 1)
                    new_beam_nodes[torch.arange(len(token_top_ids)).cuda(), :, curr_token_idx] = (
                        token_top_ids.unsqueeze(-1))
                    beam_pool_next.extend(new_beam_nodes)
                beam_pool_next = torch.stack(beam_pool_next, dim=0).to('cuda')
                # [OPTIONALLY] Retokenize all inputs:
                if evaluate_flips_on_strs:
                    beam_pool_next, _ = _decode_and_retokenize(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        flipped_input_ids_lst=beam_pool_next,
                        flipped_idx_and_id_lst=[None] * len(beam_pool_next),
                    )
                    beam_pool_next = torch.stack(beam_pool_next, dim=0).cuda()
                # Calc losses:
                _beam_calc_losses_kwargs = calc_losses_kwargs.copy()
                _beam_calc_losses_kwargs.update(
                    flipped_input_ids_lst=beam_pool_next,
                    flipped_idx_and_id_lst=None,
                    flu_model=flu_model,
                    # flipped_idx_for_flu_score_calc=curr_token_idx,  # [DISABLED] Currently we evaluate on the whole passage
                )
                losses, _ = _calc_losses_of_flip(**_beam_calc_losses_kwargs)
                # Take the top-B next beam-nodes and update the pool
                beam_pool = beam_pool_next[losses.argsort(descending=True)[:bs_params['B']]]
                beam_pool_losses = losses[losses.argsort(descending=True)[:bs_params['B']]]

            # [Hack] to work with the final flip picker
            losses = beam_pool_losses
            loss_without_flip = -np.inf
            flipped_input_ids_lst = beam_pool
            flipped_idx_and_id_lst = [None] * len(beam_pool)
        else:  # performing regular (single) flip; relevant for older attacks (`attack_variant`)
            # Calculate the losses of the flips
            losses, loss_without_flip = _calc_losses_of_flip(**calc_losses_kwargs)

        # 6. [GCG] Get the index with the biggest loss
        # TODO? use simulated-annealing; if the loss is not improved, we can still accept it with a certain probability, that decreases over time.?
        flip_only_if_improve = (attack_variant == 'autoprompt')
        if loss_without_flip > losses.max() and flip_only_if_improve:
            # we update only if the post-flip loss is bigger than the previous loss [like AUTOPROMPT]
            loss = loss_without_flip
        else:
            # TODO sample from top-{?} instead of argmax?
            idx_flip_with_max_loss = losses.argmax().item()
            loss = losses[idx_flip_with_max_loss]
            flipped_input_ids = flipped_input_ids_lst[idx_flip_with_max_loss]

        # 6. Execute the chosen best flip
        input_ids = flipped_input_ids

        if use_rephraser and i in [(n_iter // 3) * 2]:  # [UNUSED]
            # refresh the phrasing, to avoid getting stuck in a local-minima
            # TODO currently it support TRIGGER-ONLY! needs to be expanded.
            curr_trigger_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[1]

            # Rephrase the trigger
            new_trigger_strs = rephraser.rephrase(curr_trigger_str)
            # choose the best rephrasing (most sim to the 'curr' trigger)
            new_cos_sims = torch.nn.functional.cosine_similarity(model.embed([curr_trigger_str]),
                                                                 model.embed(new_trigger_strs))
            new_trigger_str = new_trigger_strs[new_cos_sims.argmax()]
            logger.info(f"Rephrased the trigger (sim={new_cos_sims.max().item() :.4f}): "
                        f"{curr_trigger_str} -> {new_trigger_str}")
            new_trigger_str = new_trigger_str * 2  # makes sure the trigger is long enough

            # Tokenize, trim and update:
            new_input_ids = model.tokenizer(new_trigger_str, return_tensors="pt", padding='max_length',
                                            truncation=True).input_ids.cuda()
            new_input_ids[:, trigger_slice.stop] = model.tokenizer.sep_token_id  # add the sep token
            new_input_ids[:, (trigger_slice.stop + 1):] = model.tokenizer.pad_token_id
            input_ids = new_input_ids

        text_inputs = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        other_metrics = _calc_other_metrics(model, input_ids, attention_mask, text_inputs,
                                            trigger_slice, flu_model, kwargs['emb_targets'])
        logger.info(f"<iter-{i}> loss: {loss}, {', '.join([f'{k}: {v}' for k, v in other_metrics.items()])}")
        if log_to == 'wandb':
            wandb.log({"attack_loss": loss, **other_metrics,
                       "adv_text": text_inputs[0],  # logs the adversarial text; disable when not needed.
                       })
        if log_to == 'livelossplot':
            # print(loss)
            plotlosses.update({"loss": loss.item()})
            plotlosses.send()
        logger.info("adv_text: " + text_inputs[0])  # logs the adversarial text; disable when not needed.

        if ((flu_alpha == 0 and l2_alpha == 0 and loss > best_loss)  # no fluency
            or (flu_alpha != 0 and i >= n_iter // 2 and loss > best_loss)  # for flu we start checking after half of the iterations
            or (l2_alpha != 0 and i >= n_iter // 2 and loss > best_loss)  # for l2 we start checking after half of the iterations
        ):  # increased the loss -> update best_input ids
            best_loss = loss
            best_input_ids = input_ids
            logger.info("Loss is improved; best instance was updated.")
        if flu_alpha != 0 and i >= n_iter // 2 and other_metrics['fluency_score'] > best_flu:
            best_flu = other_metrics['fluency_score']
            best_flu_input_ids = input_ids
            logger.info("Fluency score is improved; best-fluency instance was updated.")
        if i != 0 and i == flu_alpha_inc.get('freq', 1):
            flu_alpha *= flu_alpha_inc.get('factor', 1.0)
            logger.info(f"Fluency weight (`flu_alpha`) was updated to {flu_alpha}")

        if time_limit_in_seconds is not None and time_limit_in_seconds < (time.time() - start_time):
            logger.info(f"Time limit ({time_limit_in_seconds} sec) reached; stopping the attack.")
            break

    # Calculate the final metrics of the best instances
    final_metrics = {}

    if flu_alpha != 0:  # Evaluate the best-flu instance and log the metrics
        text_inputs = model.tokenizer.batch_decode(best_flu_input_ids, skip_special_tokens=True)
        other_metrics = _calc_other_metrics(model, best_flu_input_ids, attention_mask, text_inputs,
                                            trigger_slice, flu_model, kwargs['emb_targets'])
        logger.info(">>>> Evaluating the best-flu instance...")
        logger.info({"adv_text_after_attack_best_flu": text_inputs[0], **other_metrics})
        final_metrics.update({"best_flu_instance_text": text_inputs[0],
                             "best_flu_instance": other_metrics})

    text_inputs = model.tokenizer.batch_decode(best_input_ids, skip_special_tokens=True)
    final_metrics.update(_calc_other_metrics(model, best_input_ids, attention_mask, text_inputs,
                                             trigger_slice, flu_model, kwargs['emb_targets']))
    final_metrics.update({'loss': best_loss.item()})
    return best_input_ids, final_metrics


def _calc_other_metrics(model, input_ids, attention_mask, text_inputs, trigger_slice, flu_model, emb_targets):
    emb = model.embed(inputs={'input_ids': input_ids, 'attention_mask': attention_mask})
    fluency_score = flu_model.calc_fluency_scores(
        inputs=None,
        text_inputs=text_inputs,
    ).mean().item()
    return dict(
        adv_l2_norm=emb.norm(p=2, dim=-1).mean().item(),
        cos_sim_adv_target=torch.nn.functional.cosine_similarity(emb, emb_targets).mean().item(),
        cos_sim_adv_target_on_text=torch.nn.functional.cosine_similarity(model.embed(texts=text_inputs), emb_targets).mean().item(),
        fluency_score=fluency_score
    )


def _hack_to_enlarge_kwargs(kwargs, enlarge_factor):
    # Hack to repeat labels:  # TODO fix this hack
    new_kwargs = {}
    if 'labels' in kwargs:
        new_kwargs['labels'] = kwargs['labels'].repeat(enlarge_factor)
    if 'emb_targets' in kwargs:
        new_kwargs['emb_targets'] = kwargs['emb_targets'].repeat(enlarge_factor, 1)
    if 'emb_anchors' in kwargs:
        new_kwargs['emb_anchors'] = kwargs['emb_anchors'].repeat(enlarge_factor, 1)
    return new_kwargs


# TODO make the batching a decorator
def _token_gradients_batch(
        model,  # HuggingFace model
        input_embedding_layer,  # the embedding layer of the model
        trigger_slice: slice,  # targeted slice
        inputs: dict,  # with `input_ids`, (n_samples, seq_len)
        backprop_batch_size: int = 150,  # maximal samples to backprop in parallel
        flu_model=None,
        flu_alpha: float = 0,
        l2_alpha: float = 0,
        **kwargs,  # additional kwargs to pass to the model for calculating the loss (e.g., labels)
) -> torch.Tensor:
    """Performs 'token_gradients' for multiple batches (to avoid OOM). """
    n_samples = inputs['input_ids'].shape[0]
    n_batches = (n_samples + backprop_batch_size - 1) // backprop_batch_size
    grads = []

    for i in trange(n_batches, desc="Calculating token gradients..."):
        batch_slice = slice(i * backprop_batch_size, (i + 1) * backprop_batch_size)
        batch_inputs = {k: v[batch_slice] for k, v in inputs.items()}
        batch_kwargs = {k: v[batch_slice] for k, v in kwargs.items()}
        batch_grads = token_gradients(
            model=model,
            input_embedding_layer=input_embedding_layer,
            trigger_slice=trigger_slice,
            inputs=batch_inputs,
            flu_model=flu_model,
            flu_alpha=flu_alpha,
            l2_alpha=l2_alpha,
            **batch_kwargs,
        )
        grads.append(batch_grads)

    grads = torch.cat(grads, dim=0)  # TODO this is a potential bottleneck, need to somehow aggregate ahead
    return grads


def _decode_and_retokenize(
        model,  # HuggingFace model
        input_ids: torch.Tensor,  # original input_ids; (n_samples, seq_len)
        attention_mask: torch.Tensor,  # original attention_mask; (n_samples, seq_len)
        flipped_input_ids_lst: List[torch.Tensor],  # flipped input_ids; `n_flipped` each (n_samples, seq_len)
        flipped_idx_and_id_lst: list,  # list of tuples, each is (token_idx, token_id)
        **kwargs
) -> Tuple[List[torch.Tensor], List[Any]]:  # (flipped_input_ids_lst, flipped_idx_and_id_lst)
    """Decodes the flipped input_ids, and filters out the ones that are not properly decoded."""

    pad_token_id = model.tokenizer.pad_token_id
    old_decoded_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    # allowed_diff_of_decoding = 0  # `0` means the generated text must be reversible
    new_flipped_input_ids_lst, new_flipped_idx_and_id_lst = [], []
    for flipped_input_ids, flipped_idx_and_id in zip(flipped_input_ids_lst, flipped_idx_and_id_lst):
        new_decoded_str = model.tokenizer.batch_decode(flipped_input_ids, skip_special_tokens=True)
        new_flipped_input_ids = model.tokenizer(new_decoded_str, return_tensors="pt", padding='max_length',
                                                truncation=True).input_ids

        # Verify that the string was changed
        text_was_changed = old_decoded_str != new_decoded_str  # also in GCG
        # Verify the amount of token was kept
        len_tokens_kept = all(
            [(new_ids != pad_token_id).sum().item() == (old_ids != pad_token_id).sum().item() for new_ids, old_ids in
             zip(new_flipped_input_ids, flipped_input_ids)])  # also in GCG
        # is_reversible = np.sum(model.tokenizer.encode(new_decoded_str, add_special_tokens=False).input_ids == flipped_input_ids) < allowed_diff_of_decoding
        if text_was_changed and len_tokens_kept:
            # TODO consider replacing the old input_ids with the new ones, ONLY once in few iterations (to let the attack adapt to the new trigger)
            # if i % 5 != 0:  # replace the input_ids only once in few iterations
            #     new_flipped_input_ids = flipped_input_ids  # now we keep the non-str input_ids
            new_flipped_input_ids_lst.append(new_flipped_input_ids)
            new_flipped_idx_and_id_lst.append(flipped_idx_and_id)
            if not torch.all(model.tokenizer(new_decoded_str, return_tensors="pt", padding='max_length',
                                             truncation=True).attention_mask.cpu() == attention_mask.cpu()):
                logger.info(f"[WARNING] attention seem to not align with the token list; ")
                logger.info(flipped_input_ids[0])
                logger.info(attention_mask[0])
                logger.info("---------------")
                logger.info(new_decoded_str)
                logger.info(new_flipped_input_ids[0])
                logger.info(model.tokenizer(new_decoded_str, return_tensors='pt', padding='max_length',
                                            truncation=True).attention_mask[0])
        elif model.tokenizer.__class__.__name__.endswith('T5TokenizerFast') and all(
                [(new_ids != pad_token_id).sum().item() == (old_ids != pad_token_id).sum().item() - 1 for
                 new_ids, old_ids in
                 zip(new_flipped_input_ids, flipped_input_ids)]):
            # HACK to address T5 tokenizer specifically
            new_flipped_input_ids_lst.append(flipped_input_ids.to(new_flipped_input_ids.device))
            new_flipped_idx_and_id_lst.append(flipped_idx_and_id)

    logger.info(f"Ratio of flips kept: {len(new_flipped_input_ids_lst) / len(flipped_input_ids_lst)}")

    return new_flipped_input_ids_lst, new_flipped_idx_and_id_lst


def _calc_losses_of_flip(
        model,
        input_ids: torch.Tensor,  # original input_ids; (n_samples, seq_len)
        attention_mask: torch.Tensor,  # original attention_mask; (n_samples, seq_len)
        trigger_slice: slice,  # targeted slice

        # Flips to eval
        flipped_input_ids_lst: torch.Tensor,  # flipped input_ids; (n_flipped, n_samples, seq_len)
        flipped_idx_and_id_lst: list,  # list of tuples, each is (token_idx, token_id)

        # Fluency params
        flu_model=None,  # fluency model
        flu_alpha: float = 0,  # weight of the fluency score (0 means no fluency score)
        l2_alpha: float = 0,  # loss weight of the L2 norm of the trigger (0 means no L2 norm term)
        flipped_idx_for_flu_score_calc: int = None,  # the token index to calculate the fluency score for

        **kwargs
) -> Tuple[torch.Tensor, float]:
    """Calculates the loss for each flip, and the loss without flip."""
    n_flipped, _, seq_len = flipped_input_ids_lst.shape
    trigger_len = trigger_slice.stop - trigger_slice.start

    # Hack to repeat labels:  # TODO fix this hack
    calc_loss_for_eval_kwargs = _hack_to_enlarge_kwargs(kwargs, n_flipped)

    losses = model.calc_loss_for_eval(
        inputs={
            'input_ids': flipped_input_ids_lst.view(-1, seq_len),  # flatten to calculate loss
            'attention_mask': attention_mask.repeat(n_flipped, 1)  # repeat the attention_mask for each flip
        },
        **calc_loss_for_eval_kwargs
    )
    losses = losses.view(n_flipped, -1).mean(dim=-1)  # aggregate per flip, (n_flipped,)

    # 5.2. Calculate the loss without flip
    loss_without_flip = model.calc_loss_for_eval(
        inputs={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        },
        **kwargs
    ).mean().item()

    # 5.3 [Optional] Add fluency score / l2 penalty
    if flu_alpha != 0 and flu_model is not None:
        flipped_input_ids_lst__for_flu = flipped_input_ids_lst.view(-1, seq_len)
        input_ids__for_flu = input_ids

        fluency_scores = flu_model.calc_fluency_scores(
            inputs={
                'input_ids': flipped_input_ids_lst__for_flu,  # flatten to calculate loss
                'attention_mask': attention_mask[..., :flipped_input_ids_lst__for_flu.shape[-1]].repeat(n_flipped, 1)
                # repeat the attention_mask for each flip
            },
        )
        fluency_scores = fluency_scores.view(n_flipped, -1).mean(dim=-1)  # aggregate per flip, (n_flipped,)
        losses += fluency_scores * flu_alpha

        flu_without_flip = flu_model.calc_fluency_scores(
            inputs={
                'input_ids': input_ids__for_flu,
                'attention_mask': attention_mask[..., :input_ids__for_flu.shape[-1]]
            },
        ).mean().item()
        loss_without_flip += flu_without_flip * flu_alpha
    if l2_alpha != 0:
        emb = model.embed(inputs={'input_ids': flipped_input_ids_lst.view(-1, seq_len),
                                  'attention_mask': attention_mask.repeat(n_flipped, 1)})
        losses += emb.norm(p=2, dim=-1).mean(dim=-1) * -l2_alpha
        loss_without_flip += model.embed(inputs={'input_ids': input_ids, 'attention_mask': attention_mask}).norm(p=2, dim=-1).mean().item() * -l2_alpha
    return losses, loss_without_flip
