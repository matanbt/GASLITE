import json
import random

import numpy as np
from huggingface_hub.utils import EntryNotFoundError

from src import data_utils
from src.attacks.gaslite import gaslite_attack
from src.covering.covering import CoverAlgorithm
from src.covering.covering import calc_centroid_objective
from src.models.retriever import RetrieverModel

from typing import List, Dict, Any
import torch

import logging

logger = logging.getLogger(__name__)


def attack_ret(
        batch_size: int = 128,

        query_choice: str = 'cluster',  # 'naive', 'cluster'
        target_choice: str = 'queries',  # 'queries' (accumalates grad of all queries), 'centroid'  (targets the mean query) # TODO 'median' # TODO queries_with_non_square_l2_loss
        sim_func_name: str = 'dot',  # dot', 'cos_sim'

        # GASLITE config:
        attack_n_iter: int = 200,
        n_flips_to_gen: int = 150,
        n_sim_tokens_flips_to_gen: int = 50,
        evaluate_flips_on_strs: bool = True,
        adv_loss_name: str = 'sum_of_sim',
        perform_arca_opt: bool = False,
        avg_grad_batch_size: int = 1,
        k_candidates: int = 200,
        flu_alpha: float = 0,
        flu_alpha_inc: dict = {},
        l2_alpha: float = 0,
        fluency_model_name: str = "gpt2",
        use_rephraser: bool = False,
        beam_search_config: dict = False,
        time_limit_in_seconds: int = None,

        # Generated trigger config:
        adv_passage_init: str = 'random_golden',  # 'random_golden', 'dummy_token', 'random_passage' (from corpus)
        trigger_len: int = 25,
        trigger_loc: str = 'override_prefix',  # 'override_suffix', 'add_suffix', 'add_prefix', 'override_middle'
        mal_info: str = None,  # defaults to a random toxic passage
        random_seed: int = 42,

        # Cover alg:
        cover_alg: dict = None,
        cluster_idx: int = None,

        model_hf_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # model to attack
        dataset_name: str = "scifact",  # dataset to attack
        data_split: str = 'train',  # we craft the attack on the training-set
        data_portion=1.0,  # to simulate an attacker's limited access to the data
        **kwargs
):
    # Load model:
    model = RetrieverModel(
        model_hf_name=model_hf_name,
        sim_func_name=sim_func_name,
        adv_loss_name=adv_loss_name,
        max_batch_size=batch_size,
    )

    # Load dataset:
    _, _, qrels, qp_pairs_dataset = data_utils.load_dataset(
        dataset_name=dataset_name,
        data_split=data_split,
        data_portion=data_portion,
        embedder_model_name=model_hf_name,
    )
    # Set random seed only after loading dataset
    from transformers import set_seed
    set_seed(random_seed)
    # Embed all the (training) queries:
    q_embs = model.embed(qp_pairs_dataset['query']).cuda()
    # Map query-ids to their embeddings
    qid_to_emb = {qid: q_embs[i] for i, qid in enumerate(qp_pairs_dataset['query_id'])}

    # Load the covering scheme:
    if cover_alg.get('concept_qids', None) is not None:
        concept_qids = cover_alg['concept_qids']
        cover_alg['concept_train_qids'] = concept_qids[:int(len(concept_qids) * cover_alg['concept_portion_to_train'])]
        cover_alg['filter_in_qids_name'] = f"heldin-concept-{cover_alg['concept_name']}-{cover_alg['concept_portion_to_train']}"
    if cover_alg['data_split'] == 'gen_qs':  # Hack to allow synthetic queries
        cover_alg['concept_train_qids'] = [cover_alg['concept_name']]
        cover_alg['filter_in_qids_name'] = f"gen_qs-{cover_alg['concept_name']}",
    cover_algo = CoverAlgorithm(
        model_hf_name=model_hf_name,
        sim_func=sim_func_name,
        batch_size=batch_size,

        dataset_name=dataset_name,
        n_clusters=cover_alg['n_clusters'],
        covering_algo_name=cover_alg['covering_algo_name'],
        data_split=cover_alg['data_split'],
        data_portion=cover_alg['data_portion'],

        # When targeting a concept:
        filter_in_qids=cover_alg.get('concept_train_qids', None),
        filter_in_qids_name=cover_alg.get('filter_in_qids_name', None),
    )
    qid_to_cluster_idx, cluster_idx_to_qids, centroid_vecs = cover_algo.fit_predict()
    if 'toxic_passage' in cover_alg:  # if the toxic passage is predefined (e.g., by concept)
        mal_info = cover_alg['toxic_passage']

    # Fetch targeted queries and (golden) passage
    Q, Qid, Pgold, Pgold_id = get_qs_and_golden_ps(qp_pairs_dataset, query_choice, target_choice,
                                                   clustering_labels=qid_to_cluster_idx,
                                                   cluster_choice_kmeans=cluster_idx)
    logger.info(f"Attacked queries ids: {Qid}")
    logger.info(f"Attacked queries text: {Q}")

    # Calc objective artifacts
    emb_targets, centroid_vec = calc_objective(model, Q, sim_func_name, target_choice, centroid_vecs[cluster_idx])
    n_targets = len(emb_targets)

    # Initialize adversarial passage
    P_adv, trigger_slice, P_adv_before = initialize_p_adv(trigger_loc, trigger_len, adv_passage_init,
                                            mal_info=mal_info, model_hf_name=model_hf_name, model=model,
                                            dummy_token='!', Q=Q, Pgold=Pgold, qp_pairs_dataset=qp_pairs_dataset,
                                            n_targets=n_targets)
    P_adv = P_adv.to('cuda')
    logger.info(f"[[decoded-trigger-slice]]{model.tokenizer.decode(P_adv['input_ids'][0][trigger_slice])}")
    logger.info(f"[[full decoded-passage]]{model.tokenizer.decode(P_adv['input_ids'][0])}")

    # deep copy the original inputs
    P_adv_orig = P_adv_before['input_ids']

    # Execute the attack
    best_input_ids, out_metrics = gaslite_attack(
        model=model,
        trigger_slice=trigger_slice,
        inputs=P_adv,
        n_iter=attack_n_iter,
        n_flips_to_gen=n_flips_to_gen,
        n_sim_tokens_flips_to_gen=n_sim_tokens_flips_to_gen,
        evaluate_flips_on_strs=evaluate_flips_on_strs,
        perform_arca_opt=perform_arca_opt,
        n_grad=avg_grad_batch_size,
        k_candidates=k_candidates,
        use_rephraser=use_rephraser,
        beam_search_config=dict(beam_search_config),
        time_limit_in_seconds=time_limit_in_seconds,
        log_to='wandb',

        # Fluency:
        flu_alpha=flu_alpha,
        flu_alpha_inc=flu_alpha_inc,
        fluency_model_name=fluency_model_name,
        l2_alpha=l2_alpha,

        emb_targets=emb_targets.cuda(),
        emb_anchors=model.embed(texts=Pgold).cuda(),
    )

    P_adv['input_ids'] = best_input_ids

    # Print the attack:
    def _decode(_input_ids, is_clean=True):
        return model.tokenizer.decode(_input_ids, skip_special_tokens=is_clean, clean_up_tokenization_spaces=is_clean)

    # Note: it is important that when encoding the adv passage, the same tokenization (as was attacked) will be yieleded.
    # TODO consider simply evaluating on the `input_ids`, without decoding (as done in 'corpus_poisoning paper);
    #      this could improve performance, albeit could be argued as sub-realistic.
    adv_pass_before_attack = _decode(P_adv_orig[0])
    adv_pass_after_attack = _decode(P_adv['input_ids'][0])
    adv_tokens_list_after_attack = P_adv['input_ids'][0].cpu().tolist()
    adv_tokens_list_after_attack = adv_tokens_list_after_attack[:adv_tokens_list_after_attack.index(model.tokenizer.pad_token_id)]  # remove padding

    logger.info(f"\n BEFORE:\n {P_adv_orig[0]=} \n ----------------- "
                f"\n AFTER:\n {P_adv['input_ids'][0]=}")
    logger.info(f"\n BEFORE: {adv_pass_before_attack} \n ----------------- "
                f"\n AFTER:{adv_pass_after_attack}")
    logger.info(f"Decoding `adv_pass_after_attack` with special tokens: "
                f"{_decode(P_adv['input_ids'][0], is_clean=False)}")

    if not (P_adv_orig[0].cpu() ==model.tokenizer(adv_pass_before_attack, return_tensors="pt",
                                                               padding='max_length', truncation=True)['input_ids'][0].cpu()).all():
        logger.warning("[WARNING] Note that the `adv_pass_before_attack` cannot be reversed correctly and "
                       "might degrade the attack performance.")
    orig = P_adv['input_ids'][0].cpu()
    dec = model.tokenizer(adv_pass_after_attack, return_tensors="pt", padding='max_length',
                          truncation=True)['input_ids'][0].cpu()
    if not (orig == dec).all():
        logger.warning("[WARNING] Note that the `adv_pass_after_attack` cannot be reversed correctly and "
                       "might degrade the attack performance.")
        # if evaluate_flips_on_strs:  # under this mode, the attack should have discard such flips
        #     raise AssertionError

    # Evaluate
    metrics = evaluate_attack(
        model=model,
        dataset_name=dataset_name,
        data_split=data_split,
        attacked_qids=Qid,
        qrels=qrels,
        qid_to_emb=qid_to_emb,
        adv_text_before_attack=adv_pass_before_attack,
        adv_text_after_attack=adv_pass_after_attack,
        adv_tokens_list_after_attack=adv_tokens_list_after_attack,
        sim_func_name=sim_func_name,
        model_hf_name=model_hf_name,
        centroid_vec=centroid_vec,
        best_flu_instance_text=out_metrics.get('best_flu_instance_text', None),
    )
    metrics.update(out_metrics)

    return metrics


def get_qs_and_golden_ps(
        qp_pairs_dataset, query_choice, target_choice, clustering_labels, cluster_choice_kmeans,
):
    # >>>>  Choose queries and target to attack:
    Q, Qid, Pgold, Pgold_id = [], [], [], []
    n_samples = None

    if query_choice == 'cluster' or target_choice == 'predefined_centroid':
        # Option 1: Choose queries from a single K-Means cluster (analogue to attacking a semantic area)
        for idx, qid in enumerate(qp_pairs_dataset['query_id']):
            if qid in Qid:
                continue  # skip if already added the query id (shouldn't get here)
            if clustering_labels.get(qid, -1) == cluster_choice_kmeans:
                Q.append(qp_pairs_dataset['query'][idx])
                Qid.append(qp_pairs_dataset['query_id'][idx])
                Pgold.append(qp_pairs_dataset['passage'][idx])
                Pgold_id.append(qp_pairs_dataset['pid'][idx])
    elif query_choice.startswith('from_file__') and '3rd_party' in query_choice:
        clusters_file_name = query_choice.split('___')[1]
        cluster_idx = int(query_choice.split('___')[2])
        with open(clusters_file_name, 'r') as f:
            query_choice = json.load(f)[cluster_idx]
        for idx, qid in enumerate(qp_pairs_dataset['query_id']):
            if qid in query_choice:
                Q.append(qp_pairs_dataset['query'][idx])
                Qid.append(qp_pairs_dataset['query_id'][idx])
                Pgold.append(qp_pairs_dataset['passage'][idx])
                Pgold_id.append(qp_pairs_dataset['pid'][idx])
    elif query_choice == 'single_random':  # a single query should be chosen, according to the random seed
        qid = random.choice(qp_pairs_dataset['query_id'])
        idx = qp_pairs_dataset['query_id'].index(qid)
        Q.append(qp_pairs_dataset['query'][idx])
        Qid.append(qp_pairs_dataset['query_id'][idx])
        Pgold.append(qp_pairs_dataset['passage'][idx])
        Pgold_id.append(qp_pairs_dataset['pid'][idx])
    elif isinstance(query_choice, list):  # query_choice is a list of pre-chosen query-ids
        for idx, qid in enumerate(qp_pairs_dataset['query_id']):
            if qid in query_choice:
                Q.append(qp_pairs_dataset['query'][idx])
                Qid.append(qp_pairs_dataset['query_id'][idx])
                Pgold.append(qp_pairs_dataset['passage'][idx])
                Pgold_id.append(qp_pairs_dataset['pid'][idx])
    if len(Q) == 0:  # no `if` was chosen
        # Option 0: Naive choice of the first n_samples queries (also relevant for predefined centroid, where the cluster is unknown).
        # Forms mock cluster of `n_samples` queries
        n_samples = 5
        Q = qp_pairs_dataset['query'][:n_samples]
        Qid = qp_pairs_dataset['query_id'][:n_samples]
        Pgold = qp_pairs_dataset['passage'][:n_samples]
        Pgold_id = qp_pairs_dataset['pid'][:n_samples]
    if n_samples is not None:
        assert n_samples <= len(
            Q), f"{n_samples=} must be smaller/eq to the number of queries in the cluster ({len(Q)})"
        Q, Qid, Pgold, Pgold_id = Q[:n_samples], Qid[:n_samples], Pgold[:n_samples], Pgold_id[:n_samples]
    else:
        n_samples = len(Q)

    return Q, Qid, Pgold, Pgold_id


def calc_objective(
        model, Q, sim_func, target_choice,
        predefined_centroid_vec
):
    # calc centroid:
    if target_choice != 'predefined_centroid':
        centroid_vec = calc_centroid_objective(
            q_embs=model.embed(Q),
            sim_func=sim_func,
            do_normalize=False,
        ).unsqueeze(0)
    else:
        centroid_vec = predefined_centroid_vec.unsqueeze(0)

    if target_choice == 'queries':
        # Option 1: choose the queries batch as targets
        emb_targets = model.embed(Q)
    elif target_choice in ['centroid', 'predefined_centroid']:
        # Option 2: choose "artificial" query, that can cover many others via close algorithm (K-means' centroid)
        emb_targets = centroid_vec
    else:
        raise ValueError(f"No suitable {target_choice=}")

    return emb_targets, centroid_vec


def initialize_p_adv(trigger_loc, trigger_len, adv_passage_init,
                     model, mal_info=None, model_hf_name='', dummy_token='!',
                     Q=None, Pgold=None, qp_pairs_dataset=None, n_targets=1):
    # construct adv passage, e.g., "{instruction prefix} {mal info}  {init trigger}"
    # 1. Model-specific instruction for passages:
    embedder_passage_prefix = ''
    if model_hf_name.endswith("e5-base-v2"):
        embedder_passage_prefix = "passage: "

    # 2. Malicious Information:
    if mal_info is None or mal_info == 'random_toxic_text':
        from datasets import load_dataset
        toxic_dataset = load_dataset("toxigen/toxigen-data")['train']
        toxic_prefixes = [row['text'] for row in toxic_dataset
                          if row['toxicity_ai'] == 5 and 50 < len(
                row['text']) < 100]  # filter to highly toxic text, with reasonable length
        mal_info = random.choice(toxic_prefixes).strip()
    # add period at the end of the toxic prefix if it doesn't end with one
    if mal_info[-1] not in ['.', '!', '?']:
        mal_info += '.'

    # 3. Trigger initialization:
    if adv_passage_init == 'lm_gen':
        from transformers import pipeline
        generator = pipeline('text-generation', model='gpt2')
        gen_texts = generator(mal_info, return_full_text=False, max_new_tokens=trigger_len * 2, num_return_sequences=5)
        # Choose the generated text of maximal length
        gen_text = max(gen_texts, key=lambda x: len(x['generated_text']))['generated_text']
        trigger_init = gen_text * 20  # repeat to match the trigger len
    elif adv_passage_init == 'dummy_token':
        trigger_init = dummy_token * trigger_len
    elif adv_passage_init == 'random_passages':
        assert qp_pairs_dataset is not None, f"{qp_pairs_dataset=}"
        trigger_init = ' '.join(random.choices(qp_pairs_dataset['passage'], k=10))  # repeat to match the trigger len
    elif adv_passage_init == 'golden_passages':
        assert Pgold is not None, f"{Pgold=}"
        trigger_init = ' '.join(random.choices(Pgold, k=min(len(Pgold), 10)))  # repeat to match the trigger len
    elif adv_passage_init == 'att_queries':
        assert Q is not None, f"{Q=}"
        trigger_init = ' '.join(random.choices(Q, k=min(len(Q), 20)))  # repeat to match the trigger len
    else:
        raise ValueError(f"No suitable {adv_passage_init=}")

    # 4. Tokenize {1,2,3}
    embedder_passage_prefix = model.tokenizer(embedder_passage_prefix, add_special_tokens=False)['input_ids']
    mal_info = model.tokenizer(mal_info, add_special_tokens=False)['input_ids']
    trigger_init = model.tokenizer(trigger_init, add_special_tokens=False)['input_ids'][:trigger_len]

    # 4'. Retokenize (decode and encode) to avoid weird tokens
    mal_info = model.tokenizer(model.tokenizer.decode(mal_info), add_special_tokens=False)['input_ids']
    trigger_init = model.tokenizer(model.tokenizer.decode(trigger_init), add_special_tokens=False)['input_ids'][
                   :trigger_len]

    # 4''. Fill the trigger_init if needed
    trigger_init += model.tokenizer(dummy_token, add_special_tokens=False)['input_ids'] * (
                trigger_len - len(trigger_init))

    # 5. Construct the tokenized adv passage (following `trigger_loc`)
    trigger_slice = None
    if trigger_loc == 'suffix':
        P_adv = embedder_passage_prefix + mal_info + trigger_init
        trigger_slice = slice(len(embedder_passage_prefix) + len(mal_info),
                              len(embedder_passage_prefix) + len(mal_info) + trigger_len)
    elif trigger_loc == 'prefix':
        P_adv = embedder_passage_prefix + trigger_init + mal_info
        trigger_slice = slice(len(embedder_passage_prefix),
                              len(embedder_passage_prefix) + trigger_len)
    elif trigger_loc == 'middle':  # can be done in a more sophisticated way
        P_adv = embedder_passage_prefix + mal_info[:len(mal_info) // 2] + trigger_init + mal_info[len(mal_info) // 2:]
        trigger_slice = slice(len(embedder_passage_prefix) + len(mal_info) // 2,
                              len(embedder_passage_prefix) + len(mal_info) // 2 + trigger_len)
    elif trigger_loc == 'trigger_only':
        P_adv = embedder_passage_prefix + trigger_init
        trigger_slice = slice(len(embedder_passage_prefix), len(embedder_passage_prefix) + trigger_len)
    else:
        raise ValueError(f"No suitable {trigger_loc=}")

    # 6. Add special tokens (e.g., [CLS], [SEP], [PAD])
    print(">>")
    P_adv_before = model.tokenizer.decode(embedder_passage_prefix + mal_info)
    logger.info(f"0: {P_adv_before}")
    P_adv_before = model.tokenizer([P_adv_before] * n_targets, return_tensors="pt",
                                   padding='max_length', truncation=True)
    logger.info(f"1: {P_adv_before}")
    P_adv = model.tokenizer.decode(P_adv)
    logger.info(f"2: {P_adv}")
    P_adv = model.tokenizer([P_adv] * n_targets, return_tensors="pt",
                            padding='max_length', truncation=True)
    logger.info(f"3: {P_adv}")

    # 6'. Offset the trigger slice according to the special tokens (e.g., [CLS])
    MODELS_WITHOUT_BOS = ['dunzhang/stella_en_1.5B_v5']
    if model_hf_name not in MODELS_WITHOUT_BOS:
        trigger_slice = slice(trigger_slice.start + 1, trigger_slice.stop + 1)

    # Validation  # [DISABLED: as last token modify sometimes, e.g. due to merger, and we allow it)
    # assert (P_adv['input_ids'][0, trigger_slice] == torch.tensor(
    #     trigger_init)).all(), f"{P_adv[0, trigger_slice]=} != {trigger_init=}"

    return P_adv, trigger_slice, P_adv_before


def evaluate_attack(
        model,
        qrels: Dict[str, Dict[str, int]],  # qid -> {pid1: gold_rank, pid2: gold_rank, ...}
        qid_to_emb: Dict[str, torch.Tensor],  # maps each query-id to the query embedding; (emb_dim,).
        attacked_qids: List[str],
        adv_text_before_attack: str,
        adv_text_after_attack: str,
        adv_tokens_list_after_attack: List[int],
        sim_func_name: str,
        dataset_name: str,
        data_split: str,
        model_hf_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        centroid_vec: torch.Tensor = None,
        best_flu_instance_text: str = None,
) -> Dict[str, Any]:
    logger.info(f"Evaluating queries ids: {attacked_qids}")

    metrics = {}
    # limit evaluated queries to those attacked
    attacked_qrels = {qid: qrel for qid, qrel in qrels.items() if qid in attacked_qids}

    # Load the cached BEIR evaluation
    from src.evaluate.evaluate_beir_offline import load_cached_eval
    from src.evaluate.evaluate_beir_online import full_evaluation_with_adv_passage_vecs
    try:
        results = load_cached_eval(
            dataset_name=dataset_name,
            model_hf_name=model_hf_name,
            sim_func_name=sim_func_name,
            data_split=data_split,
            # data_portion=data_portion,  # TODO generalize
        )
    except (FileNotFoundError, EntryNotFoundError) as e:
        # In this case the results file doesn't exist, we create a mock results dict,
        # to allow the evaluation to proceed
        logger.warning(f"Cached evaluation file not found; creating a mock results dict for the evaluation. "
                       f"Note that as a result, results depends on the model's benign metrics are not true.")
        results = {qid: {} for qid in qrels}

    # BEFORE ATTACK
    # feed the corpus with the adv passage pre-attack
    logger.info(f"{'<' * 10} Evaluating with the adv passage pre-attack {'>' * 10} ")
    emb_adv_before_attack = model.embed(adv_text_before_attack).squeeze(0).cuda()
    metrics.update(full_evaluation_with_adv_passage_vecs(
        adv_passage_vecs=[emb_adv_before_attack],
        attacked_qrels=attacked_qrels,
        results=results,
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
        metrics_suffix='_before_attack',
        return_gold_metrics=True,
        #  k_values: List[int] = [1, 3, 5, 10, 100, 1000],
    ))

    # IMAGINARY ATTACK (centroid) - to be used as a comparable baseline ("what if we reached the centroid?")
    logger.info(f"{'<' * 10} Evaluating with the centroid of the attacked queries {'>' * 10} ")
    metrics.update(full_evaluation_with_adv_passage_vecs(
        adv_passage_vecs=[centroid_vec.squeeze(0)],
        attacked_qrels=attacked_qrels,
        results=results,
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
        metrics_suffix='_centroid_hyp_attack',
    ))

    # hyp attack with the best query in the cluster: `_best_query_hyp_attack`
    logger.info(f"{'<' * 10} Evaluating with the best query in the cluster {'>' * 10} ")
    best_query_emb = _get_best_query_emb(attacked_qids=attacked_qids, results=results, qid_to_emb=qid_to_emb)
    metrics.update(full_evaluation_with_adv_passage_vecs(
        adv_passage_vecs=[best_query_emb],
        attacked_qrels=attacked_qrels,
        results=results,
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
        metrics_suffix='_best_query_hyp_attack',
    ))

    # AFTER ATTACK
    # feed the corpus with the adv passage post-attack
    logger.info(f"{'<' * 10} Evaluating with the (text) adv passage post-attack {'>' * 10} ")
    emb_adv_after_attack = model.embed(adv_text_after_attack).squeeze(0).cuda()
    metrics.update(full_evaluation_with_adv_passage_vecs(
        adv_passage_vecs=[emb_adv_after_attack],
        attacked_qrels=attacked_qrels,
        results=results,
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
        metrics_suffix='_after_attack',
        return_for_k_values=True,  # return metrics for many k-values
    ))

    # AFTER ATTACK (tokens list)
    # This approach generated embedding directly on the token list generated in the attack.
    # Note: non-realistic evaluation approach; used in Corpus poisoning.
    logger.info(f"{'<' * 10} Evaluating with the (tokens-list) adv passage post-attack {'>' * 10} ")
    adv_toks_after_attack_pt = torch.tensor([adv_tokens_list_after_attack]).cuda()
    adv_toks_after_attack_pt_input = {
        'input_ids': adv_toks_after_attack_pt,
        'attention_mask': torch.ones_like(adv_toks_after_attack_pt).cuda()
    }
    emb_adv_tokens_list_after_attack = model.embed(inputs=adv_toks_after_attack_pt_input).squeeze(0).cuda()
    metrics.update(full_evaluation_with_adv_passage_vecs(
        adv_passage_vecs=[emb_adv_tokens_list_after_attack],
        attacked_qrels=attacked_qrels,
        results=results,
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
        metrics_suffix='__tokens_list__after_attack',
    ))

    # AFTER ATTACK WITH THE HIGHEST FLU TEXT:
    if best_flu_instance_text is not None:
        logger.info(f"{'<' * 10} Evaluating with the best flu text {'>' * 10} ")
        emb_best_flu_instance_text = model.embed(best_flu_instance_text).squeeze(0).cuda()
        metrics.update(full_evaluation_with_adv_passage_vecs(
            adv_passage_vecs=[emb_best_flu_instance_text],
            attacked_qrels=attacked_qrels,
            results=results,
            qid_to_emb=qid_to_emb,
            sim_func_name=sim_func_name,
            metrics_suffix='_best_flu__after_text',
        ))

    # Save attack artifacts:
    metrics.update(dict(
        attacked_qids=attacked_qids,
        n_attacked_qids=len(attacked_qids),
        adv_text_before_attack=adv_text_before_attack,
        adv_text_after_attack=adv_text_after_attack,
        adv_text_after_attack_tokens=adv_tokens_list_after_attack,
    ))

    return metrics


def _get_best_query_emb(
        attacked_qids: List[str],
        results: Dict[str, Dict[str, float]],
        qid_to_emb: Dict[str, torch.Tensor],
) -> torch.Tensor:
    attacked_q_embs = torch.stack([qid_to_emb[q] for q in attacked_qids])
    sim_matrix = torch.matmul(attacked_q_embs, attacked_q_embs.T)
    top_10_score_per_q = []
    for q_cand in attacked_qids:
        top_10_score_per_q.append(list(results[q_cand].items())[9][-1])
    top_10_score_per_q = torch.tensor(top_10_score_per_q).cuda()
    best_query_idx = (top_10_score_per_q < sim_matrix).sum(dim=-1).argmax().item()
    best_query_emb = attacked_q_embs[best_query_idx]

    return best_query_emb
