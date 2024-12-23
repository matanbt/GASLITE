import math
from copy import deepcopy
from typing import Dict, List, Tuple, Union
import logging

import numpy as np
import torch.nn.functional as F

import torch
from beir.retrieval import evaluation

from src.evaluate.evaluate_beir_offline import load_cached_eval
from src.models.retriever import RetrieverModel

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = [1, 3, 5, 10, 15, 20, 30, 50, 100, 1000]


def _get_scores_for_adv_passage(
        # Attack-specific:
        adv_passage_vecs: torch.Tensor,  # the vector representation of the adversarial passage; (emb_dim,).
        adv_passage_ids: List[str],  # the ids of the adversarial passages
        attacked_qids: List[str],  # list of query-ids that were attacked

        # General:
        qid_to_emb: Dict[str, torch.Tensor],  # maps each query-id to the query embedding; (emb_dim,).
        sim_func_name: str = "cos_sim",  # or "dot"

        # Tech:
        batch_size: int = 64,
):
    """
    Returns a dict that maps each query-id to a dict of the adv-passage name mapped to its similarity score.
    """

    assert sim_func_name in ["cos_sim", "dot"], f"Unknown similarity function: {sim_func_name}"

    adv_results = {qid: {} for qid in attacked_qids}  # {qid -> {adv_pid1: score1, adv_pid1: score2, ...}}

    q_embs = torch.stack([qid_to_emb[qid] for qid in attacked_qids])  # shape: (n_queries, emb_dim)
    sim_to_adv = torch.zeros(len(attacked_qids), len(adv_passage_vecs), device='cpu')  # in CPU to avoid GPU OOM

    for i in range(0, len(adv_passage_vecs), batch_size):
        adv_passage_vecs_batch = adv_passage_vecs[i:i + batch_size]
        if sim_func_name == 'cos_sim':
            sim_to_adv_batch = F.cosine_similarity(q_embs.unsqueeze(1), adv_passage_vecs_batch.unsqueeze(0), dim=-1)  # shape: (n_queries, n_adv_p_batch)
        elif sim_func_name == 'dot':
            sim_to_adv_batch = torch.matmul(q_embs, adv_passage_vecs_batch.t())  # shape: (n_queries, n_adv_passages)
        sim_to_adv[:, i:i + batch_size] = sim_to_adv_batch.cpu()

    for q_idx, qid in enumerate(attacked_qids):
        for p_idx, pid in enumerate(adv_passage_ids):
            adv_results[qid].update({pid: sim_to_adv[q_idx, p_idx].item()})

    return adv_results


def _inject_adv_results_to_results(
        results: Dict[str, Dict[str, float]],  # the results dict to inject the scores into (it is deep-copied)
        adv_results: Dict[str, Dict[str, float]],  # the adversarial results to inject
):
    """
    Injects the adversarial results to the `results` dict; given `results` dict is not changed.
    """
    injected_results = deepcopy(results)
    for qid, adv_pid_to_score in adv_results.items():
        injected_results[qid].update(adv_pid_to_score)
    return injected_results


def _evaluate_with_adv_passage(
        results: Dict[str, Dict[str, float]],  # the full retrieval results to evaluate with (after inserting the adv scores)
        pid_advs: List[str],  # the pid of the adversarial passages to evaluate
        attacked_gold_qrels: Dict[str, Dict[str, int]],  # qid -> {pid1: gold_rank, pid2: gold_rank, ...}

        # metrics:
        k_values: List[int] = DEFAULT_K_VALUES,
        metrics_suffix='',  # to format the returned metrics names
        return_gold_metrics: bool = False,
        return_for_k_values: bool = False,
):
    max_evaluated_rank = max(k_values) + 1
    results = {qid: results[qid] for qid in attacked_gold_qrels.keys()}  # filter `results` by attacked queries
    attacked_adv_qrels = {qid: {adv_pid: 1 for adv_pid in pid_advs} for qid in attacked_gold_qrels.keys()}

    # Evaluate classic metrics on benign data (i.e. relative to golden passages):
    ndcg, _map, recall, precision = evaluation.EvaluateRetrieval.evaluate(attacked_gold_qrels, results, k_values)
    mrr = evaluation.EvaluateRetrieval.evaluate_custom(attacked_gold_qrels, results, k_values, metric='mrr')
    logger.info(f"\n [BENIGN PASSAGES] \n {ndcg}, \n {_map}, \n {recall}, \n {precision}, \n {mrr} \n ")
    # Evaluate relative to adversarial passages (as if the goal is to reach them):
    ndcg_adv, _map_adv, recall_adv, precision_adv = evaluation.EvaluateRetrieval.evaluate(attacked_adv_qrels, results, k_values)
    mrr_adv = evaluation.EvaluateRetrieval.evaluate_custom(attacked_adv_qrels, results, k_values, metric='mrr')
    logger.info(f"\n [ADV PASSAGES] \n {ndcg_adv}, \n {_map_adv}, \n {recall_adv}, \n {precision_adv}, \n {mrr_adv} \n ")

    # Evaluate on adversarial metrics:
    # - Average ranks we find adv or golden passage
    adv_ranks = np.full(len(attacked_gold_qrels), max_evaluated_rank)
    gold_ranks = np.full(len(attacked_gold_qrels), max_evaluated_rank)
    # - Average similarity scores for adv or golden passage
    adv_scores = np.full(len(attacked_gold_qrels), -np.inf)
    gold_scores = np.full(len(attacked_gold_qrels), -np.inf)

    for i, q in enumerate(attacked_gold_qrels.keys()):
        pid_and_score = sorted(list(results[q].items()), key=lambda x: x[1], reverse=True)
        gold_cou = 0
        curr_adv_qrels = set(attacked_adv_qrels[q].keys())
        curr_gold_qrels = set(attacked_gold_qrels[q].keys())

        # find the index (=rank) where the adv-passage / golden-passage are
        for rank, (pid, score) in enumerate(pid_and_score):
            if pid in curr_adv_qrels:
                adv_ranks[i] = min([rank, adv_ranks[i]])  # we care about the first time we find the adv-passage
                adv_scores[i] = max([score,  adv_scores[i]])
                logger.info(f">> {q=}, {pid=}, {rank=}, {score=}  [[ADVERSARIAL]]")
            if pid in curr_gold_qrels:
                gold_ranks[i] = min([rank, gold_ranks[i]])
                gold_scores[i] += max([score,  adv_scores[i]])
                gold_cou += 1
                logger.info(f">> {q=}, golden-{pid=}, {rank=}, {score=}")

        if gold_cou >= 1:
            gold_scores[i] /= gold_cou
        if gold_ranks[i] == max_evaluated_rank:  # if the golden-passage was not found in the results (quite odd...)
            logger.info(f">> {q=}, golden-{attacked_gold_qrels[q]=}, {'not found in results'}")
        if adv_ranks[i] == max_evaluated_rank:  # if the adv passage was not found in the results
            logger.info(f">> {q=}, adv passage not found in results  [[ADVERSARIAL]]")

    # Summary:
    logger.info(f"average gold-passage rank: {gold_ranks.mean()}")
    logger.info(f"average adv-passage rank: {adv_ranks.mean()}")
    logger.info(f"average adv-passage rank: {adv_scores}")
    logger.info(f"average adv-passage mean-score: {adv_scores.mean()}")
    logger.info(f"average gold-passage mean-score: {gold_scores.mean()}")
    logger.info(f"average adversary wins over golden: {(adv_ranks < gold_ranks).mean()}")

    metrics = {
        f'adv_MRR@10{metrics_suffix}': mrr_adv['MRR@10'],
        f'adv_NDCG@10{metrics_suffix}': ndcg_adv['NDCG@10'],
        f'adv_RECALL@10{metrics_suffix}': recall_adv['Recall@10'],
        f'adv_appeared@10{metrics_suffix}': adv_ranks[adv_ranks < 10].size / adv_ranks.size,
        f'adv_ranks_mean{metrics_suffix}': adv_ranks.mean(),  # main metric, lower means better
        f'adv_scores_mean{metrics_suffix}': adv_scores.mean(),
        f'adv_wins_over_golden{metrics_suffix}': (adv_ranks < gold_ranks).mean(),  # main metric, higher means better
        f'adv_ranks_lst{metrics_suffix}': adv_ranks.tolist(),

        # f'avg_scores_ratio{metrics_suffix}': (adv_scores / gold_scores).mean(),
        # f'avg_ranks_diff{metrics_suffix}': (gold_ranks - adv_ranks).mean(),  # inverse?
    }

    if return_gold_metrics:
        metrics.update({
            f'gold_MRR@10{metrics_suffix}': mrr['MRR@10'],
            f'gold_NDCG@10{metrics_suffix}': ndcg['NDCG@10'],
            f'gold_RECALL@10{metrics_suffix}': recall['Recall@10'],
            f'gold_appeared@10{metrics_suffix}': gold_ranks[gold_ranks < 10].size / gold_ranks.size,
            f'gold_ranks_mean{metrics_suffix}': gold_ranks.mean(),
            f'gold_scores_mean{metrics_suffix}': gold_scores.mean(),
        })
    if return_for_k_values is not None:
        for k in k_values:
            metrics.update({
                f'adv_MRR@{k}{metrics_suffix}': mrr_adv[f'MRR@{k}'],
                f'adv_NDCG@{k}{metrics_suffix}': ndcg_adv[f'NDCG@{k}'],
                f'adv_RECALL@{k}{metrics_suffix}': recall_adv[f'Recall@{k}'],
                f'adv_appeared@{k}{metrics_suffix}': adv_ranks[adv_ranks < k].size / adv_ranks.size,
            })
    return metrics


def full_evaluation_with_adv_passage_vecs(
        # Attack-specific:
        adv_passage_vecs: Union[List[torch.Tensor], torch.Tensor],
        attacked_qrels: Dict[str, Dict[str, int]],

        # General:
        results: Dict[str, Dict[str, float]],
        qid_to_emb: Dict[str, torch.Tensor],
        sim_func_name: str = "cos_sim",

        # Metrics:
        k_values: List[int] = DEFAULT_K_VALUES,
        metrics_suffix: str = '',
        return_gold_metrics: bool = False,
        return_for_k_values: bool = False,
):
    """
    Returns a dict of metrics for the adversarial passage.
    :param adv_passage_vecs: the list of  vector representations of the adversarial passage; each vec (emb_dim,).
    :param attacked_qrels: attacked queries to golden p's mapping; qid -> {pid1: gold_rank, pid2: gold_rank, ...}.
    :param results: the retrieval results to evaluate with (before inserting the adv scores)
    :param qid_to_emb: maps each query-id to the query embedding; (emb_dim,).
    :param sim_func_name: "cos_sim" or "dot"

    :param k_values: the rank values to evaluate on
    :param metrics_suffix:  to format the returned metrics names
    :param return_gold_metrics: whether to return the metrics for the golden passages as well
    :param return_for_k_values: whether to return all the k-values (in `k_values`).
    :return:
    """
    assert sim_func_name in ["cos_sim", "dot"], f"Unknown similarity function: {sim_func_name}"

    pid_advs = [f'__adv{i}__' for i in range(len(adv_passage_vecs))]  # set available pids to name adversarial passages
    if isinstance(adv_passage_vecs, list):
        adv_passage_vecs = torch.stack(adv_passage_vecs, dim=0)  # shape: (n_adv_passages, emb_dim)
    # set adv_passage_ids to the same device as `qid_to_emb`:
    adv_passage_vecs = adv_passage_vecs.to(qid_to_emb[list(qid_to_emb.keys())[0]].device)

    # 1. Calculate the scores for the adversarial passage:
    adv_results = _get_scores_for_adv_passage(
        adv_passage_vecs=adv_passage_vecs,
        adv_passage_ids=pid_advs,
        attacked_qids=list(attacked_qrels.keys()),
        qid_to_emb=qid_to_emb,
        sim_func_name=sim_func_name,
    )

    # 2. Inject the adversarial results to the `results` dict (containing the benign evaluation):
    updated_results = _inject_adv_results_to_results(
        results=results,
        adv_results=adv_results,
    )

    # 3. Evaluate with the adversarial passage(s):
    metrics = _evaluate_with_adv_passage(
        results=updated_results,
        pid_advs=pid_advs,
        attacked_gold_qrels=attacked_qrels,
        k_values=k_values,
        metrics_suffix=metrics_suffix,
        return_gold_metrics=return_gold_metrics,
        return_for_k_values=True,
    )

    return metrics


def get_result_list_for_query(
        # Attack-specific:
        adv_passage_texts: List[str],
        query_id: str,

        # adv_passage_vecs: Union[List[torch.Tensor], torch.Tensor],  # TODO OPTIONAL
        # query_text: str,  # TODO to support custom queries we need to embed the entire corpus

        # General:
        model: RetrieverModel,
        queries: Dict[str, str],  # qid -> query text
        corpus: Dict[str, Dict[str, str]],  # pid -> passage text

        # To fetch `results`:
        dataset_name: str,
        data_split: str,
        data_portion: float,

        # results: Dict[str, Dict[str, float]] = None,

        # Metrics:
        top_k: int = 10,  # number of top passages to return
):
    # Load results (works for previously known queries)
    results = load_cached_eval(  # query_id -> {passage_id -> sim score}
        dataset_name=dataset_name,
        model_hf_name=model.model_hf_name,
        sim_func_name=model.sim_func_name,
        data_split=data_split,
        data_portion=data_portion,
    )
    if results is None:
        raise Exception("No cached retreival results found.")  # no cached results found

    # Get the query embeddings
    query_text = queries[query_id]
    query_emb = model.embed(texts=[query_text])

    # Calculate similarity scores for the adversarial passages
    adv_p_emb = model.embed(texts=adv_passage_texts)
    if model.sim_func_name == 'cos_sim':
        adv_sim_scores = torch.nn.functional.cosine_similarity(query_emb, adv_p_emb)
    elif model.sim_func_name == 'dot':
        adv_sim_scores = torch.matmul(query_emb, adv_p_emb.t())
    adv_sim_score = adv_sim_scores.max().item()  # we only care about the best passage (when examining a single query)
    adv_passage_text = adv_passage_texts[adv_sim_scores.argmax().item()]

    # Get the top-k passages for the query
    query_results = results[query_id]

    # calculate what's the rank of the adv passage:
    adv_rank = sum(1 for score in query_results.values() if score > adv_sim_score)
    adv_rank = math.inf if adv_rank == len(query_results) else adv_rank + 1

    # list top-k passages, including the adv-passage
    query_results["__adv__"] = adv_sim_score  # include the adversarial passage in results
    top_passages = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_passages_text = [corpus[pid]['text'] if pid != "__adv__" else adv_passage_text for pid, _ in top_passages]

    return dict(
        query_text=query_text,
        adv_sim_score=adv_sim_score,
        adv_rank=adv_rank,

        top_passages=top_passages,
        top_passages_text=top_passages_text,
    )
