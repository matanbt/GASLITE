"""
Perform the major evaluate required for a BEIR dataset, and cache it for later use.
    A later use can include injecting the similarity score of an adversarial passage and utilize the cache to re-produce
    the evaluation.
"""
import json
import os

from beir.retrieval import models, evaluation
import logging

from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer

from src import data_utils

logger = logging.getLogger(__name__)


def save_beir_eval(
        dataset_name: str,
        model_hf_name: str,
        sim_func_name: str = "cos_sim",  # or "dot"
        batch_size: int = 256,
        data_split: str = "train",  # or "test"
        data_portion: float = 1.0,
):
    out_dir = os.path.join(os.getcwd(), "data", "cached_evals")
    out_file = os.path.join(out_dir, _get_cached_eval_name(dataset_name, data_split,
                                                           data_portion, model_hf_name, sim_func_name))
    if os.path.exists(out_file):
        logger.info(f"[!!!] Cached results already found at `{out_file}`. Skipping.")
        return

    corpus, queries, qrels, _ = data_utils.load_dataset(dataset_name,
                                                        data_split=data_split,
                                                        data_portion=data_portion,
                                                        embedder_model_name=model_hf_name)

    # Hack to allow models code execution [TODO REPLACE ME I'M UGLY]
    TRUST_CODE_MODELS = ["nomic-ai/nomic-embed-text-v1", "dunzhang/stella_en_1.5B_v5", 'Alibaba-NLP/gte-base-en-v1.5']
    if model_hf_name in TRUST_CODE_MODELS:
        model = models.SentenceBERT()
        sent_model = SentenceTransformer(model_hf_name, trust_remote_code=True)
        model.q_model = model.doc_model = sent_model
        beir_model = DenseRetrievalExactSearch(  # Re-load and wrap the same model attacked
            model,  # use the model object
            batch_size=batch_size,
        )
    else:  # load regular model
        beir_model = DenseRetrievalExactSearch(  # Re-load and wrap the same model attacked
            # models.SentenceBERT("sentence-transformers/all-MiniLM-L6-v2"),  #  Note that it has default normalization
            models.SentenceBERT(model_hf_name),  # re-load the model from HuggingFace, with BEIR's
            # model  # use the model object
            # corpus_chunk_size=40000,
            batch_size=batch_size,
        )

    # Load retrieval model and perform retrieval on the dataset
    retriever = evaluation.EvaluateRetrieval(beir_model,
                                             score_function=sim_func_name,
                                             k_values=[1, 3, 5, 10, 20, 100, 1000])
    results = retriever.retrieve(corpus, queries)

    # Evaluate on general retrieval metrics
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric='mrr')
    logger.info(f"\n {ndcg}, \n {_map}, \n {recall}, \n {precision}, \n {mrr} \n ")

    # Sort each result entry by the score
    for qid in results:
        results[qid] = dict(sorted(results[qid].items(), key=lambda x: x[1], reverse=True))

    # [OPTIONAL] Compress them to top-1500
    results = _compress_retrieval_results(results, top_k=1500)

    # Save the results to `.json` file
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved the results to `{out_file}`.")


def _compress_retrieval_results(results, top_k=1001):
    sub_results = {}
    for query_id in results:
        # Sort the results by the score (just in case it's not sorted)
        scores = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)

        # Compress the results to top-k
        sub_results[query_id] = {}
        for p_id, score in scores[:top_k]:
            sub_results[query_id][p_id] = score

    return sub_results


def load_cached_eval(
        dataset_name: str,
        model_hf_name: str,
        sim_func_name: str = "cos_sim",  # or "dot"
        data_split: str = "test",  # or "dot"
        data_portion: float = 1.0,

        load_from_hf: bool = True,
):
    """Returns the cached results, on None if not found."""
    if load_from_hf:
        # Download the relevant results file
        from huggingface_hub import hf_hub_download
        filename = _get_cached_eval_name(dataset_name, data_split, data_portion, model_hf_name, sim_func_name)
        local_results_path = hf_hub_download(repo_id="MatanBT/retrieval-datasets-similarities", filename=filename,
                                             repo_type='dataset',
                                             cache_dir="./data",
                                             local_dir="cached_retrieval_evals")
    else:  # otherwise performs legacy loading
        out_dir = os.path.join(os.getcwd(), "data", "cached_evals")
        os.makedirs(out_dir, exist_ok=True)
        local_results_path = os.path.join(out_dir, _get_cached_eval_name(dataset_name, data_split,
                                                                         data_portion, model_hf_name, sim_func_name))
    if not os.path.exists(local_results_path):
        logger.warning(f"WARNING: No cached results found at `{local_results_path}`.")
        return None  # no cached results found

    with open(local_results_path, 'r') as f:
        results = json.load(f)
    logger.info(f"Loaded the results from `{local_results_path}`.")

    # verify that each result entry is sorted by the score
    for qid in results:
        assert list(results[qid].items()) == sorted(results[qid].items(), key=lambda x: x[1], reverse=True), \
            "Each result entry must be sorted by the score."

    return results


def _get_cached_eval_name(dataset_name: str, data_split: str, data_portion: float,
                          model_hf_name: str, sim_func_name: str):
    return f"{dataset_name}-{data_split}_{data_portion}_{model_hf_name.split('/')[-1]}_{sim_func_name}.json"

