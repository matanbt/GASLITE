import json
import os
from typing import Dict, List, Tuple, Union

import torch

from src import data_utils
from src.evaluate.data_stats import RetrievalDataStats
from src.evaluate.evaluate_beir_offline import load_cached_eval
from src.evaluate.evaluate_beir_online import full_evaluation_with_adv_passage_vecs
from src.covering.covering_algos import covering_algo_name_to_func

import logging

from src.models.retriever import RetrieverModel

logger = logging.getLogger(__name__)


class CoverAlgorithm:
    """A class that wraps a covering algorithm and its evaluation on a given dataset.
    Note that we consider 'covering' wrp to the angles of the space, i.e., cos similarity."""
    def __init__(
            self,
            # Clustering algo params:
            covering_algo_name: str,
            n_clusters: int,

            # Model info:
            model_hf_name: str,
            sim_func: str,

            # Data to cluster:
            dataset_name: str,
            data_split: str = "train",
            data_portion: float = 0.02,
            filter_in_qids: List[str] = None,
            filter_in_qids_name: str = None,  # usually the name of the concept of which queries we filter

            # Tech parameters:
            batch_size: int = 512,
            **kwargs,
    ):
        # Training data:
        self.covering_algo_name = covering_algo_name
        self.n_clusters = n_clusters
        self.use_algo_vecs = '__use_algo_vecs' in covering_algo_name
        self.model_hf_name = model_hf_name
        self.sim_func = sim_func
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.data_portion = data_portion
        self.filter_in_qids = filter_in_qids
        self.batch_size = batch_size

        # Define cache dirs:
        self.cache_dir = f"data/cached_clustering/{self.dataset_name}_{self.model_hf_name.split('/')[-1]}_{self.sim_func}"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_eval_dir = f"{self.cache_dir}/cover_eval"
        os.makedirs(self.cache_eval_dir, exist_ok=True)
        self.cache_run_id_str = f"{self.covering_algo_name}={self.n_clusters}-{self.data_split}-{self.data_portion}"
        if filter_in_qids is not None:
            self.cache_run_id_str += f"--{filter_in_qids_name}"
        self.train_cache_files = dict(
            qid_to_cluster_idx=f"{self.cache_dir}/{self.cache_run_id_str}__qid2cl.json",
            cluster_idx_to_qids=f"{self.cache_dir}/{self.cache_run_id_str}__cl2qid.json",
            centroid_vecs=f"{self.cache_dir}/{self.cache_run_id_str}__centroid_vecs.pt",
        )

        def _eval_cache_files(dataset_name_to_eval: str, data_split_to_eval: str,
                              data_portion_to_eval: float, eval_id: str = None) -> Dict[str, str]:
            # Define cache paths for the evaluation it
            return dict(
                cover_eval_metrics=f"{self.cache_eval_dir}/{self.cache_run_id_str}_"
                                   f"on_{dataset_name_to_eval}-{data_split_to_eval}-{data_portion_to_eval}"
                                   f"__cover_eval{'_'+eval_id if eval_id is not None else ''}.json",
            )

        self.eval_cache_files = _eval_cache_files

        # Define the covering-algorithm function
        self.cover_algo_func = covering_algo_name_to_func[self.covering_algo_name]

    def fit_predict(self) -> Tuple[Dict[str, int], List[List[str]], torch.Tensor]:
        """Fits the cover on the training queries and returns the clustering results."""
        # 1. If the cover-training was not done yet, do it:
        if not self._check_if_train_cache_files_exist():
            self._train_cover_on_queries()

        # 2. Load cover from cache:
        logger.info(f"Loading cover from cache: {self.train_cache_files}")
        qid_to_cluster_idx: Dict[str, int] = json.load(open(self.train_cache_files['qid_to_cluster_idx']))
        cluster_idx_to_qids: List[List[str]] = json.load(open(self.train_cache_files['cluster_idx_to_qids']))
        centroid_vecs: torch.Tensor = torch.load(self.train_cache_files['centroid_vecs'])  # (n_clusters, emb_dim)

        # 3. Print quick summary on this cover:
        from collections import Counter
        clusters_counter = Counter(qid_to_cluster_idx.values())
        n_clusters = len(clusters_counter.keys())
        logger.info(f"With {n_clusters=}; Queries clustering: {clusters_counter}")

        return qid_to_cluster_idx, cluster_idx_to_qids, centroid_vecs

    def _train_cover_on_queries(self):
        # 1.1. Load model:
        model = RetrieverModel(
            model_hf_name=self.model_hf_name,
            sim_func_name=self.sim_func,
            max_batch_size=self.batch_size,
        )

        # 1.2. Load dataset:
        corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
            dataset_name=self.dataset_name,
            data_split=self.data_split,
            data_portion=self.data_portion,
            embedder_model_name=self.model_hf_name,
            filter_in_qids=self.filter_in_qids,
        )
        # 1.3. Load retrieval results:
        results = load_cached_eval(
            dataset_name=self.dataset_name,
            model_hf_name=self.model_hf_name,
            sim_func_name=self.sim_func,
            data_split=self.data_split,
            data_portion=self.data_portion,
        )
        # 1.4. Embed the queries:
        q_embs = model.embed(qp_pairs_dataset['query'], to_cpu=True).cpu()  # In CPU to avoid OOM
        q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=-1)  # Normalize query embeddings
        qid_to_emb = {qid: q_embs[i] for i, qid in enumerate(qp_pairs_dataset['query_id'])}

        # 2. Save the dataset stats: (to be used by clustering or later by the evaluation)
        ret_stats = (RetrievalDataStats(  # TODO avoid loading this, and potentially implement locally what's needed.
            model=model,
            qp_pairs_dataset=qp_pairs_dataset,
            corpus=corpus, queries=queries, qrels=qrels,
            results=results,
        ))

        # 2. Clustering (either perform&save or load)
        # Maps {qid -> cluster_idx}, assumes the `cluster_idx`s are in `range(n_clusters)`
        algo_result = self.cover_algo_func(
            n_clusters=self.n_clusters,
            q_embs=q_embs,
            q_ids=qp_pairs_dataset['query_id'],
            ret_stats=ret_stats,
        )
        centroid_vecs = None
        if self.use_algo_vecs:
            qid_to_cluster_idx: Dict[str, int] = algo_result[0]
            centroid_vecs = algo_result[1]
        else:
            qid_to_cluster_idx: Dict[str, int] = algo_result
        json.dump(qid_to_cluster_idx, open(self.train_cache_files['qid_to_cluster_idx'], 'w'), indent=2)

        # 2.2. Derive the actual amount of clusters (for algorithm where its dynamic)
        n_clusters = len(set(qid_to_cluster_idx.values()))

        # 3. Mapping each cluster to the list of queries_idx
        cluster_idx_to_qids: List[List[str]] = [[] for _ in range(n_clusters)]
        for qid, cluster_idx in qid_to_cluster_idx.items():
            cluster_idx_to_qids[cluster_idx].append(qid)

        json.dump(cluster_idx_to_qids, open(self.train_cache_files['cluster_idx_to_qids'], 'w'), indent=2)

        # 4. Map each qid to the corresponding index
        query_id_to_query_idx: Dict[int, str] = {query_idx: query_id for query_idx, query_id in
                                                 enumerate(qp_pairs_dataset['query_id'])}

        # 3. Save/load the centroids induced/created by the clustering
        if centroid_vecs is None:
            # List the `q_idx`s per cluster (=`cluster_id`)
            centroid_vecs_lst = []
            for cluster_idx, qids_in_cluster in enumerate(cluster_idx_to_qids):
                q_embs_in_cluster = torch.stack([qid_to_emb[qid] for qid in qids_in_cluster], dim=0)
                centroid_vecs_lst.append(calc_centroid_objective(q_embs_in_cluster, self.sim_func, do_normalize=False))
            centroid_vecs = torch.stack(centroid_vecs_lst, dim=0)  # (n_clusters, emb_dim)
            centroid_vecs = torch.nn.functional.normalize(centroid_vecs, p=2, dim=-1)  # Normalize centroid vectors
            if self.sim_func == 'dot':
                # We artificially add the max-norm to the centroid vectors, to simulate the actual dot-prod optimization
                #   Note: this is relevant only for experiments, to simulate an 'ideal' optimization.
                centroid_vecs *= ret_stats.passage_norms_stats['passage__99q_norm']
        torch.save(centroid_vecs, self.train_cache_files['centroid_vecs'])

    def _check_if_train_cache_files_exist(self):
        for f_path in self.train_cache_files.values():
            if not os.path.exists(f_path):
                return False
        return True

    def evaluate_retrieval(
            self,
            data_split_to_eval: str = "test",
            data_portion_to_eval: float = 1.0,
            dataset_name_to_eval: str = None,  # defaults to the training's dataset_name

            filter_in_qids_to_eval: List[str] = None,  # for concept-targeting
            centroid_real_texts: List[str] = None,
            centroid_real_toks: List[List[int]] = None,
            centroid_vecs: torch.Tensor = None,
            eval_id: str = None,
            skip_existing: bool = True,
    ):
        """Evaluates the cover on the given dataset and data-portion and saves the results.
            The centroids we evaluate wrt can be given as an argument (e.g., crafted adv-passages),
            in text (centroid_adv_texts) or in tokens (centroid_real_toks) or directly in vectors (centroid_vecs);
            if not given, we load them from cache.

            In case `centroid_real_texts` is given, we mention the attack is implementable in the filename.
        """
        # 0.A. Set the `dataset_name` to evaluate on
        dataset_name_to_eval = dataset_name_to_eval or self.dataset_name

        # 0.B. Load results from cache, if exists
        eval_cache_files = self.eval_cache_files(dataset_name_to_eval, data_split_to_eval, data_portion_to_eval,
                                                 eval_id=eval_id)
        print(eval_cache_files)
        if skip_existing and os.path.exists(eval_cache_files['cover_eval_metrics']):
            logger.info(f"Cover evaluation already exists at {eval_cache_files['cover_eval_metrics']}. Skips.")
            with open(eval_cache_files['cover_eval_metrics'], 'r') as f:
                return json.load(f)

        # 1. Load the retrieval-evaluation that was done
        results = load_cached_eval(
            dataset_name=dataset_name_to_eval,
            model_hf_name=self.model_hf_name,
            sim_func_name=self.sim_func,
            data_split=data_split_to_eval,
            data_portion=data_portion_to_eval,
        )
        if results is None:
            return  # no cached results found

        # 2. Load model and data
        model = RetrieverModel(
            model_hf_name=self.model_hf_name,
            sim_func_name=self.sim_func,
            max_batch_size=self.batch_size,
        )
        corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
            dataset_name=dataset_name_to_eval,
            data_split=data_split_to_eval,
            data_portion=data_portion_to_eval,
            embedder_model_name=self.model_hf_name,
            filter_in_qids=filter_in_qids_to_eval,
        )

        # Embed the training queries:
        q_embs = model.embed(qp_pairs_dataset['query']).cuda()
        # Map query-ids to their embeddings
        qid_to_emb = {qid: q_embs[i] for i, qid in enumerate(qp_pairs_dataset['query_id'])}

        # 3. Load the centroid vectors that were found
        if centroid_vecs is not None:
            # if the centroid vectors are given directly, use them
            centroid_vecs = centroid_vecs.cuda()
        elif centroid_real_texts is not None:
            # otherwise, embed the given `centroid_real_texts` and use them as centroids
            centroid_vecs = model.embed(texts=centroid_real_texts).cuda()

        elif centroid_real_toks is not None:
            # otherwise, embed the given `centroid_real_toks` and use them as centroids
            # first - prepare `input_ids` and `attention_mask` from the `centroid_real_toks`, then embed them
            max_length = max(max(len(seq) for seq in centroid_real_toks), model.tokenizer.model_max_length)
            input_ids = torch.full((len(centroid_real_toks), max_length), model.tokenizer.pad_token_id,
                                   dtype=torch.long)
            attention_mask = torch.zeros((len(centroid_real_toks), max_length), dtype=torch.long)
            for i, seq in enumerate(centroid_real_toks):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :len(seq)] = 1
            centroid_vecs = model.embed(inputs=dict(input_ids=input_ids.cuda(),
                                                    attention_mask=attention_mask.cuda())
                                        ).cuda()
        else:
            # otherwise, load the centroid vectors from cache ("perfect" attack)
            centroid_vecs = torch.load(self.train_cache_files['centroid_vecs']).cuda()

        n_clusters = centroid_vecs.shape[0]

        # 4. Evaluate:
        evaluate_metrics_for_values = [10, 20, 50, 100]
        metrics: Dict[str, float] = full_evaluation_with_adv_passage_vecs(
            adv_passage_vecs=centroid_vecs,
            attacked_qrels=qrels,
            results=results,
            qid_to_emb=qid_to_emb,
            sim_func_name=self.sim_func,
            return_for_k_values=evaluate_metrics_for_values,
        )

        cover_eval_metrics = dict(
            config=dict(
                # Algorithm:
                n_clusters=n_clusters,
                covering_algo_name=self.covering_algo_name,

                # Model:
                model_hf_name=self.model_hf_name,
                sim_func=self.sim_func,
                dataset_name=self.dataset_name,

                # Training data (for clustering)
                train_data_split=self.data_split,
                train_data_portion=self.data_portion,

                # Current evaluation data:
                eval_data_split=data_split_to_eval,
                eval_data_portion=data_portion_to_eval,
                eval_dataset_name=dataset_name_to_eval,
            ),
            **metrics
        )
        # 4.1. Save evaluation
        json.dump(cover_eval_metrics, open(eval_cache_files['cover_eval_metrics'], 'w'), indent=2)

        return cover_eval_metrics


def calc_centroid_objective(
        q_embs: torch.Tensor,  # embedding to find centroid of
        sim_func: str,  # 'dot' or 'cos_sim'
        do_normalize: bool = False,
) -> torch.Tensor:
    """Computes the centroid objective for the given queries; was derived from math analysis."""
    if sim_func == 'cos_sim':
        q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=-1)
    q_centroid = q_embs.mean(dim=0)
    if do_normalize:
        q_centroid = torch.nn.functional.normalize(q_centroid, p=2, dim=-1)
    return q_centroid


def get_cover_instance_of_concept(
        model_hf_name: str,
        sim_func: str,
        concept_to_attack: str,
        n_clusters: int,
        dataset_name: str = 'msmarco',
        **kwargs,  # for the cover-algo
):
    """Gets cover instance according to the concept to attack."""
    with open(f"../config/cover_alg/concept-{concept_to_attack}.yaml", "r") as f:  # TODO revert
        import yaml
        concept_config = yaml.safe_load(f)
        concept_qids = concept_config['concept_qids']  # fetched from the attack config
        concept_portion_to_train = concept_config['concept_portion_to_train']
        concept_config['data_split'] = f'train-concepts--{concept_to_attack}-{concept_portion_to_train}'  # as we want to uniquely idenfify the clustering caching
        heldin_concept_qids, heldout_concept_qids = (concept_qids[:int(len(concept_qids) * concept_portion_to_train)],
                                                     concept_qids[int(len(concept_qids) * concept_portion_to_train):])
        concept_config['filter_in_qids_name'] = f"heldin-concept-{concept_config['concept_name']}-{concept_config['concept_portion_to_train']}"
        concept_config['filter_in_qids'] = heldin_concept_qids
        concept_config['n_clusters'] = n_clusters
        concept_config.update(kwargs)

        cover_instance = CoverAlgorithm(
            model_hf_name=model_hf_name,
            sim_func=sim_func,
            dataset_name=dataset_name,
            **concept_config
        )

        return cover_instance, heldin_concept_qids, heldout_concept_qids