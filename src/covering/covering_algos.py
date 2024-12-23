# Maps {qid -> cluster_idx}, assumes `cluster_idx` in range(n_clusters)
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from src.evaluate.data_stats import RetrievalDataStats


def cover_with_kmeans(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor
        **kwargs,
) -> Dict[str, int]:  # {qid -> cluster_idx}
    # We normalize before clustering with KMeans
    q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=-1)
    # Compute K-Means clustering on the queries, and save the labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(q_embs.cpu().numpy())
    clustering_labels = {}
    for qid, cluster_idx in zip(q_ids, kmeans.labels_.tolist()):
        clustering_labels[qid] = cluster_idx

    return clustering_labels


def cover_with_our_iterative_scheme_for_budget_eq_one(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,

        ret_final_p_advs: bool = True,
        **kwargs,
) -> Dict[str, int]:  # {qid -> cluster_idx}
    # HACK Note: As an ugly hack we use `n_clusters` as the number of iterations. We always return a single cluster.

    # Find the query with the maximal mean similarity to all other queries
    pairwise_sim = q_embs @ q_embs.T
    mean_sim = pairwise_sim.mean(dim=-1)
    max_sim_idx = mean_sim.argmax().item()

    # Get similarity required to reach top-10, per query  # TODO consider using it
    # epsilons = torch.tensor(ret_stats.sims_list_q_to_kth_similar_p).cpu()

    # Initialize the attack queries set with it
    finished_queries_list = [max_sim_idx]

    # Initialize the result centroids list
    resulted_centroids = []

    for i in range(1, n_clusters):
        curr_centroid = q_embs[finished_queries_list].mean(dim=0)
        # Find the query with the maximal mean similarity to all other queries
        sim_to_centroid = q_embs @ curr_centroid
        # What is the query nearest the centroid, that is not in `finished_queries_list`
        mask = torch.ones(len(q_ids))
        mask[finished_queries_list] = 0
        mask = mask.bool()
        sim_to_centroid[~mask] = -1
        max_sim_idx = sim_to_centroid.argmax().item()
        finished_queries_list.append(max_sim_idx)

        # Save the centroid
        resulted_centroids.append(curr_centroid)

    # Save the final centroids; each is an option for a single cluster
    final_centroids = torch.stack(resulted_centroids)

    qid_to_cluster_idx = {qid: 0 for qid in q_ids}

    if ret_final_p_advs:
        return qid_to_cluster_idx, final_centroids[-1:]
    return qid_to_cluster_idx


def greedy_set_cover(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        epsilons_from: str = 'q_to_kth_similar_p',  # 'q_to_kth_similar_p' | 'q_to_gold'
        possible_p_advs_names: Tuple[str]=('queries',),  # 'queries', 'pairwise_centroids'
        ret_final_p_advs: bool = False,
        fixed_sim_for_epsilons: float = 0.92,
        **kwargs,
) -> Union[Dict[str, int], Tuple[Dict[str, int], torch.Tensor]]:  # {qid -> cluster_idx}, final_p_advs
    # Algorithm main parameters:
    possible_p_advs = []  # possible candidates to the algorithm
    if 'queries' in possible_p_advs_names:
        possible_p_advs.append(q_embs)  # we initialize the possible adv-passages to be the queries themselves
    if 'pairwise_centroids' in possible_p_advs_names:
        pairwise_centroids = torch.stack([q_embs[i] + q_embs[j]  # all pairwise centroids to the possible_p_advs
                                            for i in range(len(q_embs)) for j in range(i + 1, len(q_embs))])
        pairwise_centroids = torch.nn.functional.normalize(pairwise_centroids, p=2, dim=-1)
        possible_p_advs.append(pairwise_centroids)
    if 'triplewise_centroids' in possible_p_advs_names:
        triplewise_centroids = torch.stack([q_embs[i] + q_embs[j] + q_embs[k]  # all triplewise centroids to the possible_p_advs
                                            for i in range(len(q_embs)) for j in range(i + 1, len(q_embs)) for k in range(j + 1, len(q_embs))])
        triplewise_centroids = torch.nn.functional.normalize(triplewise_centroids, p=2, dim=-1)
        possible_p_advs.append(triplewise_centroids)
    possible_p_advs = torch.cat(possible_p_advs)

    if epsilons_from == 'q_to_gold':
        epsilons = torch.tensor(ret_stats.sims_list_q_to_gold).cpu()
    elif epsilons_from == 'q_to_gold*0.9':
        epsilons = torch.tensor(ret_stats.sims_list_q_to_gold).cpu() * 0.9
    elif epsilons_from == 'q_to_gold_avg':
        epsilons = (torch.ones(len(q_ids)).cpu() *
                    (sum(ret_stats.sims_list_q_to_gold) / len(ret_stats.sims_list_q_to_gold)))
    elif epsilons_from == 'q_to_kth_similar_p':
        epsilons = torch.tensor(ret_stats.sims_list_q_to_kth_similar_p).cpu()
    elif epsilons_from == 'fixed_sim':
        epsilons = torch.ones(len(q_ids)).cpu() * fixed_sim_for_epsilons
    else:
        raise ValueError(f"Unknown `epsilons_from` value: {epsilons_from}")

    assert q_embs.shape[0] == epsilons.shape[0], f"embedding should correspond to the epsilons, got {q_embs.shape[0]} != {epsilons.shape[0]}"

    # Algorithm variables:
    sims = possible_p_advs @ q_embs.T  # Assumes dot (or vectors were normalized for cos)
    q_idx_to_cluster_idx = np.zeros(len(q_ids), dtype=int)  # 0 is the default cluster index
    final_p_advs = []

    for i in range(0, n_clusters):  # fill each cluster `i` greedily
        # successfully 'attacked' queries mask
        can_cover_queries_mask = (sims > epsilons)
        # get the currently-best p_adv, i.e., the one that covers the most queries
        best_p_adv_idx = can_cover_queries_mask.sum(dim=-1).argmax()
        covered_queries_indices = torch.where(can_cover_queries_mask[best_p_adv_idx] == 1)[0]
        # zero the similarities of the covered queries (so we won't choose them again)
        sims[:, covered_queries_indices] = 0

        # record the cluster
        q_idx_to_cluster_idx[covered_queries_indices.cpu()] = i
        final_p_advs.append(possible_p_advs[best_p_adv_idx].unsqueeze(0))
        # total_covered_qs.extend(covered_queries_indices.tolist())
        # print(f"[{i}] | So far {len(total_covered_qs)} | Covered {len(covered_queries_indices)} queries with the #{best_p_adv_idx} p_adv")

    q_idx_to_cluster_idx = q_idx_to_cluster_idx.tolist()
    clusters_lst = list(set(q_idx_to_cluster_idx))
    qid_to_cluster_idx = {qid: clusters_lst.index(cluster_idx) for qid, cluster_idx in zip(q_ids, q_idx_to_cluster_idx)}
    final_p_advs = torch.cat(final_p_advs, dim=0)

    if ret_final_p_advs:
        return qid_to_cluster_idx, final_p_advs
    return qid_to_cluster_idx


def kmeans_after_outliers(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor
        contamination: float = 0.05,
        **kwargs,
) -> Dict[str, int]:
    # Remove outliers from data using 'LOF'
    lof = LocalOutlierFactor(n_neighbors=20, metric='cosine', contamination=contamination)
    inliers_mask = lof.fit_predict(q_embs.cpu().numpy()) == 1
    # logger.info(f"Outliers rate: {(~inliers_mask).sum() / len(q_embs) * 100 :.3f}%")
    if inliers_mask.sum() < n_clusters:  # if the inliers are less than the clusters, we discard the outlier removal
        inliers_mask = np.ones_like(inliers_mask)

    # Run KMeans without the outliers
    kmeans = KMeans(n_clusters=n_clusters - 1, random_state=0)
    kmeans.fit(q_embs[inliers_mask].cpu().numpy())

    # Set all the outliers to the last cluster, and the rest to the KMeans clusters
    labels = np.full(len(q_embs), fill_value=n_clusters - 1)
    labels[inliers_mask] = kmeans.labels_

    clustering_labels = {}
    for qid, cluster_idx in zip(q_ids, labels.tolist()):
        clustering_labels[qid] = cluster_idx

    return clustering_labels


def dbscan(
        n_clusters: int,  # not used here
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor
        **kwargs,
) -> Dict[str, int]:
    eps_to_loss = {}
    eps_to_q_idx_to_cluster_idx = {}

    # Algorithm parameters:
    eps_grid = np.linspace(1e-3, 1.3, 100)
    min_samples = 2
    dis_metric = 'cosine'

    # Grid search for the best epsilon
    for i, eps in enumerate(eps_grid):
        db = DBSCAN(
            eps=eps,  # max distance in a neighborhood
            min_samples=min_samples,  # minimum sample in a neighborhood (the higher, the denser the clusters)
            metric=dis_metric
        ).fit(q_embs.cpu().numpy())

        q_idx_to_cluster_idx = db.labels_
        if min(q_idx_to_cluster_idx) == -1:  # fix the clusters to be in range(0, curr_n_clusters)
            q_idx_to_cluster_idx += 1

        q_idx_to_cluster_idx = q_idx_to_cluster_idx.tolist()
        curr_n_clusters = len(set(q_idx_to_cluster_idx))
        eps_to_q_idx_to_cluster_idx[eps] = q_idx_to_cluster_idx

        # We found the number of clusters ot be a good proxy for a good epsilon (higher - better)
        eps_to_loss[eps] = -curr_n_clusters

    # Get the best epsilon
    best_eps = min(eps_to_loss, key=eps_to_loss.get)
    q_idx_to_cluster_idx = eps_to_q_idx_to_cluster_idx[best_eps]

    # set the maximum number of clusters to be `n_clusters` (by putting the rest in the first cluster)
    q_idx_to_cluster_idx = [(c_idx if c_idx < n_clusters else 0)
                            for c_idx in q_idx_to_cluster_idx]

    qids_to_cluster_idx = {qid: cluster_idx for qid, cluster_idx in zip(q_ids, q_idx_to_cluster_idx)}
    return qids_to_cluster_idx


# Hack for overloading # TODO REMOVE I AM UGLY
def greedy_set_cover_gold(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        **kwargs,
):
    return greedy_set_cover(
        n_clusters=n_clusters,
        q_embs=q_embs,
        q_ids=q_ids,
        ret_stats=ret_stats,
        epsilons_from='q_to_gold',
        **kwargs,
    )


def greedy_set_cover_gold_avg(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        **kwargs,
):
    return greedy_set_cover(
        n_clusters=n_clusters,
        q_embs=q_embs,
        q_ids=q_ids,
        ret_stats=ret_stats,
        epsilons_from='q_to_gold_avg',
        **kwargs,
    )


def greedy_set_cover_gold_09(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        **kwargs,
):
    return greedy_set_cover(
        n_clusters=n_clusters,
        q_embs=q_embs,
        q_ids=q_ids,
        ret_stats=ret_stats,
        epsilons_from='q_to_gold*0.9',
        **kwargs,
    )


def greedy_set_cover_kth_similar_p(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        **kwargs,
):
    return greedy_set_cover(
        n_clusters=n_clusters,
        q_embs=q_embs,
        q_ids=q_ids,
        ret_stats=ret_stats,
        epsilons_from='q_to_kth_similar_p',
        **kwargs,
    )


def greedy_set_cover_fixed_eps(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor

        ret_stats: RetrievalDataStats,
        fixed_sim_for_epsilons: float = 0.92,
        **kwargs,
):
    return greedy_set_cover(
        n_clusters=n_clusters,
        q_embs=q_embs,
        q_ids=q_ids,
        ret_stats=ret_stats,
        epsilons_from='fixed_sim',
        fixed_sim_for_epsilons=fixed_sim_for_epsilons,
        **kwargs,
    )


def kmedoids(
        n_clusters: int,
        q_embs: torch.Tensor,
        q_ids: List[str],  # corresponds to the given tensor
        ret_final_p_advs: bool = False,
        **kwargs,
):
    # https://github.com/kno10/python-kmedoids
    import kmedoids
    from sklearn.metrics.pairwise import cosine_distances

    diss = cosine_distances(q_embs)
    fp = kmedoids.fasterpam(diss, n_clusters)
    fp_labels = fp.labels.tolist()
    qids_tp_cluster_idx = {qid: cluster_idx for qid, cluster_idx in zip(q_ids, fp_labels)}

    if ret_final_p_advs:
        return qids_tp_cluster_idx, q_embs[fp.medoids]
    return qids_tp_cluster_idx


covering_algo_name_to_func = {
    'kmeans': cover_with_kmeans,
    'kmeans_after_outliers': kmeans_after_outliers,

    # Greedy Set Cover Variants:
    'greedy_set_cover_gold': greedy_set_cover_gold,
    'greedy_set_cover_gold_avg': greedy_set_cover_gold_avg,
    'greedy_set_cover_kth_similar_p': greedy_set_cover_kth_similar_p,
    'greedy_set_cover_gold_09': greedy_set_cover_gold_09,
    'greedy_set_cover_fixed_092': greedy_set_cover_fixed_eps,

    # Greedy Set Cover Variants while using their "centers"
    'greedy_set_cover_gold__use_algo_vecs': partial(greedy_set_cover_gold, ret_final_p_advs=True),
    'greedy_set_cover_kth_similar_p__use_algo_vecs': partial(greedy_set_cover_kth_similar_p, ret_final_p_advs=True),
    'greedy_set_cover_gold_09__use_algo_vecs': partial(greedy_set_cover_gold_09, ret_final_p_advs=True),
    'greedy_set_cover_fixed_092__use_algo_vecs': partial(greedy_set_cover_fixed_eps, ret_final_p_advs=True),

    # Other clustering algos:
    'dbscan': dbscan,
    'kmedoids': kmedoids,
    'our_iterative_1_budget__use_algo_vecs': partial(cover_with_our_iterative_scheme_for_budget_eq_one, ret_final_p_advs=True),
}
