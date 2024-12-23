import json
import os.path

import typer

import wandb
from src import data_utils
from src.covering.covering import CoverAlgorithm, get_cover_instance_of_concept
from src.evaluate.data_stats import RetrievalDataStats

from src.evaluate.evaluate_beir_offline import save_beir_eval, load_cached_eval
from src.evaluate.evaluate_beir_online import _evaluate_with_adv_passage
from src.models.retriever import RetrieverModel

app = typer.Typer()


@app.command()
def cache_retrieval_eval(dataset_name: str, model_hf_name: str, sim_func_name: str,
                         data_split: str, batch_size: int = 1024,
                         data_portion: float = 1.0):
    """Caches the retrieval results of the given model on the given dataset."""
    print(f"Evaluating: {dataset_name=}, {model_hf_name=}, {sim_func_name=}, {data_split=}, "
          f"{batch_size=}, {data_portion=}")
    save_beir_eval(
        dataset_name=dataset_name,
        model_hf_name=model_hf_name,
        sim_func_name=sim_func_name,
        data_split=data_split,
        batch_size=batch_size,
        data_portion=data_portion,
    )


@app.command()
def evaluate_benign(
        dataset_name: str, model_hf_name: str, sim_func_name: str,
        data_split: str, batch_size: int = 512, data_portion: float = 1.0,
):
    """Evaluates an existing cached retrieval of test-set, and logs the results to W&B."""
    eval_json_path = f"results/benign/eval-benign-retrieval__{dataset_name}__{model_hf_name.split('/')[-1]}-{sim_func_name}__{data_split}-{data_portion}.json"
    if os.path.exists(eval_json_path):
        print(f"Skipping the evaluation as the results are already cached at `{eval_json_path}`.")
        return
    wandb.init(project="eval-benign-retrieval-final",
               config=dict(
                   dataset_name=dataset_name,
                   model_hf_name=model_hf_name,
                   sim_func_name=sim_func_name,
                   data_split=data_split,
                   data_portion=data_portion,
               ))

    # 1.1. Load model:
    model = RetrieverModel(
        model_hf_name=model_hf_name,
        sim_func_name=sim_func_name,
        max_batch_size=batch_size,
    )

    # 1.2. Load dataset:
    corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
        dataset_name=dataset_name,
        data_split=data_split,
        data_portion=data_portion,
        embedder_model_name=model_hf_name,
    )

    # Initialize the saved metrics:
    metrics = {}

    # Load the existing cached retrieval results:
    results = load_cached_eval(
        dataset_name=dataset_name,
        model_hf_name=model_hf_name,
        sim_func_name=sim_func_name,
        data_split=data_split,
        data_portion=data_portion,
    )
    metrics.update(_evaluate_with_adv_passage(
        results=results,
        pid_advs=[],
        attacked_gold_qrels=qrels,
        # metrics_suffix=metrics_suffix,
        return_gold_metrics=True,
    ))

    # Update with the dataset stats:
    metrics.update(RetrievalDataStats(
        model=model,
        qp_pairs_dataset=qp_pairs_dataset,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        results=results,
    ).get_summary())

    print(metrics)
    # save to JSON
    with open(eval_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    wandb.log(metrics)
    wandb.finish()


@app.command()
def cache_clustering_and_eval(
        # Model and data:
        dataset_name: str, model_hf_name: str, sim_func_name: str,
        # Training set:
        data_split: str, data_portion: float = 1.0,
        # Eval set:
        dataset_name_to_eval: str = None, data_split_to_eval: str = "test", data_portion_to_eval: float = 1.0,
        # Cover algorithm:
        n_clusters: int = 1000, covering_algo_name="kmeans",
        # Misc:
        batch_size: int = 2048
):
    cover_algo = CoverAlgorithm(
        dataset_name=dataset_name,
        model_hf_name=model_hf_name,
        sim_func=sim_func_name,

        covering_algo_name=covering_algo_name,
        n_clusters=n_clusters,

        data_split=data_split,
        data_portion=data_portion,

        batch_size=batch_size,
    )
    # Cluster the training queries:
    cover_algo.fit_predict()
    # Evaluate on the training-set (used to cluster):
    cover_algo.evaluate_retrieval(
        data_split_to_eval=data_split,
        data_portion_to_eval=data_portion,
    )
    # Evaluate on the held-out test-set:
    cover_algo.evaluate_retrieval(
        data_split_to_eval=data_split_to_eval,
        data_portion_to_eval=data_portion_to_eval,
        dataset_name_to_eval=dataset_name_to_eval
    )


@app.command()
def cache_clustering_and_eval__targeted_concepts(
        # Model and data:
        dataset_name: str, model_hf_name: str, sim_func_name: str, concept_to_attack: str,
        concept_portion_to_train: float = 0.5,
        # Cover algorithm:
        n_clusters: int = 1000, covering_algo_name="kmeans",
        # Misc:
        batch_size: int = 2048
):
    cover_algo, heldin_concept_qids, heldout_concept_qids = get_cover_instance_of_concept(
        dataset_name=dataset_name,
        model_hf_name=model_hf_name,
        sim_func=sim_func_name,
        concept_to_attack=concept_to_attack,
        n_clusters=n_clusters,

        # additional args
        covering_algo_name=covering_algo_name,
        batch_size=batch_size,
    )
    # Cluster the training queries:
    cover_algo.fit_predict()
    # Evaluate on the training-set (used to cluster):
    cover_algo.evaluate_retrieval(
        data_split_to_eval='train-concepts',
        data_portion_to_eval=1.0,
        dataset_name_to_eval=dataset_name,
        filter_in_qids_to_eval=heldin_concept_qids,
        eval_id="simulated-heldin"
    )
    # Evaluate on the held-out test-set:
    cover_algo.evaluate_retrieval(
        data_split_to_eval='train-concepts',
        data_portion_to_eval=1.0,
        dataset_name_to_eval=dataset_name,
        filter_in_qids_to_eval=heldout_concept_qids,
        eval_id="simulated-heldout"
    )


@app.command()
def cache_clustering_and_eval__targeted_concepts_gen(  # synthetic (lm-generated) queries
        # Model and data:
        dataset_name: str, model_hf_name: str, sim_func_name: str, concept_to_attack: str,
        concept_portion_to_train: float = 0.5,
        # Cover algorithm:
        n_clusters: int = 1000, covering_algo_name="kmeans",
        # Misc:
        batch_size: int = 2048
):
    cover_algo = CoverAlgorithm(
        dataset_name=dataset_name,
        model_hf_name=model_hf_name,
        sim_func=sim_func_name,

        covering_algo_name=covering_algo_name,
        n_clusters=n_clusters,

        data_split='gen_qs',
        data_portion=1.0,
        filter_in_qids=[concept_to_attack],
        filter_in_qids_name=f"gen_qs-{concept_to_attack}",

        batch_size=batch_size,
    )

    # Cluster the training queries:
    cover_algo.fit_predict()

    # Evaluate on the held-out test-set:
    with open(f"config/cover_alg/concept-{concept_to_attack}.yaml", "r") as f:
        import yaml
        concept_config = yaml.safe_load(f)
        concept_qids = concept_config['concept_qids']  # fetched from the attack config

    heldin_concept_qids, heldout_concept_qids = (concept_qids[:int(len(concept_qids)*concept_portion_to_train)],
                                                 concept_qids[int(len(concept_qids)*concept_portion_to_train):])
    data_split = f'train-concepts--{concept_to_attack}-{concept_portion_to_train}'  # as we want to uniquely idenfify the clustering caching
    data_portion = 1.0  # as we fetch from the whole training set
    cover_algo.evaluate_retrieval(
        data_split_to_eval='train-concepts',
        data_portion_to_eval=1.0,
        dataset_name_to_eval=dataset_name,
        filter_in_qids_to_eval=heldout_concept_qids,
        eval_id="simulated-gen_qs-heldout"
    )


if __name__ == "__main__":
    app()
