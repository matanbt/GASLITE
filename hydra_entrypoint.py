import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb


from src.full_attack import attack_ret

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_attack(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # [HACK] To support legacy config:
    cfg.update(**cfg.pop('attack'), **cfg.pop('model'), **cfg.pop('constraints'),
               **cfg.pop('dataset'), **cfg.pop('core_objective'))

    result_path = _get_result_path(cfg)

    wandb_kwargs = {}
    if cfg.get("skip_if_cached", False):
        if os.path.exists(result_path):
            logger.info(f"Skipping the attack as the results are already cached at `{result_path}`.")
            return
    if not cfg.get("log_to_wandb", False):
        wandb_kwargs = dict(mode="disabled")
    os.environ["WANDB__SERVICE_WAIT"] = "300"  # wandb fix
    wandb.init(project="attack-retrieval-prel-research",
               name=_get_exp_name(cfg),
               config=cfg,
               tags=[cfg['exp_tag']],
               **wandb_kwargs)

    # Run attack:
    metrics = attack_ret(**cfg)
    logger.info(metrics)

    wandb.log(metrics)

    # Save the 'results' artifact
    with open(result_path, 'w') as f:
        json.dump({
            **metrics,
            'config': cfg,
        }, f, indent=2)
    logger.info(f"Saved the results to `{result_path}`.")
    # artifact = wandb.Artifact('results', type='results')
    # artifact.add_file(result_path)
    # wandb.log_artifact(artifact)

    wandb.finish()


def _get_exp_name(cfg):
    model_code = {
        'sentence-transformers/all-MiniLM-L6-v2': 'minilm',
        'sentence-transformers/gtr-t5-base': 'gtr',
        'thenlper/gte-base': 'gte',
        'Alibaba-NLP/gte-base-en-v1.5': 'gte-v1.5',
        'intfloat/e5-base-v2': 'e5',
        'sentence-transformers/all-mpnet-base-v2': 'mpnet-cos',
        'Snowflake/snowflake-arctic-embed-m': 'snowflake',
        'dunzhang/stella_en_1.5B_v5': 'stella',

        'facebook/contriever': 'contriever',
        'facebook/contriever-msmarco': 'contriever-ms',
        'sentence-transformers/multi-qa-mpnet-base-dot-v1': 'mpnet-dot',
        'sentence-transformers/msmarco-distilbert-dot-v5': 'distilbert-dot',
        'sentence-transformers/msmarco-roberta-base-ance-firstp': 'ance',
    }[cfg['model_hf_name']]

    # [BAD-CODING] The following is a messy way to encode the experiment name.
    exp_code = "naive"
    if cfg['trigger_loc'] == 'suffix':
        exp_code = "suffix-tox"
    # if cfg['adv_passage_init'] == 'random_toxic_text':
    #     exp_code = "suffix-tox"

    if cfg['flu_alpha'] > 0:
        exp_code += ",flu"
    exp_code += f"-l={cfg['trigger_len']}"

    if cfg['exp_desc'] is not None:
        exp_code += f"-{cfg['exp_desc']}"
    if 'exp0-cap' == cfg['exp_tag']:
        exp_code += f",cap-{cfg['random_seed']}"
    elif cfg['exp_tag'] in ['exp1-cover-concepts', 'exp1-cover-concepts-gen']:
        exp_code += f",{cfg['cover_alg']['concept_name']}-cover_n={cfg['cover_alg']['n_clusters']}"
    elif 'exp2-cover-all' == cfg['exp_tag']:
        exp_code += f",cover_n={cfg['cover_alg']['n_clusters']}"

    return f"{model_code}-[{exp_code}]"


def _get_result_path(cfg):
    query_choice = cfg['query_choice'].split('/')[-1]
    if cfg['cluster_idx'] is not None:
        query_choice += f"___{cfg['cluster_idx']}"
    return os.path.join('results', f"results__{_get_exp_name(cfg)}__{query_choice}.json")


if __name__ == "__main__":
    run_attack()

