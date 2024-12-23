#! /bin/sh

# Attacking a concept, by crafting an adversarial passage with GASLITE, using portion of 'held-in' queries.
# Concepts from the paper: ["potter", "iphone", "vaccine", "sandwich", "mortgage", "boston", "golf", "flower"]
# Explanation: the concept promotable content and query ids from MSMARCO are kept in the `config/cover_alg/*` folder.

DATASET=msmarco-train-concepts
MODEL=sentence-transformers/all-MiniLM-L6-v2
SIM_FUNC=cos_sim

# Target choice:
N_CLUSTERS=10  # budget size (# of adv. passages), and the amount of times the attack is run
CONCEPT="potter"

# Additional config:
BATCH_SIZE=2048

for c_idx in $(seq 0 $((N_CLUSTERS - 1))); do
    python hydra_entrypoint.py --config-name default \
     model.sim_func_name=${SIM_FUNC} "model.model_hf_name=${MODEL}" dataset=${DATASET}  \
     core_objective=covering cover_alg=concept-${CONCEPT} cover_alg.n_clusters=${N_CLUSTERS} core_objective.cluster_idx="${c_idx}"   \
     exp_tag=exp1_knows-what random_seed="${c_idx}" batch_size=${BATCH_SIZE}
done

# NOTE: modify to `cover_alg=concept_gen-${concept}` for using synthetic queries
# NOTE: to make the attack fluent, set `constraints=as-suffix-tox-and-flu`
