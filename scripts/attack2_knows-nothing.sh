#! /bin/sh

# Attacking a whole diverse query set (MSMARCO eval set), by crafting an adversarial passage with GASLITE, using portion of 'held-in' queries.

DATASET=msmarco-test
MODEL=sentence-transformers/all-MiniLM-L6-v2
SIM_FUNC=cos_sim

# Target choice:
N_CLUSTERS=100  # budget size, also the # of crafted adv. passages

# Additional config:
BATCH_SIZE=2048

for c_idx in $(seq 0 $((N_CLUSTERS - 1))); do
    python hydra_entrypoint.py --config-name default \
    model.sim_func_name=${SIM_FUNC} "model.model_hf_name=${MODEL}" dataset=${DATASET} \
    core_objective=covering cover_alg.n_clusters=${N_CLUSTERS} core_objective.cluster_idx="${c_idx}" \
    exp_tag=exp2_knows-nothing random_seed="${c_idx}" constraints=as-suffix-tox batch_size=${BATCH_SIZE}
done

