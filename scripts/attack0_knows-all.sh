#! /bin/sh

# Attacking a single query, by crafting an adversarial passage with GASLITE
# (in paper this is repeated for 50 random queries, using different seeds)

DATASET=msmarco-test
MODEL=sentence-transformers/all-MiniLM-L6-v2
SIM_FUNC=cos_sim
RANDOM_SEED=0  # determines the sampled query
BATCH_SIZE=2048

# Run the attack
#   (set model and dataset -> set attack parameters (to attack a single query) -> additional config)
python hydra_entrypoint.py --config-name default  \
  model.sim_func_name=${SIM_FUNC} "model.model_hf_name=${MODEL}" dataset=${DATASET}  \
  core_objective=single-query batch_size=${BATCH_SIZE} \
  "random_seed=${RANDOM_SEED}" exp_tag=exp0_knows-all


