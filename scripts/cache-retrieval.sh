#! /bin/sh

# This script caches the retrieval results of the models on the benign data and then evaluates the benign performance.

# Define the dataset
export DATASET=msmarco
export DATA_SPLIT=test
# export DATA_SPLIT=train-concepts  # for the concept attack
export DATA_PORTION=1.0

# Additional run config
export BATCH_SIZE=512
export MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export SIM_FUNC=cos_sim

# Cache retrieval results (saves to `./data/cached_evals/{...}.json`)
python cli_utils.py cache-retrieval-eval ${DATASET} ${MODEL_NAME} ${SIM_FUNC} ${DATA_SPLIT} --batch-size ${BATCH_SIZE} --data-portion ${DATA_PORTION}

# [OPTIONAL] Evaluate the benign performance (saves to `results/benign/{...}.json`)
python cli_utils.py evaluate-benign ${DATASET} ${MODEL_NAME} ${SIM_FUNC} ${DATA_SPLIT} --batch-size ${BATCH_SIZE} --data-portion ${DATA_PORTION}
