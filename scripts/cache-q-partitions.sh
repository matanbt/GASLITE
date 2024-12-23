#! /bin/sh

# Partitioning a query set, which can used for the attack, or to simply explore it for simulating the attack.
# - Partitioning methods in paper: kmeans kmeans_after_outliers kmedoids greedy_set_cover_kth_similar_p greedy_set_cover_gold greedy_set_cover_gold_avg greedy_set_cover_gold_09 dbscan
# - The chosen method - k-means.

# Default method:
ALGO=kmeans

# Dataset to partition
DATASET=msmarco
# Dataset to evaluate the partitioning on (simulated attack)
DATASET_EVAL=msmarco

# Config:
BATCH_SIZE=512
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
SIM_FUNC=cos_sim


### >>> Knows-What (targeting a concept)
CMD=cache-clustering-and-eval--targeted-concepts
DATA_SPLIT=train-concepts
DATA_PORTION=1.0
CONCEPT_TRAIN_PORTION=0.5
CONCEPT="potter"  # can be: potter, iphone, vaccine, sandwich, mortgage, boston, golf, flower

for n_clusters in 1 5 10 15 20 25 30 35 40 45 50; do
    python cli_utils.py ${CMD} ${DATASET} ${MODEL_NAME} ${SIM_FUNC} \
     ${CONCEPT} --concept-portion-to-train ${CONCEPT_TRAIN_PORTION} \
        --n-clusters ${n_clusters} --covering-algo-name ${ALGO} --batch-size ${BATCH_SIZE}
done


### >>> Knows-All (targeting a whole dataset)
CMD=cache-clustering-and-eval
DATA_SPLIT=train
DATA_PORTION=0.05

for n_clusters in 1 5 10 25 50 75 100 150 200 300 400 500 600 700 750 1000; do
    python cli_utils.py ${CMD} ${DATASET} ${MODEL_NAME} ${SIM_FUNC} ${DATA_SPLIT}  \
    --data-portion ${DATA_PORTION} --n-clusters ${n_clusters} --covering-algo-name ${ALGO} \
    --dataset-name-to-eval ${DATASET_EVAL} --batch-size ${BATCH_SIZE}
done


### >>> Knows-What on synthetic (lm-generated) queries
#       Cache retrieval clustering that was trained on synthetic (LM-Gen) concepts
CMD=cache-clustering-and-eval--targeted-concepts-gen
DATA_SPLIT=train-concepts
CONCEPT_TRAIN_PORTION=0.5
CONCEPT="potter"

for n_clusters in 1 5 10 15 20 25 30 35 40 45 50; do  #  60 70 80 90 100
  python cli_utils.py ${CMD} ${DATASET} ${MODEL_NAME} ${SIM_FUNC} ${CONCEPT} --concept-portion-to-train ${CONCEPT_TRAIN_PORTION} \
      --n-clusters ${n_clusters} --covering-algo-name ${ALGO} --batch-size ${BATCH_SIZE}
done


