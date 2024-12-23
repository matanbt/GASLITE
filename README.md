# GASLITEing the Retrieval - Reproduction


This repository contains code for attacking retrieval models via crafting of passages (with _GASLITE_ method) to poison the used retrieval corpus and evaluating these attacks. The repository also allows reproduction of experiments and ablation studies presented in the paper "[TODO]".

![fig1.png](assets/fig1.png)

[//]: # (**_Note:_** This is a _research_ branch, and as such . For building on top of this work, further exploring GASLITE attack, we highly recommend using the slightly refactored and cleaner branch, which is the [**`main` branch of this repo**]&#40;https://github.com/matanbt/attack-retrieval/tree/main&#41;.)
> ℹ️ Currently the repo is meant for _reproduction_ of the attack and evaluation in the original work. **We intend to continue to develop this attack and useful codebase in [this repo](https://github.com/matanbt/attack-retrieval).**


## Demo Notebook
For a quick demonstration of the attack, see the [demo notebook](./demo.ipynb). It showcases the attack on concept-specific queries with a single adversarial passage. [TODO - add link to colab with a cached run; githubtocolab.com ?]

## Setup
The project requires Python `3.8.5` and on and the installation of `pip install -r requirements.txt` (preferably in an isolated `venv`). 

Some logic may require cloning additional repositories into this project dir (see [Related Projects](#related-projects) section).


## Usage
Run the attack script (on of the following) to craft adversarial passage(s). Results will be saved to a JSON file in `./results/`.
   - `attack0_know-all.sh`: attack a single query with _GASLITE_.
   - `attack1_know-what.sh`: attack a specific concept (e.g., all the queries related to Harry Potter queries) with _GASLITE_.
   - `attack2_know-nothing.sh`: attack a whole dataset (e.g., MSMARCO's eval set) with _GASLITE_.

For further modifying the attack parameters, refer to the configuration files in (`./config/`) and use Hydra's CLI [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/).


## Additional Usage
- **Run `cache-retrieval.sh`.** This script caches the similarities of a model on a dataset (to a json, `./data/cached_evals/`). It is a prerequisite for the following steps, to not repeat on this heavy logic per attack evaluation. For models evaluated in the paper this caching was [uploaded to HuggingFace](https://huggingface.co/datasets/MatanBT/retrieval-datasets-similarities/tree/main), so it is not required. Script can be used with any BEIR-supported dataset and SentenceTransformer embedding model.
- **Run `cache-q-partitions.sh`.** This script caches the partition of queries, using a specified method that defaults to _k_-means. Resulted parition is saved into a JSON, `./data/cached_clustering/`. These partitions can be used for exploration, to simulate the _perfect_ attack, or to run a multi-budget attack.
- **Evaluate with `covering.py`'s API.** This module can be used to evaluate retrieval-corpus poisoning attacks (such as _GASLITE_). In particular, the method `evaluate_retrieval(..)`, which given with a set of adversarial passages, evaluates the poisoning attack on a retrieval model with common measures (including visibility of these passages in the top-10 retrieved passages) w.r.t. the targeted queries.


## Related Projects
This project utilizes the following projects:
- [retrieval-datasets-similarities](https://huggingface.co/datasets/MatanBT/retrieval-datasets-similarities/tree/main):
cached similarities of various retrieval datasets and models, downloaded  and used within the project for evaluating the attack (to avoid recalculating these per attack run).
- [GPT2 on BERT's tokenizer](https://github.com/matanbt/nanoGPT/tree/master): a nanoGPT fork and weight that uses BERT's tokenizer (instead of GPT2's); used for crafting fluent adversarial passages.



[//]: # (## Tested models and datasets)
[//]: # (This repository tested to work with MSMARCO and NQ datasets and the following models:)
[//]: # (- Cosine similarity models: `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/e5-base-v2`, `Snowflake/snowflake-arctic-embed-m`, `intfloat/e5-base-v2`, `sentence-transformers/gtr-t5-base`, `sentence-transformers/all-mpnet-base-v2` )
[//]: # (- Dot-product similarity models: `facebook/contriever`, `facebook/contriever-msmarco`, `sentence-transformers/multi-qa-mpnet-base-dot-v1`, `sentence-transformers/msmarco-roberta-base-ance-firstp`)

[//]: # (## Improving this codebase)
[//]: # (- Migrate all BEIR usage to [MTEB retrieval eval]&#40;https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/RetrievalEvaluator.py&#41; &#40;including relying on HF's MTEB's databases, and their caching mechanism&#41;)
[//]: # (- possibly decouple all retrieval-eval logic to a new repo, to allow eval of poisoning attacks.)
[//]: # (-  Discard unused config.)

## Acknowledgements
Some code snippets are loosely inspired by the following codebases:
- [Corpus Poisoning Attack for Dense Retrievers
](https://github.com/princeton-nlp/corpus-poisoning)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [AutoPrompt Attack (Automatic Prompt Construction for Masked Language Models)](https://github.com/ucinlp/autoprompt)
- [GCG Attack (Universal and Transferable Attacks on Aligned Language Models)](https://github.com/llm-attacks/llm-attacks)
- [ARCA Attack (Automatically Auditing Large Language Models via Discrete Optimization)](https://github.com/ejones313/auditing-llms)

## Citation
If you find this work useful, please cite our paper as follows:
```
TODO	 
```
