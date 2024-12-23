import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


class RetrievalDataStats:
    def __init__(
            self,
            model,
            qp_pairs_dataset,
            corpus, queries, qrels,
            results: dict,

            qid_subset: List[str] = None,  # if not None, will only consider the given subset of queries

    ):
        self.model = model
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.results = results
        self.qp_pairs_dataset = qp_pairs_dataset
        self.sim_func = model.sim_func
        self.qid_subset = qid_subset if qid_subset is not None else list(qp_pairs_dataset['query_id'])

    @property
    def tokens_stats(self):
        """Extracts some useful stats from the dataset"""

        # 2.1. Initialize metrics with the dataset stats:
        tokenized_qs = self.model.tokenizer(self.qp_pairs_dataset['query'], return_tensors="pt", padding=True,
                                            truncation=True)
        tokenized_ps = self.model.tokenizer(self.qp_pairs_dataset['passage'], return_tensors="pt", padding=True,
                                            truncation=True)

        return {
            'tot_queries_count': len(self.queries),
            'queries__avg_tokens': tokenized_qs['attention_mask'].sum(dim=1).float().mean().item(),
            'queries__std_tokens': tokenized_qs['attention_mask'].sum(dim=1).float().std().item(),
            'queries__lst_tokens': tokenized_qs['attention_mask'].sum(dim=-1).float().tolist(),

            'tot_corpus_count': len(self.corpus),
            'passages__avg_tokens': tokenized_ps['attention_mask'].sum(dim=1).float().mean().item(),
            'passages__std_tokens': tokenized_ps['attention_mask'].sum(dim=1).float().std().item(),
            'passages__lst_tokens': tokenized_ps['attention_mask'].sum(dim=-1).float().tolist(),
        }

    @property
    def passage_norms_stats(self, n_evals=25_000):
        shuffled_passages = [content['text'] for pid, content in self.corpus.items()]
        random.shuffle(shuffled_passages)
        passages_to_eval_with = self.model.embed(shuffled_passages[:n_evals])
        norms = torch.norm(passages_to_eval_with, dim=-1)

        return {
            'passage__avg_norm': norms.mean().item(),
            'passage__max_norm': norms.max().item(),
            'passage__75q_norm': norms.quantile(0.75).item(),
            'passage__90q_norm': norms.quantile(0.90).item(),
            'passage__95q_norm': norms.quantile(0.95).item(),
            'passage__99q_norm': norms.quantile(0.99).item(),
            'passage__std_norm': norms.std().item(),
            'passage__lst_norms': norms.tolist(),
        }

    @property
    def sims_list_q_to_kth_similar_p(self, k=9) -> list:
        assert self.results is not None

        the_kth_place_sim_lst = []
        for qid in self.qid_subset:
            the_kth_place_sim = sorted(self.results[qid].values())[k]
            the_kth_place_sim_lst.append(the_kth_place_sim)
        return the_kth_place_sim_lst

    @property
    def sims_list_q_to_gold(self) -> list:
        assert self.results is not None

        sim_q_to_gold_lst = []

        for qid in self.qid_subset:
            p_gold_id = self.qrels[qid]
            p_gold_id = list(p_gold_id.keys())[0]
            gold_sim = 0
            if p_gold_id in self.results[qid]:
                gold_sim = self.results[qid][p_gold_id]
            sim_q_to_gold_lst.append(gold_sim)

        return sim_q_to_gold_lst

    def avg_pairwise_sim_x_to_y(self, x='passage', y='passage', n_evals=1500, batch_size=100):
        """Calculate the average similarity (`sim_func`) of a random query to a random passage."""
        assert x in ['query', 'passage'] and y in ['query', 'passage']
        x_texts = self.qp_pairs_dataset[x].copy()[:n_evals]
        y_texts = self.qp_pairs_dataset[y].copy()[-n_evals:]
        n_evals = min(n_evals, len(x_texts), len(y_texts))
        lst_sim_to_rand = []

        for _ in range(0, n_evals, batch_size):
            x_batch = self.model.embed(random.choices(x_texts, k=batch_size))
            y_batch = self.model.embed(random.choices(y_texts, k=batch_size))
            # if self.sim_func == 'cos_sim':  # then normalize before dot product  [WE CURRENTLY EXAMINE COS-SIM FOR ALL]
            x_batch = F.normalize(x_batch, p=2, dim=-1)
            y_batch = F.normalize(y_batch, p=2, dim=-1)
            curr_sim = torch.matmul(x_batch, y_batch.T)  # calculate the (pairwise) similarity matrix

            # Discard diagonal and flatten
            curr_sim = curr_sim[~torch.eye(curr_sim.shape[0]).bool()].flatten()
            lst_sim_to_rand.extend(curr_sim.tolist())

        return {
            'sim_avg': sum(lst_sim_to_rand) / len(lst_sim_to_rand),
            'sim_lst': lst_sim_to_rand,
        }

    def calc_passage_ppl(self, flu_model_name='our_gpt2', n_evals=25_000, batch_size=256):
        """Calculate the perplexity of a random passages."""
        texts = [content['text'] for pid, content in self.corpus.items()]
        random.shuffle(texts)
        n_evals = min(n_evals, len(texts))
        lst_ppl = []

        if flu_model_name == 'our_gpt2':
            from src.attacks.fluency_scorer import GPT2FluencyScorer
            batch_size = 1  # since nanoGPT has no attention-mask available for inference.
            flu_model = GPT2FluencyScorer()
        else:
            raise NotImplementedError(f"Fluency model `{flu_model_name}` not implemented.")

        for i in range(0, n_evals, batch_size):  # TODO assume batch is done internaly (by calc_fluency_scores)
            batch = texts[i:i + batch_size]
            ppl = flu_model.calc_fluency_scores(None, text_inputs=batch).cpu()
            lst_ppl.extend(ppl.tolist())

        return {
            'ppl_avg': sum(lst_ppl) / len(lst_ppl),
            'ppl_lst': lst_ppl,
        }

    def get_summary(self) -> dict:
        return {
            **self.tokens_stats,
            **self.passage_norms_stats,
            'sims_avg_q_to_kth_similar_p': np.array(self.sims_list_q_to_kth_similar_p).mean(),
            'sims_avg_q_to_gold': np.array(self.sims_list_q_to_gold).mean(),

            'rand_sim_q_to_p': self.avg_pairwise_sim_x_to_y(x='query', y='passage'),
            'rand_sim_p_to_p': self.avg_pairwise_sim_x_to_y(x='passage', y='passage'),
            'rand_sim_q_to_q': self.avg_pairwise_sim_x_to_y(x='query', y='query'),

            'passage_ppl': self.calc_passage_ppl(),
        }
