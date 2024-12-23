import json
from typing import Dict, List, Union, Any

import sentence_transformers
from sentence_transformers import SentenceTransformer
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
import torch
import os


class RetrieverModel:
    def __init__(self,
                 device='cuda',
                 max_batch_size: int = 128,
                 sim_func_name: str = 'dot',
                 adv_loss_name: str = 'sum_of_sim',
                 model_hf_name: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1',  # 'sentence-transformers/all-MiniLM-L6-v2'
                 ):
        # Note this wrapper avoids normalizing the embeddings by default.

        self.model_hf_name = model_hf_name  # saved as metadata
        self.sim_func_name = sim_func_name

        # Hack to allow code execution, flag for loading model [TODO REMOVE ME I'M UGLY]
        _trust_remote_code = False
        if model_hf_name in ["nomic-ai/nomic-embed-text-v1", 'Alibaba-NLP/gte-base-en-v1.5', 'dunzhang/stella_en_1.5B_v5']:
            _trust_remote_code = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_name, trust_remote_code=_trust_remote_code)
        self.tokenizer.model_max_length = 512
        self.tokenizer.padding_side = 'right'  # to align with the passage init scheme  # [TODO remove when init passage will not include padding]

        self.model = SentenceTransformer(model_hf_name, trust_remote_code=_trust_remote_code)
        self.model_encoder: sentence_transformers.models.Transformer = self.model[0].auto_model  # TODO generalize
        self.model_pooler: sentence_transformers.models.Pooling = self.model[1]  # TODO generalize
        self.model_final_layers: List[Any] = list(self.model)[2:]

        assert isinstance(self.model[0], sentence_transformers.models.Transformer)
        assert isinstance(self.model_pooler, sentence_transformers.models.Pooling)

        if model_hf_name == "sentence-transformers/gtr-t5-base":  # HACK to generalize to T5 tokenizer
            self.tokenizer.sep_token_id = self.tokenizer.eos_token_id
            self.tokenizer.sep_token = self.tokenizer.eos_token

        self.model = self.model.to(device)
        self.device = device
        self.max_batch_size = max_batch_size
        self.model.eval()  # since the model is used for inference only
        for param in self.model.parameters():  # prevent the model from updating the gradients
            param.requires_grad = False

        # Define the objective to optimize on
        def _dot_prod(x, y):
            # calculate dot-prod similarities
            return torch.bmm(x.unsqueeze(dim=1), y.unsqueeze(dim=-1)).squeeze([-2, -1])
        def _cos_sim(x, y):
            # calculate cos-sim similarities
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
            y = torch.nn.functional.normalize(y, p=2, dim=-1)
            return torch.bmm(x.unsqueeze(dim=1), y.unsqueeze(dim=-1)).squeeze([-2, -1])
        def _l2_sum(x, y):
            # calculate L2 distance
            return -1 * torch.norm(x - y, p=2, dim=-1)
        self.sim_func = {
            'dot': _dot_prod,
            'cos_sim': _cos_sim,
            'l2_sum': _l2_sum,
        }[sim_func_name]

        # Define the adversary loss
        def _sum_of_sim(x, y, *args):
            return self.sim_func(x, y)  # mean of similarities
        def _sum_of_l2_dist(x, y, *args):
            return -1 * torch.norm(x - y, p=2, dim=1)
        def _sum_of_high_margin(x, targets, anchors):
            # sum of high-margin similarities
            assert anchors is not None, "Must provide `emb_anchors` for `sum_of_high_margin` loss."
            sim_x_to_anchors = self.sim_func(x, anchors)
            sim_x_to_target = self.sim_func(x, targets)
            eps = 0.05  # TODO configurable and grid-search
            return -1 * torch.nn.functional.relu(sim_x_to_anchors - sim_x_to_target + eps)  # `-1` since adversary would like to minimize this loss
        def _l2_norm_inc(x, y, *args):
            # simply calculate L2 norm of `x`  (to increase it)
            return torch.norm(x, p=2, dim=-1)
        self.adv_loss_func = {
            'sum_of_sim': _sum_of_sim,
            'sum_of_l2_dist': _sum_of_l2_dist,
            'sum_of_high_margin': _sum_of_high_margin,
            'l2_norm_inc': _l2_norm_inc,
        }[adv_loss_name]

    def embed(self, texts: List[str] = None, inputs: dict = None, to_cpu: bool = False):
        """Evaluates the accuracy of the model on the given data."""
        assert texts is not None or inputs is not None, "Either `texts` or `inputs` must be provided."

        # Tokenize if needed
        if inputs is None:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.device)

        # Calculate embeddings in batches
        n_samples = inputs['input_ids'].shape[0]
        n_batches = (n_samples // self.max_batch_size
                     + (1 if n_samples % self.max_batch_size != 0 else 0))

        sentence_embeddings = []
        for i in trange(n_batches, desc="Embedding..."):
            # 1. Extract the batch
            batch_slice = slice(i * self.max_batch_size, (i + 1) * self.max_batch_size)
            batch_inputs = {k: inputs[k][batch_slice] for k in inputs.keys()}
            with torch.no_grad():
                # Perform encoding
                model_output = self.model_encoder(**batch_inputs)
                batch_inputs['token_embeddings'] = model_output[0]  # add the model's token embedding to the `inputs`

                # Perform pooling
                sentence_embedding = self.model_pooler(batch_inputs)

                # Apply final layers (if any)
                for module in self.model_final_layers:
                    sentence_embedding = module(sentence_embedding)

                # Normalize embeddings [DISABLED]  # we keep the embedding normalized to calculate dot-prod as similarity.
                # if self.normalize_embeddings:
                #     sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

                sentence_embedding = sentence_embedding['sentence_embedding']  # extract the sentence embedding
                sentence_embeddings.append(sentence_embedding.cpu() if to_cpu else sentence_embedding)

        sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        return sentence_embeddings

    def calc_loss_for_grad(self, inputs_embeds, inputs_attention_mask, emb_targets, emb_anchors=None):
        """Calculates the loss (a value) wrp to the given inputs, to maximize sim to target inputs.
        :param inputs_embeds: input after input-embedding; (n_samples, embed_dim).
        :param inputs_attention_mask: attention mask of the input (n_samples, seq_len).
        :param emb_targets: target (model) embeddings; (n_samples, embed_dim).
        :param emb_anchors: anchors to beat (model) embeddings; (n_samples, embed_dim).
        """
        # Embed the input (while allowing gradients)
        inputs = {
            'inputs_embeds': inputs_embeds,  # one-hot encoded inputs
            'attention_mask': inputs_attention_mask,
        }
        model_output = self.model_encoder(**inputs)
        inputs['token_embeddings'] = model_output[0]  # add the model's token embedding to the `inputs`

        # Perform pooling
        emb_inputs = self.model_pooler(inputs)

        # Apply final layers (if any)
        for module in self.model_final_layers:
            emb_inputs = module(emb_inputs)

        emb_inputs = emb_inputs['sentence_embedding']  # extract the sentence embedding

        # Calculate the loss
        adv_loss = self.adv_loss_func(emb_inputs, emb_targets, emb_anchors).mean()

        return adv_loss

    def calc_loss_for_eval(
            self,
            inputs: Dict[str, torch.Tensor],  # input ids (n_samples, seq_len)
            emb_targets: torch.Tensor,  # queries to reach to (n_samples, emb_dim)
            emb_anchors: torch.Tensor = None,  # anchors to beat (n_samples, emb_dim)
    ):
        emb_inputs = self.embed(inputs=inputs)
        n_samples = inputs['input_ids'].shape[0]
        _max_batch_size = n_samples   #  self.max_batch_size [BATCH IS CURRENTLY DISSABLED]
        n_batches = (n_samples // _max_batch_size
                     + (1 if n_samples % _max_batch_size != 0 else 0))

        losses = []
        for i in trange(n_batches, desc="Calculating loss..."):
            # 1. Extract the batch
            batch_slice = slice(i * _max_batch_size, (i + 1) * _max_batch_size)
            batch_emb_inputs = emb_inputs[batch_slice]
            batch_emb_targets = emb_targets[batch_slice]
            batch_emb_anchors = emb_anchors[batch_slice] if emb_anchors is not None else None

            # 3. Calculate loss over batch
            with torch.no_grad():
                batch_losses = self.adv_loss_func(batch_emb_inputs, batch_emb_targets, batch_emb_anchors)

            losses.append(batch_losses)

        losses = torch.cat(losses, dim=0)
        return losses

    def get_input_embeddings(self):
        if self.model_hf_name == 'nomic-ai/nomic-embed-text-v1':
            return self.model_encoder.embeddings.word_embeddings
        return self.model_encoder.get_input_embeddings()

    def get_special_token_ids(self) -> List[int]:
        # Get special token ids
        special_tokens = [getattr(self.model.tokenizer, attr) for attr in
                          ['cls_token', 'sep_token', 'mask_token', 'pad_token', 'unk_token', 'eos_token', 'bos_token']]
        special_token_ids = [self.model.tokenizer.convert_tokens_to_ids(token)
                             for token in special_tokens if token is not None]

        # Get non-ascii token ids
        def is_ascii(s):
            return s.isascii() and s.isprintable()
        non_ascii_toks = []
        for i in range(self.model.tokenizer.vocab_size):
            if not is_ascii(self.model.tokenizer.decode([i])):
                non_ascii_toks.append(i)

        return special_token_ids + non_ascii_toks