from huggingface_hub import hf_hub_download
from tqdm import trange
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import torch

import logging
logger = logging.getLogger(__name__)

# NOTE: scorer should have THE SAME tokenizer as the model's!


class BertMLMFluencyScorer:
    def __init__(self, batch_size: int = 128, **kwargs):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.model = DistilBertForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased").cuda()
        self.input_embedding = self.model.distilbert.embeddings.word_embeddings
        self.batch_size = batch_size

        self.model.eval()
        for param in self.model.parameters():  # prevent the model from updating the gradients
            param.requires_grad = False

    # Fluency constraint (applied on the triggers together, suitable for a repeated trigger)
    def calc_score_for_grad(self, one_hot: torch.tensor, trigger_slice: slice, inputs: dict):
        """returns a backward-able tensor that represents the fluency score (a single number) of the trigger"""
        trigger_len = trigger_slice.stop - trigger_slice.start

        # Get the embeddings of the inputs
        trigger_embeds = (one_hot @ self.input_embedding.weight)  # (batch_size, targeted_sub_seq_len, embed_dim)
        embeds = self.input_embedding(inputs['input_ids']).detach()  # (batch_size, seq_len, embed_dim)

        # Get the mask token embedding
        mask_token_emb = self.input_embedding(
            torch.tensor([self.tokenizer.mask_token_id] * one_hot.shape[0], device='cuda')
        ).unsqueeze(1)

        # Replace each token of the trigger with [mask] token. and aggregate to the fluency score
        fluency_score = 0
        for i in trange(0, trigger_len, desc="Calc grad for fluency"):  # TODO VECTORIZE
            orig_tokens = inputs['input_ids'][:, trigger_slice.start + i]  # tokens that are masked
            masked_embeds = torch.cat(
                [
                    embeds[:, :trigger_slice.start, :],
                    trigger_embeds[:, :i, :],
                    mask_token_emb,  # replace the i-th token with [mask]
                    trigger_embeds[:, (i + 1):, :],
                    embeds[:, trigger_slice.stop:, :]
                ],
                dim=1)
            logits = self.model(inputs_embeds=masked_embeds, attention_mask=inputs['attention_mask']).logits
            logits = logits[:, trigger_slice, :].log_softmax(dim=-1)
            # the fluency score take the average of the log-probabilities of the original tokens
            fluency_score += logits[torch.arange(logits.shape[0]), i, orig_tokens].mean()

        fluency_score /= trigger_len  # average over the number of evaluated tokens
        return fluency_score

    def _calc_fluency_matrix(self, inputs: dict, trigger_slice: slice):
        """
        Calculate the fluency scores of the input tokens, returns a score for each token (had it been chosen for the flip)
        """
        # TODO GENERALIZE: note that currently this assumes a _single_ passage was given as the input (i.e., input_ids is a repeated tensor)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        trigger_len = trigger_slice.stop - trigger_slice.start
        orig_trigger_tokens = input_ids[0, trigger_slice]

        # Create a tensor of mask tokens with the same shape as input_ids
        input_ids_masked = input_ids[0].repeat(trigger_len, 1)
        attention_mask_masked = attention_mask[0].repeat(trigger_len, 1)
        input_ids_masked[
            np.arange(trigger_len), np.arange(trigger_slice.start, trigger_slice.stop)
        ] = self.tokenizer.mask_token_id

        # Calculate the logits (batch_size, seq_len, vocab_size)
        with torch.no_grad():
            mask_probs = self.model(input_ids=input_ids_masked, attention_mask=attention_mask_masked).logits
            mask_probs = mask_probs.log_softmax(dim=-1)

            # Select the probabilities of the masked positions
            fluency_scores = mask_probs[np.arange(trigger_len), np.arange(trigger_slice.start, trigger_slice.stop)]

            # Calc sum of original tokens' fluency scores [DISABLED]
            # fluency_orig_sum = fluency_scores[np.arange(trigger_len), orig_trigger_tokens].sum()
            # Add this sum to the fluency scores of the other tokens, as to maintain the definition fluency scores
            # for i in range(trigger_len):  # iterate within the trigger
            #     fluency_orig_curr = fluency_scores[i, orig_trigger_tokens[i]]  # score of original token
            #     fluency_scores[i] += fluency_orig_sum - fluency_orig_curr  # add the sum of the original tokens
            #     fluency_scores[i, orig_trigger_tokens[i]] = fluency_orig_curr  # restore the original score
            fluency_scores /= trigger_len  # normalize

            # [DISABLED] another option: [Note that is might induce a different _scale_ than the gradient score]
            # Subtract the probability of the current tokens (as we look to maximize the gain of the replacement)
            # fluency_scores -= fluency_scores[np.arange(trigger_len), orig_trigger_tokens].unsqueeze(-1)

        return fluency_scores

    def calc_fluency_scores(self, inputs: dict, trigger_slice: slice):
        """
        Calculate the fluency scores of _each_ input of `inputs` (n_inputs, seq_len)
        returns a tensor with the scores (n_inputs,)
        """
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        n_inputs = input_ids.shape[0]
        trigger_len = trigger_slice.stop - trigger_slice.start

        # 1. Build the masking in batches
        input_ids_masked_lst, attention_mask_masked_lst = [], []

        for i in range(n_inputs):
            input_ids_masked = input_ids[i].repeat(trigger_len, 1)
            attention_mask_masked = attention_mask[i].repeat(trigger_len, 1)
            input_ids_masked[
                np.arange(trigger_len), np.arange(trigger_slice.start, trigger_slice.stop)
            ] = self.tokenizer.mask_token_id

            input_ids_masked_lst.append(input_ids_masked)
            attention_mask_masked_lst.append(attention_mask_masked)

        input_ids_masked_lst = torch.stack(input_ids_masked_lst, dim=0)
        input_ids_masked_lst = input_ids_masked_lst.view(n_inputs * trigger_len, -1)  # (n_inputs*trigger_len, seq_len)
        attention_mask_masked_lst = torch.stack(attention_mask_masked_lst, dim=0)
        attention_mask_masked_lst = attention_mask_masked_lst.view(n_inputs * trigger_len, -1)

        # 2. Calculate the logits, in batches
        mask_probs_lst = []
        for i in trange(0, input_ids_masked_lst.shape[0], self.batch_size,
                        desc="Calculating fluency scores..."):
            with torch.no_grad():
                mask_probs = self.model(input_ids=input_ids_masked_lst[i:i+self.batch_size],
                                        attention_mask=attention_mask_masked_lst[i:i+self.batch_size]).logits
                mask_probs = mask_probs.log_softmax(dim=-1)
                mask_probs = mask_probs[:, trigger_slice.start:trigger_slice.stop, :]  # keep only trigger_slice, for memory efficiency
                mask_probs = mask_probs.cpu()  # to avoid GPU OOM when keeping these tensors
            mask_probs_lst.append(mask_probs)

        mask_probs_lst = torch.cat(mask_probs_lst, dim=0)  # (n_inputs*trigger_len, trigger_len, vocab_size)

        # 3.A. Select relevant indices and reshape to (n_inputs, trigger_len, vocab_size)
        mask_probs_lst = mask_probs_lst[
            torch.arange(n_inputs * trigger_len), torch.arange(trigger_len).repeat(n_inputs),
        ].view(n_inputs, trigger_len, -1)

        # 3.B. Calculate fluency; select the relevant tokens in each index, an aggregate the score
        mask_probs_lst = mask_probs_lst.to(input_ids.device)
        fluency_scores = torch.zeros(n_inputs, device=input_ids.device)
        for i in range(n_inputs):  # TODO vectorize
            fluency_scores[i] = mask_probs_lst[
                i, np.arange(trigger_len), input_ids[i, trigger_slice.start:trigger_slice.stop]
            ].mean()

        return fluency_scores.to(input_ids.device)


class GPT2FluencyScorer:
    def __init__(self, batch_size: int = 128, **kwargs):
        self.batch_size = batch_size

        def load_nanogpt():
            from src.nanoGPT.model import GPTConfig, GPT   # [IMPORTANT]: requires nanoGPT on BERT tokenizer
            # init from a model saved in a specific directory

            ckpt_path = hf_hub_download(repo_id="MatanBT/nanoGPT-BERT-Tokenizer", filename="ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location='cuda')
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf).to('cuda')
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)

            tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")  # = BERT tokenizer
            max_length = gptconf.block_size

            return model, tokenizer

        self.model, self.tokenizer = load_nanogpt()
        self.input_embedding = self.model.transformer.wte

        self.model.eval()
        for param in self.model.parameters():  # prevent the model from updating the gradients
            param.requires_grad = False

    def calc_score_for_grad(self, one_hot: torch.tensor, trigger_slice: slice, inputs: dict):

        # Get the embeddings of the trigger
        trigger_embeds = (one_hot @ self.input_embedding.weight)  # (batch_size, targeted_sub_seq_len, embed_dim)

        # now stitch it together with the rest of the embeddings
        embeds = self.input_embedding(inputs['input_ids']).detach()  # (batch_size, seq_len, embed_dim)
        full_embeds = torch.cat(
            [
                embeds[:, :trigger_slice.start, :],
                trigger_embeds,
                embeds[:, trigger_slice.stop:, :]
            ],
            dim=1)

        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        targets = input_ids.clone()
        # if targets[0, 0] == self.tokenizer.cls_token_id:  # [DISABLED]
        #     targets[:, :2] = -100  # ignore prediction _from_ CLS (was not trained on)
        targets[targets == self.tokenizer.pad_token_id] = -100  # ignore padding

        # [HACK] optimize calculation by discarding long padding suffix (if batch is of same length - also aligns the input!)
        padding_suffix_start_idx = min([i for i in range(input_ids.shape[-1]) if attention_mask[:, i].sum() == 0] + [input_ids.shape[-1]])
        input_ids = input_ids[:, :padding_suffix_start_idx]
        targets = targets[:, :padding_suffix_start_idx]
        attention_mask = attention_mask[:, :padding_suffix_start_idx]
        full_embeds = full_embeds[:, :padding_suffix_start_idx, :]
        if (attention_mask == 0).sum() != 0:
            logger.warning("Attention mask contains zeros, which might affect the fluency score calculation. "
                           "nanoGPT does not support attention mask in inference time!")
            raise NotImplementedError()

        logits, loss = self.model(input_ids, tok_emb=full_embeds, targets=targets)

        fluency_score = -loss  # the lower the CE loss -> the higher the fluency
        return fluency_score

    def calc_fluency_scores(self, inputs: dict, targets=None,
                            return_logits=False,
                            text_inputs=False,  # if True, the `inputs` are the given text, and should be tokenized
                            ):
        if text_inputs:
            inputs = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        targets = targets.clone() if targets is not None else input_ids.clone()
        targets[targets == self.tokenizer.pad_token_id] = -100  # ignore padding

        # [HACK] optimize calculation by discarding long padding suffix (if batch is of same length - also aligns the input!)
        padding_suffix_start_idx = min([i for i in range(input_ids.shape[-1]) if attention_mask[:, i].sum() == 0] + [input_ids.shape[-1]])
        input_ids = input_ids[:, :padding_suffix_start_idx]
        targets = targets[:, :padding_suffix_start_idx]
        attention_mask = attention_mask[:, :padding_suffix_start_idx]
        if (attention_mask == 0).sum() != 0:
            logger.warning("Attention mask contains zeros, which might affect the fluency score calculation. "
                           "nanoGPT does not support attention mask in inference time!")

        for i in trange(0, input_ids.shape[0], self.batch_size, desc="Calculating fluency scores..."):
            batch_input_ids = input_ids[i: i+self.batch_size]
            batch_targets = targets[i: i+self.batch_size]
            batch_logits, batch_losses = self.model(batch_input_ids, targets=batch_targets, reduce_loss=False)
            batch_fluency_score = -batch_losses.cpu()  # the lower the CE loss -> the higher the fluency
            if i == 0:
                fluency_scores = batch_fluency_score
                logits = batch_logits.cpu()
            else:
                fluency_scores = torch.cat((fluency_scores, batch_fluency_score), dim=0)
                logits = torch.cat((logits, batch_logits.cpu()), dim=0)

        if return_logits:
            return logits.to('cuda')
        return fluency_scores.to('cuda')


def initialize_fluency_model(fluency_model_name: str, **kwargs):
    if fluency_model_name == 'gpt2':
        return GPT2FluencyScorer(**kwargs)
    elif fluency_model_name == 'bert_mlm':
        return BertMLMFluencyScorer(**kwargs)

    # TODO perform some check to make sure the tokenizer is the same
    # assert orig_tokenizer.__class__ == flu_model.tokenizer.__class__, \
    #     "The tokenizer of the fluency model should be the same as the original model's tokenizer"

