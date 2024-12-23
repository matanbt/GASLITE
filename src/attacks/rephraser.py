from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5Rephraser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").cuda()

    def rephrase(self, text: str, num_return_sequences=5):
        # To template:
        text = "paraphrase: " + text + " </s>"

        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )

        par_texts = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                     for output in outputs]

        return par_texts
