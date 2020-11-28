from transformers import AutoTokenizer


class Seq2SeqTokenizer:
    def __init__(self, tokenizer_path, predict_abstract=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.predict_abstract = predict_abstract

    def __call__(self, src_texts, tgt_texts=None, **kwargs):
        return self.tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts, tgt_texts=tgt_texts, **kwargs
        )

    def collate_fn(self, batch):

        titles = [data["title"] for data in batch]
        abstracts = [data["abstract"] for data in batch]

        return self(
            src_texts=titles if self.predict_abstract else abstracts,
            tgt_texts=abstracts if self.predict_abstract else titles,
            return_tensors="pt",
            padding="longest",
        )

    def batch_decode(self, sequences, **kwargs):
        return self.tokenizer.batch_decode(sequences, **kwargs)
