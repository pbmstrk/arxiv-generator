from transformers import AutoTokenizer


class Seq2SeqTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, src_texts, tgt_texts=None, **kwargs):
        return self.tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts, tgt_texts=tgt_texts, **kwargs
        )

    def collate_fn(self, batch):
        src_texts = [data[0] for data in batch]
        tgt_texts = [data[1] for data in batch]

        return self(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            return_tensors="pt",
            padding="longest",
        )

    def batch_decode(self, sequences, **kwargs):
        return self.tokenizer.batch_decode(sequences, **kwargs)
