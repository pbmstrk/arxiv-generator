from transformers import T5Tokenizer

class T5Encoder:

    def __init__(self, tokenizer_path='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

    def __call__(self, src_texts, tgt_texts=None, **kwargs):
        return self.tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts, 
            tgt_texts=tgt_texts, 
            **kwargs
        )

    def collate_fn(self, batch):
        src_texts = [data[0] for data in batch]
        tgt_texts = [data[1] for data in batch]

        return self(
            src_texts=src_texts, 
            tgt_texts=tgt_texts, 
            return_tensors='pt', 
            padding='longest'
        )