import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Tokenizer


class TranslationTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    
    def train(self, texts: list[str], save_dir: str = "./tokenizer"):
        bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        bpe_tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        bpe_tokenizer.train_from_iterator(texts, trainer)
        
        os.makedirs(save_dir, exist_ok=True)
        
        vocab = bpe_tokenizer.get_vocab()
        vocab_file = os.path.join(save_dir, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        
        merges = []
        if hasattr(bpe_tokenizer.model, 'get_merges'):
            merges = bpe_tokenizer.model.get_merges()
        
        merges_file = os.path.join(save_dir, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        self.tokenizer = GPT2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]"
        )
        
        return self.tokenizer
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        tokens = self.tokenizer.encode(text)
        if add_special_tokens:
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            tokens = [bos_id] + tokens + [eos_id]
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def vocab_size_actual(self) -> int:
        return len(self.tokenizer)
