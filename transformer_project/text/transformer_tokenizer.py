import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

class TokenizerTrainer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size

    def train_and_get_tokenizer(self, corpus_iterator, save_path="tokenizer_data"):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        tokenizer_file = os.path.join(save_path, "tokenizer.json")
        tokenizer.save(tokenizer_file)
        
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]"
        )
        
        return hf_tokenizer
