import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_length=50):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        assert self.tokenizer.pad_token_id is not None, "Tokenizer must have pad_token set"
        assert self.tokenizer.bos_token_id is not None, "Tokenizer must have bos_token set"
        assert self.tokenizer.eos_token_id is not None, "Tokenizer must have eos_token set"
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text, tgt_text = self.dataset[idx]

        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        src_ids = src_ids[:self.max_seq_length - 2]
        src_ids = [self.bos_token_id] + src_ids + [self.eos_token_id]
        
        n_pad_src = self.max_seq_length - len(src_ids)
        src_ids = src_ids + [self.pad_token_id] * n_pad_src
        
        tgt_ids_raw = self.tokenizer.encode(tgt_text, add_special_tokens=False)
        tgt_ids_raw = tgt_ids_raw[:self.max_seq_length - 1]
        
        decoder_input_ids = [self.bos_token_id] + tgt_ids_raw
        n_pad_dec = self.max_seq_length - len(decoder_input_ids)
        decoder_input_ids = decoder_input_ids + [self.pad_token_id] * n_pad_dec
        
        label_ids = tgt_ids_raw + [self.eos_token_id]
        label_ids = label_ids + [self.pad_token_id] * n_pad_dec
        
        encoder_input = torch.tensor(src_ids, dtype=torch.long)
        decoder_input = torch.tensor(decoder_input_ids, dtype=torch.long)
        label = torch.tensor(label_ids, dtype=torch.long)
        
        src_mask = (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0)
        tgt_mask = (decoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0)
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        }
