import torch
from torch.utils.data import Dataset
from typing import Optional


class TranslationDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, src_key: str = 'de', tgt_key: str = 'en',
                 max_len: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        src_ids = self.tokenizer.encode(item[self.src_key])
        tgt_ids = self.tokenizer.encode(item[self.tgt_key])
        
        src_ids = self._pad_or_truncate(src_ids)
        tgt_ids = self._pad_or_truncate(tgt_ids)
        
        src_mask = [1 if id != self.pad_id else 0 for id in src_ids]
        tgt_mask = [1 if id != self.pad_id else 0 for id in tgt_ids]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'src_mask': torch.tensor(src_mask, dtype=torch.long),
            'tgt_mask': torch.tensor(tgt_mask, dtype=torch.long)
        }
    
    def _pad_or_truncate(self, ids: list[int]) -> list[int]:
        if len(ids) > self.max_len:
            return ids[:self.max_len]
        return ids + [self.pad_id] * (self.max_len - len(ids))


def collate_fn(batch: list[dict]) -> dict:
    return {
        'src_ids': torch.stack([item['src_ids'] for item in batch]),
        'tgt_ids': torch.stack([item['tgt_ids'] for item in batch]),
        'src_mask': torch.stack([item['src_mask'] for item in batch]),
        'tgt_mask': torch.stack([item['tgt_mask'] for item in batch])
    }
