import os
import sys
import logging
from datetime import datetime

import torch
import yaml
import sacrebleu
from tqdm import tqdm
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelling import Transformer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "wmt17_sinusoidal.yaml")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_data", "tokenizer.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "weights_sinusoidal_first_run", "best_val_loss.pt")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"practical10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def greedy_decode(model, tokenizer, src_text, device, max_src_len):
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_ids = src_ids[:max_src_len - 2]
    src_ids = [tokenizer.bos_token_id] + src_ids + [tokenizer.eos_token_id]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (src_tensor != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    memory = model.encode(src_tensor, src_mask)
    ys = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    
    for _ in range(max_src_len - 1):
        tgt_len = ys.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1) == 0
        tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        decoder_output = model.decode(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = model.projection(decoder_output[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        ys = torch.cat([ys, next_token], dim=1)
    
    return tokenizer.decode(ys[0, 1:].tolist(), skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH,
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )
    
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_seq_length'],
        positional_encoding_type=config['positional_encoding_type'],
        use_gqa=False
    )
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    logger.info("Model loaded")
    
    test_dataset = load_dataset("wmt17", "de-en", split="test")
    test_pairs = [(item['translation']['de'], item['translation']['en']) for item in test_dataset]
    logger.info(f"Test samples: {len(test_pairs)}")
    
    max_src_len = config['max_seq_length']
    predictions = []
    references = []
    
    for src_text, ref_text in tqdm(test_pairs, desc="Translating"):
        pred = greedy_decode(model, tokenizer, src_text, device, max_src_len)
        predictions.append(pred)
        references.append(ref_text)
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    logger.info(f"BLEU Score: {bleu.score:.2f}")

    results = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        src = test_pairs[i][0]
        score = sacrebleu.sentence_bleu(pred, [ref]).score
        results.append({'src': src, 'ref': ref, 'pred': pred, 'score': score})

    results.sort(key=lambda x: x['score'], reverse=True)

    def log_examples(title, items):
        logger.info(f"\n--- {title} ---")
        for item in items:
            logger.info(f"Source: {item['src']}")
            logger.info(f"Ref:    {item['ref']}")
            logger.info(f"Pred:   {item['pred']}")
            logger.info(f"BLEU:   {item['score']:.2f}")

    import random
    log_examples("Top 5 Best Predictions", results[:5])
    log_examples("Bottom 5 Worst Predictions", results[-5:])
    log_examples("5 Random Predictions", random.sample(results, min(5, len(results))))

    logger.info(f"Log saved: {log_filename}")


if __name__ == "__main__":
    main()
