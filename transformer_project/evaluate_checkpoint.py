import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import sys
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import time
import numpy as np
from tabulate import tabulate
import sacrebleu

from transformer_project.modelling import Transformer
from transformer_project.dataset import TranslationDataset
from transformer_project.text.transformer_tokenizer import TokenizerTrainer


def setup_logging(log_dir="logs", checkpoint_name="checkpoint"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = checkpoint_name.replace("/", "_").replace("\\", "_").replace(".pt", "")
    log_file = os.path.join(log_dir, f"eval_{safe_name}_{timestamp}.log")
    
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def translate_sentence(model, tokenizer, src_text, device, max_len=100):
    model.eval()
    
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
    src_ids = src_ids[:max_len - 2]
    src_ids = [tokenizer.bos_token_id] + src_ids + [tokenizer.eos_token_id]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)
    
    ys = torch.ones(1, 1).fill_(tokenizer.bos_token_id).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        sz = ys.size(1)
        tgt_mask = torch.triu(torch.ones(sz, sz, device=device)).transpose(0, 1).type(torch.bool)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            out = model.decode(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
            logits = model.projection(out[:, -1])
        
        next_word = logits.argmax(dim=1).item()
        if next_word == tokenizer.eos_token_id:
            break
        ys = torch.cat([ys, torch.ones(1, 1).to(device).fill_(next_word).long()], dim=1)
    
    tokens = ys[0, 1:].tolist()
    return tokenizer.decode(tokens)


def evaluate_perplexity(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Perplexity"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            label = batch['label'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            
            output = model(encoder_input, decoder_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            target = label.view(-1)
            loss = criterion(output, target)
            
            non_pad = (target != criterion.ignore_index).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)
    
    return {"loss": avg_loss, "perplexity": perplexity}


def evaluate_bleu(model, tokenizer, test_data, device, num_samples=100, max_len=100):
    if num_samples and num_samples < len(test_data):
        test_data = test_data[:num_samples]
    
    predictions = []
    references = []
    
    model.eval()
    for src_text, ref_text in tqdm(test_data, desc="Evaluating BLEU"):
        pred_text = translate_sentence(model, tokenizer, src_text, device, max_len=max_len)
        predictions.append(pred_text)
        references.append(ref_text)
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references


def measure_inference_latency(model, tokenizer, device, batch_sizes=[8, 16, 32, 64, 128, 256, 512], 
                               seq_len=50, num_warmup=5, num_runs=20):
    model.eval()
    results = {}
    vocab_size = len(tokenizer)
    
    for batch_size in batch_sizes:
        src = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
        tgt = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
        src_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)
        tgt_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(src, tgt, src_mask, tgt_mask)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(src, tgt, src_mask, tgt_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
        
        results[batch_size] = {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies)
        }
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer checkpoint")
    
    # Checkpoint path
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint file (e.g., weights_sinusoidal_first_run/epoch_5.pt)")
    
    # Model configuration (should match training config)
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--positional_encoding", type=str, default="sinusoidal",
                        choices=["sinusoidal", "rope"], help="Positional encoding type")
    
    # GQA options (for evaluating GQA models)
    parser.add_argument("--use_gqa", action="store_true", help="Model uses GQA")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of KV heads (for GQA)")
    
    # Evaluation options
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for perplexity evaluation")
    parser.add_argument("--num_bleu_samples", type=int, default=None, help="Number of samples for BLEU evaluation (default: all)")
    parser.add_argument("--skip_bleu", action="store_true", help="Skip BLEU evaluation (faster)")
    parser.add_argument("--skip_latency", action="store_true", help="Skip latency benchmarks")
    parser.add_argument("--show_samples", type=int, default=5, help="Number of translation samples to show")
    
    args = parser.parse_args()
    
    # Setup logging
    script_dir = os.path.dirname(__file__)
    log_dir = os.path.join(script_dir, "logs")
    log_file = setup_logging(log_dir, os.path.basename(args.checkpoint))
    logging.info(f"Logging to: {log_file}")
    
    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(script_dir, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        logging.info(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info("=" * 70)
    logging.info("TRANSFORMER CHECKPOINT EVALUATION")
    logging.info("=" * 70)
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Attention: {'GQA' if args.use_gqa else 'MHA'}")
    logging.info("=" * 70)
    
    # ==========================================================================
    # Load Dataset
    # ==========================================================================
    logging.info("\n[1/5] Loading dataset...")
    CACHE_DIR = os.path.join(script_dir, "cache_huggingface")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    dataset = load_dataset("wmt17", "de-en", trust_remote_code=True, cache_dir=CACHE_DIR)
    train_data = dataset['train']
    
    if 'validation' in dataset:
        val_data = dataset['validation']
    else:
        split = train_data.train_test_split(test_size=0.01, seed=42)
        train_data = split['train']
        val_data = split['test']
    
    test_data = dataset.get('test', val_data)
    
    logging.info(f"   Train samples: {len(train_data)}")
    logging.info(f"   Validation samples: {len(val_data)}")
    logging.info(f"   Test samples: {len(test_data)}")
    
    # ==========================================================================
    # Load Tokenizer
    # ==========================================================================
    logging.info("\n[2/5] Loading tokenizer...")
    tokenizer_path = os.path.join(script_dir, "tokenizer_data")
    
    if os.path.exists(tokenizer_path):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"))
        tokenizer.pad_token = "[PAD]"
        tokenizer.bos_token = "[BOS]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.unk_token = "[UNK]"
        logging.info(f"   Loaded tokenizer from {tokenizer_path}")
    else:
        logging.info("   Tokenizer not found, training new one...")
        trainer = TokenizerTrainer(vocab_size=args.vocab_size)
        def corpus_iterator():
            for item in train_data:
                yield item['translation']['de']
                yield item['translation']['en']
        tokenizer = trainer.train_and_get_tokenizer(corpus_iterator(), save_path=tokenizer_path)
    
    logging.info(f"   Vocabulary size: {len(tokenizer)}")
    
    # ==========================================================================
    # Create Model and Load Checkpoint
    # ==========================================================================
    logging.info("\n[3/5] Loading model...")
    
    num_kv_heads = args.num_kv_heads if args.num_kv_heads else args.n_heads
    
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_seq_length,
        positional_encoding_type=args.positional_encoding,
        use_gqa=args.use_gqa,
        num_kv_heads=num_kv_heads
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    # Model statistics
    n_params = count_parameters(model)
    model_memory = get_model_memory_mb(model)
    kv_info = model.get_kv_cache_info(batch_size=1, seq_len=args.max_seq_length)
    
    logging.info(f"   Parameters: {n_params:,}")
    logging.info(f"   Model Memory: {model_memory:.2f} MB")
    logging.info(f"   Attention Type: {model.get_attention_type()}")
    logging.info(f"   KV Cache Savings: {kv_info['memory_savings_percent']:.1f}%")
    
    # ==========================================================================
    # Create DataLoaders
    # ==========================================================================
    def process_data(data):
        return [(item['translation']['de'], item['translation']['en']) for item in data]
    
    val_raw = process_data(val_data)
    test_raw = process_data(test_data)
    
    val_ds = TranslationDataset(val_raw, tokenizer, max_seq_length=args.max_seq_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # ==========================================================================
    # Evaluate Perplexity
    # ==========================================================================
    logging.info("\n[4/5] Evaluating perplexity...")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ppl_results = evaluate_perplexity(model, val_loader, criterion, DEVICE)
    logging.info(f"   Validation Loss: {ppl_results['loss']:.4f}")
    logging.info(f"   Validation Perplexity: {ppl_results['perplexity']:.2f}")
    
    # ==========================================================================
    # Evaluate BLEU
    # ==========================================================================
    bleu_score = None
    predictions = []
    references = []
    
    if not args.skip_bleu:
        num_samples_msg = f"{args.num_bleu_samples}" if args.num_bleu_samples else "all"
        logging.info(f"\n[5/5] Evaluating BLEU ({num_samples_msg} samples)...")
        bleu_score, predictions, references = evaluate_bleu(
            model, tokenizer, test_raw, DEVICE, 
            num_samples=args.num_bleu_samples,
            max_len=args.max_seq_length
        )
        logging.info(f"   BLEU Score: {bleu_score:.2f}")
    else:
        logging.info("\n[5/5] Skipping BLEU evaluation...")
    
    # ==========================================================================
    # Latency Benchmarks
    # ==========================================================================
    latency_results = {}
    if not args.skip_latency:
        logging.info("\n[Bonus] Running latency benchmarks...")
        latency_results = measure_inference_latency(
            model, tokenizer, DEVICE,
            batch_sizes=[1, 8, 16, 32, 64, 128, 256, 512],
            seq_len=args.max_seq_length
        )
        logging.info(f"   Latency (batch=1): {latency_results[1]['mean_ms']:.2f} ± {latency_results[1]['std_ms']:.2f} ms")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 70)
    
    summary_data = [
        ["Metric", "Value"],
        ["Checkpoint", os.path.basename(checkpoint_path)],
        ["Parameters", f"{n_params:,}"],
        ["Model Memory", f"{model_memory:.2f} MB"],
        ["Attention Type", model.get_attention_type()],
        ["KV Cache Savings", f"{kv_info['memory_savings_percent']:.1f}%"],
        ["Validation Loss", f"{ppl_results['loss']:.4f}"],
        ["Validation Perplexity", f"{ppl_results['perplexity']:.2f}"],
    ]
    
    if bleu_score is not None:
        summary_data.append(["BLEU Score", f"{bleu_score:.2f}"])
    
    if latency_results:
        summary_data.append(["Latency (bs=1)", f"{latency_results[1]['mean_ms']:.2f} ± {latency_results[1]['std_ms']:.2f} ms"])
        summary_data.append(["Latency (bs=8)", f"{latency_results[8]['mean_ms']:.2f} ± {latency_results[8]['std_ms']:.2f} ms"])
        summary_data.append(["Latency (bs=32)", f"{latency_results[32]['mean_ms']:.2f} ± {latency_results[32]['std_ms']:.2f} ms"])
    
    logging.info(tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    
    # KV Cache Analysis
    logging.info("\n" + "-" * 70)
    logging.info("KV CACHE ANALYSIS")
    logging.info("-" * 70)
    d_k = args.d_model // args.n_heads
    kv_heads = num_kv_heads if args.use_gqa else args.n_heads
    
    logging.info(f"   d_model: {args.d_model}")
    logging.info(f"   d_k (per head): {d_k}")
    logging.info(f"   Query heads: {args.n_heads}")
    logging.info(f"   KV heads: {kv_heads}")
    logging.info(f"   Layers: {args.num_layers} encoder + {args.num_layers} decoder")
    logging.info(f"\n   KV Cache per layer (batch=1, seq={args.max_seq_length}):")
    kv_cache_per_layer = 2 * 1 * kv_heads * args.max_seq_length * d_k * 4  # 4 bytes for float32
    logging.info(f"   - Size: {kv_cache_per_layer / 1024:.2f} KB")
    total_kv_cache = kv_cache_per_layer * (args.num_layers + 2 * args.num_layers)  # enc + dec(self+cross)
    logging.info(f"   - Total (all layers): {total_kv_cache / (1024 * 1024):.2f} MB")
    
    # Latency table
    if latency_results:
        logging.info("\n" + "-" * 70)
        logging.info("LATENCY BENCHMARKS")
        logging.info("-" * 70)
        latency_table = [["Batch Size", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)"]]
        for bs, lat in latency_results.items():
            latency_table.append([bs, f"{lat['mean_ms']:.2f}", f"{lat['std_ms']:.2f}", 
                                  f"{lat['min_ms']:.2f}", f"{lat['max_ms']:.2f}"])
        logging.info(tabulate(latency_table, headers="firstrow", tablefmt="grid"))
    
    # Sample translations
    if args.show_samples > 0 and predictions:
        logging.info("\n" + "-" * 70)
        logging.info("SAMPLE TRANSLATIONS")
        logging.info("-" * 70)
        for i in range(min(args.show_samples, len(predictions))):
            src, ref = test_raw[i]
            pred = predictions[i]
            logging.info(f"\n--- Sample {i+1} ---")
            logging.info(f"Source (DE):    {src[:100]}{'...' if len(src) > 100 else ''}")
            logging.info(f"Reference (EN): {ref[:100]}{'...' if len(ref) > 100 else ''}")
            logging.info(f"Prediction:     {pred[:100]}{'...' if len(pred) > 100 else ''}")
    
    logging.info("\n" + "=" * 70)
    logging.info("Evaluation complete!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
