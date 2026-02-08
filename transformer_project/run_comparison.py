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
import yaml
import warnings
import wandb
import numpy as np
from tabulate import tabulate
import gc
import sacrebleu

from transformer_project.modelling import Transformer
from transformer_project.optimization import TransformerLRScheduler, build_optimizer
from transformer_project.dataset import TranslationDataset
from transformer_project.text.transformer_tokenizer import TokenizerTrainer

warnings.filterwarnings("ignore")


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"comparison_{timestamp}.log")
    
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


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_memory_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def measure_inference_latency(model, tokenizer, device, batch_sizes=[1, 2, 4, 8, 16, 32], 
                               seq_len=50, num_warmup=3, num_runs=10):
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


def measure_memory_usage(model, tokenizer, device, batch_size, seq_len):
    if device != "cuda":
        return {"allocated_mb": 0, "reserved_mb": 0}
    
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    vocab_size = len(tokenizer)
    src = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
    tgt = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
    src_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)
    tgt_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)
    
    with torch.no_grad():
        _ = model(src, tgt, src_mask, tgt_mask)
    
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    
    torch.cuda.empty_cache()
    
    return {"allocated_mb": allocated, "reserved_mb": reserved}


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None, 
                scaler=None, mixed_precision=False, amp_dtype=torch.float16):
    model.train()
    total_loss = 0
    total_tokens = 0
    nan_count = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        label = batch['label'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        optimizer.zero_grad()
        
        if mixed_precision:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(encoder_input, decoder_input, src_mask, tgt_mask)
                output = output.view(-1, output.size(-1))
                target = label.view(-1)
                loss = criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            output = model(encoder_input, decoder_input, src_mask, tgt_mask)
            output = output.view(-1, output.size(-1))
            target = label.view(-1)
            loss = criterion(output, target)
            
            # Check for NaN before backward
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        non_pad = (target != criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    
    if nan_count > 0:
        logging.info(f"  Warning: {nan_count} batches had NaN/Inf loss and were skipped")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    perplexity = np.exp(min(avg_loss, 100)) if not np.isnan(avg_loss) else float('nan')
    
    return {"loss": avg_loss, "perplexity": perplexity}


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
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


def translate_sentence(model, tokenizer, src_text, device, max_len=64):
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


def evaluate_bleu(model, tokenizer, test_data, device, num_samples=50, max_len=64):
    if num_samples:
        test_data = test_data[:num_samples]
    
    predictions = []
    references = []
    
    model.eval()
    for src_text, ref_text in tqdm(test_data, desc="BLEU Eval", leave=False):
        pred_text = translate_sentence(model, tokenizer, src_text, device, max_len=max_len)
        predictions.append(pred_text)
        references.append(ref_text)
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def run_comparison_experiment(args):
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    log_file = setup_logging(log_dir)
    logging.info(f"Logging to: {log_file}")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {DEVICE}")
    
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    
    wandb_project = os.environ.get("WANDB_PROJECT", "mha-gqa-comparison")
    wandb_entity = os.environ.get("WANDB_ENTITY", None)
    
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"mha-vs-gqa-comparison-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "experiment_type": "mha_vs_gqa_comparison",
            "batch_sizes_tested": args.batch_sizes,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "num_kv_heads": args.num_kv_heads
        }
    )
    
    logging.info("\nLoading dataset...")
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_huggingface")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    dataset = load_dataset("wmt17", "de-en", trust_remote_code=True, cache_dir=CACHE_DIR)
    train_data = dataset['train']
    
    if 'validation' in dataset:
        val_data = dataset['validation']
    else:
        split = train_data.train_test_split(test_size=0.01, seed=42)
        train_data = split['train']
        val_data = split['test']
    
    if args.quick:
        train_data = train_data.select(range(min(10000, len(train_data))))
        val_data = val_data.select(range(min(1000, len(val_data))))
    
    logging.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    logging.info("\nPreparing tokenizer...")
    trainer = TokenizerTrainer(vocab_size=args.vocab_size)
    
    def corpus_iterator():
        for item in train_data:
            yield item['translation']['de']
            yield item['translation']['en']
    
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer_data")
    tokenizer = trainer.train_and_get_tokenizer(corpus_iterator(), save_path=tokenizer_path)
    
    def process_data(data):
        return [(item['translation']['de'], item['translation']['en']) for item in data]
    
    train_raw = process_data(train_data)
    val_raw = process_data(val_data)
    
    train_ds = TranslationDataset(train_raw, tokenizer, max_seq_length=args.max_seq_length)
    val_ds = TranslationDataset(val_raw, tokenizer, max_seq_length=args.max_seq_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    logging.info("\n" + "="*60)
    logging.info("Creating Models for Comparison")
    logging.info("="*60)
    
    model_configs = {
        "MHA": {
            "use_gqa": False,
            "num_kv_heads": args.n_heads  # Same as query heads for MHA
        },
        "GQA": {
            "use_gqa": True,
            "num_kv_heads": args.num_kv_heads
        }
    }
    
    results = {"MHA": {}, "GQA": {}}
    
    for model_name, model_cfg in model_configs.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing: {model_name}")
        logging.info(f"{'='*60}")
        
        # Create model
        model = Transformer(
            vocab_size=len(tokenizer),
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_seq_length,
            positional_encoding_type="sinusoidal",
            use_gqa=model_cfg["use_gqa"],
            num_kv_heads=model_cfg["num_kv_heads"]
        )
        model.to(DEVICE)
        
        n_params = count_parameters(model)
        model_memory = get_model_memory_mb(model)
        kv_info = model.get_kv_cache_info(args.batch_size, args.max_seq_length)
        
        logging.info(f"  Parameters: {n_params:,}")
        logging.info(f"  Model Memory: {model_memory:.2f} MB")
        logging.info(f"  Attention Type: {model.get_attention_type()}")
        logging.info(f"  KV Cache Savings: {kv_info['memory_savings_percent']:.1f}%")
        
        results[model_name]["params"] = n_params
        results[model_name]["model_memory_mb"] = model_memory
        results[model_name]["kv_cache_savings_percent"] = kv_info["memory_savings_percent"]
        
        wandb.log({
            f"{model_name}/parameters": n_params,
            f"{model_name}/model_memory_mb": model_memory,
            f"{model_name}/kv_cache_savings_percent": kv_info["memory_savings_percent"],
            f"{model_name}/kv_cache_reduction_ratio": kv_info["memory_reduction_ratio"]
        })
        
        logging.info(f"\n  Running latency benchmarks...")
        latency_results = measure_inference_latency(
            model, tokenizer, DEVICE, 
            batch_sizes=args.batch_sizes,
            seq_len=args.max_seq_length
        )
        
        results[model_name]["latency"] = latency_results
        
        for bs, lat in latency_results.items():
            wandb.log({
                f"{model_name}/latency_bs{bs}_mean_ms": lat["mean_ms"],
                f"{model_name}/latency_bs{bs}_std_ms": lat["std_ms"]
            })
        
        logging.info(f"  Latency (batch=1): {latency_results[1]['mean_ms']:.2f} ± {latency_results[1]['std_ms']:.2f} ms")
        
        if DEVICE == "cuda":
            logging.info(f"\n  Running memory benchmarks...")
            for bs in args.batch_sizes[:3]:
                mem = measure_memory_usage(model, tokenizer, DEVICE, bs, args.max_seq_length)
                wandb.log({
                    f"{model_name}/memory_bs{bs}_allocated_mb": mem["allocated_mb"],
                    f"{model_name}/memory_bs{bs}_reserved_mb": mem["reserved_mb"]
                })
        
        logging.info(f"\n  Training for {args.epochs} epochs...")
        
        optimizer = build_optimizer(model, learning_rate=1.0, weight_decay=0.1)
        scheduler = TransformerLRScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = None
            logging.info(f"  Using bfloat16 mixed precision")
        elif DEVICE == "cuda":
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            logging.info(f"  Using float16 mixed precision with GradScaler")
        else:
            amp_dtype = torch.float32
            scaler = None
        
        training_history = []
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, DEVICE,
                scheduler=scheduler, scaler=scaler, mixed_precision=(DEVICE == "cuda"),
                amp_dtype=amp_dtype
            )
            
            val_metrics = validate(model, val_loader, criterion, DEVICE)
            
            epoch_time = time.time() - epoch_start
            
            wandb.log({
                f"{model_name}/train_loss": train_metrics["loss"],
                f"{model_name}/train_perplexity": train_metrics["perplexity"],
                f"{model_name}/val_loss": val_metrics["loss"],
                f"{model_name}/val_perplexity": val_metrics["perplexity"],
                f"{model_name}/epoch_time_s": epoch_time,
                "epoch": epoch + 1
            })
            
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_ppl": train_metrics["perplexity"],
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["perplexity"],
                "time_s": epoch_time
            })
            
            logging.info(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"Train Loss={train_metrics['loss']:.4f}, PPL={train_metrics['perplexity']:.2f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, PPL={val_metrics['perplexity']:.2f} | "
                  f"Time={epoch_time:.1f}s")
        
        results[model_name]["training_history"] = training_history
        results[model_name]["total_training_time"] = sum(h["time_s"] for h in training_history)
        
        logging.info(f"\n  Evaluating BLEU score...")
        bleu_score = evaluate_bleu(model, tokenizer, val_raw[:50], DEVICE, num_samples=50, max_len=args.max_seq_length)
        results[model_name]["bleu_score"] = bleu_score
        logging.info(f"  BLEU Score: {bleu_score:.2f}")
        
        del model, optimizer, scheduler
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    logging.info("\n" + "="*60)
    logging.info("Creating Comparison Plots")
    logging.info("="*60)
    
    epochs = list(range(1, args.epochs + 1))
    
    mha_train_loss = [h["train_loss"] for h in results["MHA"]["training_history"]]
    gqa_train_loss = [h["train_loss"] for h in results["GQA"]["training_history"]]
    
    wandb.log({
        "Training Loss Comparison": wandb.plot.line_series(
            xs=[epochs, epochs],
            ys=[mha_train_loss, gqa_train_loss],
            keys=["MHA", "GQA"],
            title="Training Loss Comparison",
            xname="Epoch"
        )
    })
    
    mha_val_loss = [h["val_loss"] for h in results["MHA"]["training_history"]]
    gqa_val_loss = [h["val_loss"] for h in results["GQA"]["training_history"]]
    
    wandb.log({
        "Validation Loss Comparison": wandb.plot.line_series(
            xs=[epochs, epochs],
            ys=[mha_val_loss, gqa_val_loss],
            keys=["MHA", "GQA"],
            title="Validation Loss Comparison",
            xname="Epoch"
        )
    })
    
    mha_val_ppl = [h["val_ppl"] for h in results["MHA"]["training_history"]]
    gqa_val_ppl = [h["val_ppl"] for h in results["GQA"]["training_history"]]
    
    wandb.log({
        "Perplexity Comparison": wandb.plot.line_series(
            xs=[epochs, epochs],
            ys=[mha_val_ppl, gqa_val_ppl],
            keys=["MHA", "GQA"],
            title="Validation Perplexity Comparison",
            xname="Epoch"
        )
    })
    
    bleu_table = wandb.Table(columns=["Model", "BLEU Score"])
    bleu_table.add_data("MHA", results["MHA"]["bleu_score"])
    bleu_table.add_data("GQA", results["GQA"]["bleu_score"])
    
    wandb.log({
        "BLEU Score Comparison": wandb.plot.bar(
            bleu_table, "Model", "BLEU Score",
            title="Validation BLEU Score Comparison"
        )
    })
    
    wandb.log({
        "Inference Latency Comparison": wandb.plot.line_series(
            xs=[args.batch_sizes, args.batch_sizes],
            ys=[[results["MHA"]["latency"][bs]["mean_ms"] for bs in args.batch_sizes],
                [results["GQA"]["latency"][bs]["mean_ms"] for bs in args.batch_sizes]],
            keys=["MHA", "GQA"],
            title="Inference Latency vs Batch Size (ms)",
            xname="Batch Size"
        )
    })
    
    speed_table = wandb.Table(columns=["Model", "Training Time (s)"])
    speed_table.add_data("MHA", results["MHA"]["total_training_time"])
    speed_table.add_data("GQA", results["GQA"]["total_training_time"])
    
    wandb.log({
        "Training Speed Comparison": wandb.plot.bar(
            speed_table, "Model", "Training Time (s)",
            title="Total Training Time Comparison"
        )
    })
    
    kv_cache_reduction_ratio = args.num_kv_heads / args.n_heads
    kv_cache_savings_percent = (1 - kv_cache_reduction_ratio) * 100
    
    wandb.log({
        "summary/mha_params": results["MHA"]["params"],
        "summary/gqa_params": results["GQA"]["params"],
        "summary/param_reduction_percent": (1 - results["GQA"]["params"] / results["MHA"]["params"]) * 100,
        "summary/kv_cache_savings_percent": kv_cache_savings_percent,
        "summary/mha_final_bleu": results["MHA"]["bleu_score"],
        "summary/gqa_final_bleu": results["GQA"]["bleu_score"],
        "summary/mha_final_ppl": results["MHA"]["training_history"][-1]["val_ppl"],
        "summary/gqa_final_ppl": results["GQA"]["training_history"][-1]["val_ppl"],
    })
    
    logging.info("\n" + "="*60)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("="*60)
    
    kv_cache_reduction_ratio = args.num_kv_heads / args.n_heads
    kv_cache_savings_percent = (1 - kv_cache_reduction_ratio) * 100
    
    param_savings_percent = (1 - results["GQA"]["params"] / results["MHA"]["params"]) * 100
    
    summary_data = [
        ["Metric", "MHA", "GQA", "Difference"],
        ["Parameters", f"{results['MHA']['params']:,}", f"{results['GQA']['params']:,}", 
         f"{(results['GQA']['params'] - results['MHA']['params']):,}"],
        ["KV Cache Size Ratio", "100%", f"{kv_cache_reduction_ratio*100:.0f}%", f"-{kv_cache_savings_percent:.0f}%"],
        ["Final Val PPL", f"{results['MHA']['training_history'][-1]['val_ppl']:.2f}",
         f"{results['GQA']['training_history'][-1]['val_ppl']:.2f}",
         f"{results['GQA']['training_history'][-1]['val_ppl'] - results['MHA']['training_history'][-1]['val_ppl']:.2f}"],
        ["BLEU Score", f"{results['MHA']['bleu_score']:.2f}",
         f"{results['GQA']['bleu_score']:.2f}",
         f"{results['GQA']['bleu_score'] - results['MHA']['bleu_score']:.2f}"],
        ["Training Time (s)", f"{results['MHA']['total_training_time']:.1f}",
         f"{results['GQA']['total_training_time']:.1f}",
         f"{results['GQA']['total_training_time'] - results['MHA']['total_training_time']:.1f}"],
        ["Latency bs=1 (ms)", f"{results['MHA']['latency'][1]['mean_ms']:.2f}",
         f"{results['GQA']['latency'][1]['mean_ms']:.2f}",
         f"{results['GQA']['latency'][1]['mean_ms'] - results['MHA']['latency'][1]['mean_ms']:.2f}"]
    ]
    
    logging.info("\n" + tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    
    logging.info("\n" + "-"*60)
    logging.info("NOTES ON METRICS:")
    logging.info("-"*60)
    logging.info(f"\n📊 Parameter Savings ({param_savings_percent:.1f}%):")
    logging.info("   - GQA uses smaller K and V projection matrices")
    logging.info(f"   - MHA: K,V projections are d_model × d_model = {args.d_model}×{args.d_model}")
    logging.info(f"   - GQA: K,V projections are d_model × (num_kv_heads × d_k) = {args.d_model}×{args.num_kv_heads * (args.d_model // args.n_heads)}")
    logging.info("   - This reduces total model weights")
    
    logging.info(f"\n💾 KV Cache Savings ({kv_cache_savings_percent:.0f}%):")
    logging.info("   - This refers to INFERENCE-TIME memory savings")
    logging.info("   - During autoregressive generation, K and V are cached for each token")
    logging.info(f"   - MHA caches: {args.n_heads} heads × seq_len × d_k per layer")
    logging.info(f"   - GQA caches: {args.num_kv_heads} heads × seq_len × d_k per layer")
    logging.info("   - Critical for long-context generation and large batch inference")
    
    logging.info(f"\n🎯 Quality Metrics (PPL, BLEU):")
    logging.info("   - Lower perplexity = better (model is less confused)")
    logging.info("   - Higher BLEU = better (translations match references)")
    logging.info("   - Small quality loss is acceptable trade-off for memory savings")
    
    wandb.finish()
    logging.info(f"\n✓ Results logged to wandb and saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="MHA vs GQA Comparison Experiments")
    
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--num_kv_heads", type=int, default=2, help="Number of KV heads for GQA")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder/decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Batch sizes for latency benchmarks")
    
    parser.add_argument("--quick", action="store_true", help="Use smaller dataset for quick testing")
    
    args = parser.parse_args()
    
    if args.quick:
        args.warmup_steps = min(args.warmup_steps, 100)
    
    logging.info("="*60)
    logging.info("MHA vs GQA Comparison Experiment")
    logging.info("="*60)
    logging.info(f"Query Heads: {args.n_heads}")
    logging.info(f"KV Heads (GQA): {args.num_kv_heads}")
    logging.info(f"GQA Ratio: {args.n_heads}:{args.num_kv_heads} (Q:KV)")
    logging.info(f"Theoretical KV Cache Reduction: {(1 - args.num_kv_heads/args.n_heads)*100:.0f}%")
    logging.info("="*60)
    
    run_comparison_experiment(args)


if __name__ == "__main__":
    main()
