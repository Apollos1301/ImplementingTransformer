import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import sacrebleu
import time
import yaml
import warnings
import wandb
from tabulate import tabulate

from transformer_project.modelling import Transformer
from transformer_project.optimization import TransformerLRScheduler, build_optimizer
from transformer_project.dataset import TranslationDataset
from transformer_project.text.transformer_tokenizer import TokenizerTrainer

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def translate_sentence(model, tokenizer, src_text, device, max_len=100):
    model.eval()
    
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
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


def evaluate_bleu(model, tokenizer, test_data, device, num_samples=None):
    if num_samples:
        test_data = test_data[:num_samples]
    
    predictions = []
    references = []
    
    model.eval()
    for item in tqdm(test_data, desc="BLEU Eval", leave=False):
        src_text, ref_text = item
        pred_text = translate_sentence(model, tokenizer, src_text, device)
        predictions.append(pred_text)
        references.append(ref_text)
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions, references


def main():
    parser = argparse.ArgumentParser(description="Train Transformer for Translation")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Epoch to start from (0-based)")
    parser.add_argument("--epochs", type=int, default=None, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--positional_encoding", type=str, default=None, choices=["sinusoidal", "rope"])
    parser.add_argument("--small", action="store_true", help="Use small model for quick testing")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {
            "project_name": "transformer-wmt17",
            "run_name": "wmt17-run",
            "weights_dir": "./weights",
            "batch_size": 64,
            "epochs": 5,
            "learning_rate": 1.0,
            "max_seq_length": 100,
            "vocab_size": 32000,
            "d_model": 512,
            "n_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "mixed_precision": True,
            "dataset_name": "wmt17",
            "dataset_config": "de-en",
            "val_split_size": 0.01,
            "warmup_steps": 4000,
            "use_scheduler": True,
            "use_wandb": True,
            "positional_encoding_type": "sinusoidal"
        }

    if args.resume_from:
        config['resume_from'] = args.resume_from
    if args.start_epoch is not None:
        config['start_epoch'] = args.start_epoch
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.positional_encoding:
        config['positional_encoding_type'] = args.positional_encoding

    if args.small:
        print("Using SMALL model configuration for quick testing...")
        config['batch_size'] = 32
        config['epochs'] = 3
        config['d_model'] = 128
        config['n_heads'] = 4
        config['num_encoder_layers'] = 2
        config['num_decoder_layers'] = 2
        config['dim_feedforward'] = 512
        config['warmup_steps'] = 500
        config['max_seq_length'] = 64

    start_epoch = config.get('start_epoch', 0)
    resume_from = config.get('resume_from', None)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = config.get("mixed_precision", False) and (DEVICE == "cuda")

    PROJECT_NAME = config.get("project_name", "transformer")
    RUN_NAME = config.get("run_name", "training-run")
    BATCH_SIZE = config["batch_size"]
    EPOCHS = config["epochs"]
    LEARNING_RATE = config.get("learning_rate", 1.0)
    MAX_SEQ_LENGTH = config["max_seq_length"]
    VOCAB_SIZE = config["vocab_size"]
    D_MODEL = config["d_model"]
    N_HEADS = config["n_heads"]
    NUM_ENCODER_LAYERS = config["num_encoder_layers"]
    NUM_DECODER_LAYERS = config["num_decoder_layers"]
    DIM_FEEDFORWARD = config["dim_feedforward"]
    DROPOUT = config["dropout"]
    POSITIONAL_ENCODING_TYPE = config.get("positional_encoding_type", "sinusoidal")
    DATASET_NAME = config.get("dataset_name", "wmt17")
    DATASET_CONFIG = config.get("dataset_config", "de-en")
    WARMUP_STEPS = config.get("warmup_steps", 4000)
    USE_SCHEDULER = config.get("use_scheduler", True)
    USE_WANDB = config.get("use_wandb", False)
    
    USE_GQA = config.get("use_gqa", False)
    NUM_KV_HEADS = config.get("num_kv_heads", N_HEADS)

    if USE_WANDB:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        
        wandb_project = os.environ.get("WANDB_PROJECT", PROJECT_NAME)
        wandb_entity = os.environ.get("WANDB_ENTITY", None)
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=RUN_NAME,
            config=config
        )
        print(f"WandB initialized: project={wandb_project}, entity={wandb_entity}")
    else:
        wandb.init(mode="disabled")

    print("="*60)
    print(f"Run: {RUN_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {MIXED_PRECISION}")
    print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"Positional Encoding: {POSITIONAL_ENCODING_TYPE}")
    print(f"Attention Type: {'GQA' if USE_GQA else 'MHA'}" + (f" (Q={N_HEADS}, KV={NUM_KV_HEADS})" if USE_GQA else f" (heads={N_HEADS})"))
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max Seq Length: {MAX_SEQ_LENGTH}")
    print(f"Model: d_model={D_MODEL}, n_heads={N_HEADS}, layers={NUM_ENCODER_LAYERS}")
    print("="*60)

    print(f"\nLoading {DATASET_NAME} dataset...")
    
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_huggingface")
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"Error loading {DATASET_NAME}: {e}")
        return

    train_data = dataset['train']
    
    if 'validation' in dataset:
        val_data = dataset['validation']
    else:
        print("No validation split found, splitting train...")
        split = train_data.train_test_split(test_size=config.get("val_split_size", 0.01), seed=42)
        train_data = split['train']
        val_data = split['test']

    test_data = dataset.get('test', val_data)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    print("\nPreparing tokenizer...")
    trainer = TokenizerTrainer(vocab_size=VOCAB_SIZE)
    
    def corpus_iterator():
        batch_size = 10000
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            for translation in batch["translation"]:
                yield translation["de"]
                yield translation["en"]

    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer_data")
    tokenizer = trainer.train_and_get_tokenizer(corpus_iterator(), save_path=tokenizer_path)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"PAD: {tokenizer.pad_token_id}, BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    print("\nCreating datasets...")
    
    def process_data(data):
        raw_list = []
        for item in data:
            trans = item['translation']
            raw_list.append((trans['de'], trans['en']))
        return raw_list

    train_raw = process_data(train_data)
    val_raw = process_data(val_data)
    test_raw = process_data(test_data)

    train_ds = TranslationDataset(train_raw, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
    val_ds = TranslationDataset(val_raw, tokenizer, max_seq_length=MAX_SEQ_LENGTH)

    val_bleu_subset = val_raw[:100]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\nInitializing model...")
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LENGTH,
        positional_encoding_type=POSITIONAL_ENCODING_TYPE,
        use_gqa=USE_GQA,
        num_kv_heads=NUM_KV_HEADS
    )
    model.to(DEVICE)
    
    print(f"Attention Type: {model.get_attention_type()}")
    
    kv_info = model.get_kv_cache_info(batch_size=BATCH_SIZE, seq_len=MAX_SEQ_LENGTH)
    print(f"KV Cache Memory Savings: {kv_info['memory_savings_percent']:.1f}%")
    wandb.log({
        "kv_cache/memory_reduction_ratio": kv_info["memory_reduction_ratio"],
        "kv_cache/memory_savings_percent": kv_info["memory_savings_percent"],
        "attention_type": "GQA" if USE_GQA else "MHA",
        "num_kv_heads": NUM_KV_HEADS,
        "num_query_heads": N_HEADS
    })

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    wandb.log({"model_parameters": n_params})

    if USE_SCHEDULER:
        print(f"Using TransformerLRScheduler with {WARMUP_STEPS} warmup steps.")
        optimizer = build_optimizer(model, learning_rate=1.0, weight_decay=0.1)
        scheduler = TransformerLRScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)
    else:
        print(f"Using Fixed Learning Rate: {LEARNING_RATE}")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    amp_dtype = torch.float16
    use_scaler = False
    
    if MIXED_PRECISION:
        if torch.cuda.is_bf16_supported():
            print("Using BFloat16 (no scaler needed).")
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            print("Using Float16 with GradScaler.")
            amp_dtype = torch.float16
            use_scaler = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print("\nStarting training...")
    
    weights_dir = config.get("weights_dir", "./weights")
    os.makedirs(weights_dir, exist_ok=True)
    print(f"Saving checkpoints to {weights_dir}")

    global_step = start_epoch * len(train_loader)
    best_val_loss = float('inf')
    best_bleu_score = 0.0
    start_time = time.time()

    def run_validation(current_epoch, step):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                encoder_input = batch['encoder_input'].to(DEVICE)
                decoder_input = batch['decoder_input'].to(DEVICE)
                label = batch['label'].to(DEVICE)
                src_mask = batch['src_mask'].to(DEVICE)
                tgt_mask = batch['tgt_mask'].to(DEVICE)
                
                with torch.cuda.amp.autocast(enabled=MIXED_PRECISION, dtype=amp_dtype):
                    output = model(encoder_input, decoder_input, src_mask, tgt_mask)
                    output = output.view(-1, output.shape[-1])
                    label = label.view(-1)
                    loss = criterion(output, label)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": current_epoch + 1}, step=step)
        model.train()
        return avg_val_loss

    def run_bleu_eval(step):
        if not val_bleu_subset:
            return 0.0
        print(f"Running BLEU validation...")
        bleu_score, _, _ = evaluate_bleu(model, tokenizer, val_bleu_subset, DEVICE)
        print(f"BLEU Score: {bleu_score:.2f}")
        wandb.log({"bleu_score": bleu_score}, step=step)
        return bleu_score

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            encoder_input = batch['encoder_input'].to(DEVICE)
            decoder_input = batch['decoder_input'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            tgt_mask = batch['tgt_mask'].to(DEVICE)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION, dtype=amp_dtype):
                output = model(encoder_input, decoder_input, src_mask, tgt_mask)
                output = output.view(-1, output.shape[-1])
                label = label.view(-1)
                loss = criterion(output, label)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler:
                scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            
            current_lr = scheduler.get_last_lr()[0] if scheduler else LEARNING_RATE
            progress_bar.set_postfix(loss=loss_val, lr=f"{current_lr:.6f}")
            wandb.log({"train_loss": loss_val, "epoch": epoch + 1, "lr": current_lr}, step=global_step)
            global_step += 1

            if global_step > 0 and global_step % 5000 == 0:
                v_loss = run_validation(epoch, global_step)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    save_path = os.path.join(weights_dir, "best_val_loss.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best validation loss checkpoint")
            
            if global_step > 0 and global_step % 20000 == 0:
                bleu = run_bleu_eval(global_step)
                if bleu > best_bleu_score:
                    best_bleu_score = bleu
                    save_path = os.path.join(weights_dir, "best_bleu.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best BLEU checkpoint")

            if global_step > 0 and global_step % 50000 == 0:
                save_path = os.path.join(weights_dir, f"step_{global_step}.pt")
                torch.save(model.state_dict(), save_path)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")
        wandb.log({"epoch_train_loss": avg_loss, "epoch": epoch + 1})
        
        v_loss = run_validation(epoch, global_step)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_path = os.path.join(weights_dir, "best_val_loss.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best validation loss checkpoint")
        
        save_path = os.path.join(weights_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time
    total_time_str = f"{int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s"
    
    final_bleu = run_bleu_eval(global_step)
    if final_bleu > best_bleu_score:
        best_bleu_score = final_bleu
        save_path = os.path.join(weights_dir, "best_bleu.pt")
        torch.save(model.state_dict(), save_path)

    summary_data = [
        ["Total Training Time", total_time_str],
        ["Best Validation Loss", f"{best_val_loss:.4f}"],
        ["Best BLEU Score", f"{best_bleu_score:.2f}"],
        ["Final Training Loss", f"{avg_loss:.4f}"],
        ["Global Steps", str(global_step)]
    ]

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="pretty"))
    print("="*60)

    print("\nSAMPLE TRANSLATIONS")
    print("="*60)
    
    model.load_state_dict(torch.load(os.path.join(weights_dir, "best_bleu.pt"), map_location=DEVICE))
    model.eval()
    
    for i, (src, ref) in enumerate(test_raw[:5]):
        pred = translate_sentence(model, tokenizer, src, DEVICE)
        print(f"\n--- Example {i+1} ---")
        print(f"Source (DE):    {src[:100]}...")
        print(f"Reference (EN): {ref[:100]}...")
        print(f"Prediction:     {pred[:100]}...")

    wandb.finish()


if __name__ == "__main__":
    main()
