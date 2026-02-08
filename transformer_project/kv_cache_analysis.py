import torch
import numpy as np
import wandb
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    name: str
    num_query_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: torch.dtype = torch.float16
    
    @property
    def d_k(self) -> int:
        return self.d_model // self.num_query_heads
    
    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_query_heads
    
    @property
    def is_mqa(self) -> bool:
        return self.num_kv_heads == 1
    
    @property
    def queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads


class KVCacheAnalyzer:
    
    def __init__(self, configs: List[KVCacheConfig]):
        self.configs = configs
        self.analysis_results = {}
    
    def calculate_kv_cache_size(
        self, 
        config: KVCacheConfig, 
        batch_size: int, 
        seq_len: int
    ) -> Dict:
        bytes_per_element = 2 if config.dtype == torch.float16 else 4
        
        elements_per_layer = 2 * batch_size * config.num_kv_heads * seq_len * config.d_k
        
        total_elements = 3 * config.num_layers * elements_per_layer
        
        memory_bytes = total_elements * bytes_per_element
        memory_mb = memory_bytes / (1024 * 1024)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        
        return {
            "config_name": config.name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_kv_heads": config.num_kv_heads,
            "num_query_heads": config.num_query_heads,
            "total_elements": total_elements,
            "memory_bytes": memory_bytes,
            "memory_mb": memory_mb,
            "memory_gb": memory_gb,
            "dtype": str(config.dtype)
        }
    
    def compare_configs(
        self, 
        batch_size: int, 
        seq_len: int,
        baseline_name: str = "MHA"
    ) -> Dict:
        results = {}
        baseline_size = None
        
        for config in self.configs:
            cache_info = self.calculate_kv_cache_size(config, batch_size, seq_len)
            
            if config.name == baseline_name:
                baseline_size = cache_info["memory_mb"]
            
            results[config.name] = cache_info
        
        if baseline_size and baseline_size > 0:
            for name, info in results.items():
                info["relative_to_baseline"] = info["memory_mb"] / baseline_size
                info["savings_vs_baseline_percent"] = (1 - info["memory_mb"] / baseline_size) * 100
        
        return results
    
    def analyze_scaling(
        self, 
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64],
        seq_lengths: List[int] = [64, 128, 256, 512, 1024]
    ) -> Dict:
        scaling_results = {
            "by_batch_size": {},
            "by_seq_length": {}
        }
        
        fixed_seq = 256
        for config in self.configs:
            batch_scaling = []
            for bs in batch_sizes:
                info = self.calculate_kv_cache_size(config, bs, fixed_seq)
                batch_scaling.append({
                    "batch_size": bs,
                    "memory_mb": info["memory_mb"]
                })
            scaling_results["by_batch_size"][config.name] = batch_scaling
        
        fixed_batch = 8
        for config in self.configs:
            seq_scaling = []
            for sl in seq_lengths:
                info = self.calculate_kv_cache_size(config, fixed_batch, sl)
                seq_scaling.append({
                    "seq_len": sl,
                    "memory_mb": info["memory_mb"]
                })
            scaling_results["by_seq_length"][config.name] = seq_scaling
        
        return scaling_results
    
    def log_to_wandb(
        self, 
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        seq_lengths: List[int] = [64, 128, 256, 512]
    ):
        comparison = self.compare_configs(batch_size=8, seq_len=256)
        
        memory_table = wandb.Table(columns=[
            "attention_type", "num_kv_heads", "memory_mb", 
            "savings_percent", "relative_size"
        ])
        
        for name, info in comparison.items():
            memory_table.add_data(
                name,
                info["num_kv_heads"],
                info["memory_mb"],
                info.get("savings_vs_baseline_percent", 0),
                info.get("relative_to_baseline", 1.0)
            )
        
        wandb.log({
            "kv_analysis/memory_comparison_table": memory_table,
            "kv_analysis/memory_comparison_bar": wandb.plot.bar(
                memory_table, "attention_type", "memory_mb",
                title="KV Cache Memory by Attention Type (batch=8, seq=256)"
            ),
            "kv_analysis/savings_bar": wandb.plot.bar(
                memory_table, "attention_type", "savings_percent",
                title="KV Cache Memory Savings vs MHA (%)"
            )
        })
        
        scaling = self.analyze_scaling(batch_sizes, seq_lengths)
        
        # Batch size scaling plot
        batch_xs = [batch_sizes] * len(self.configs)
        batch_ys = [
            [d["memory_mb"] for d in scaling["by_batch_size"][c.name]]
            for c in self.configs
        ]
        
        wandb.log({
            "kv_analysis/batch_scaling_plot": wandb.plot.line_series(
                xs=batch_xs,
                ys=batch_ys,
                keys=[c.name for c in self.configs],
                title="KV Cache Memory vs Batch Size (seq_len=256)",
                xname="Batch Size"
            )
        })
        
        seq_xs = [seq_lengths] * len(self.configs)
        seq_ys = [
            [d["memory_mb"] for d in scaling["by_seq_length"][c.name]]
            for c in self.configs
        ]
        
        wandb.log({
            "kv_analysis/seq_scaling_plot": wandb.plot.line_series(
                xs=seq_xs,
                ys=seq_ys,
                keys=[c.name for c in self.configs],
                title="KV Cache Memory vs Sequence Length (batch=8)",
                xname="Sequence Length"
            )
        })
        
        scaling_table = wandb.Table(columns=[
            "attention_type", "batch_size", "seq_len", "memory_mb"
        ])
        
        for bs in batch_sizes:
            for sl in seq_lengths:
                for config in self.configs:
                    info = self.calculate_kv_cache_size(config, bs, sl)
                    scaling_table.add_data(config.name, bs, sl, info["memory_mb"])
        
        wandb.log({"kv_analysis/detailed_scaling_table": scaling_table})
        
        summary = {}
        for config in self.configs:
            info_small = self.calculate_kv_cache_size(config, 1, 64)
            info_medium = self.calculate_kv_cache_size(config, 8, 256)
            info_large = self.calculate_kv_cache_size(config, 32, 1024)
            
            summary[config.name] = {
                "small_scenario_mb": info_small["memory_mb"],
                "medium_scenario_mb": info_medium["memory_mb"],
                "large_scenario_mb": info_large["memory_mb"]
            }
            
            wandb.log({
                f"kv_analysis/{config.name}_small_mb": info_small["memory_mb"],
                f"kv_analysis/{config.name}_medium_mb": info_medium["memory_mb"],
                f"kv_analysis/{config.name}_large_mb": info_large["memory_mb"]
            })
        
        return summary


def create_standard_configs(
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6
) -> List[KVCacheConfig]:
    return [
        KVCacheConfig(
            name="MHA",
            num_query_heads=num_heads,
            num_kv_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers
        ),
        KVCacheConfig(
            name="GQA-4",
            num_query_heads=num_heads,
            num_kv_heads=4,
            d_model=d_model,
            num_layers=num_layers
        ),
        KVCacheConfig(
            name="GQA-2",
            num_query_heads=num_heads,
            num_kv_heads=2,
            d_model=d_model,
            num_layers=num_layers
        ),
        KVCacheConfig(
            name="MQA",
            num_query_heads=num_heads,
            num_kv_heads=1,
            d_model=d_model,
            num_layers=num_layers
        )
    ]


def run_kv_cache_analysis(
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    log_to_wandb: bool = True
):
    configs = create_standard_configs(d_model, num_heads, num_layers)
    analyzer = KVCacheAnalyzer(configs)
    
    print("="*60)
    print("KV Cache Memory Analysis")
    print("="*60)
    print(f"Model Config: d_model={d_model}, heads={num_heads}, layers={num_layers}")
    print()
    
    comparison = analyzer.compare_configs(batch_size=8, seq_len=256)
    
    print("Memory Comparison (batch=8, seq_len=256):")
    print("-"*60)
    print(f"{'Type':<10} {'KV Heads':<10} {'Memory (MB)':<15} {'Savings':<10}")
    print("-"*60)
    
    for name, info in comparison.items():
        savings = info.get('savings_vs_baseline_percent', 0)
        print(f"{name:<10} {info['num_kv_heads']:<10} {info['memory_mb']:<15.2f} {savings:<10.1f}%")
    
    print()
    
    if log_to_wandb:
        analyzer.log_to_wandb()
        print("✓ Analysis logged to wandb")
    
    return analyzer, comparison


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "kv-cache-analysis"),
            name="kv-cache-detailed-analysis"
        )
    
    run_kv_cache_analysis(log_to_wandb=bool(api_key))
    
    if api_key:
        wandb.finish()
