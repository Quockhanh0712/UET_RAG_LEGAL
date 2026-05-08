#!/usr/bin/env python3
"""
Script to recalculate metrics from saved benchmark CSV files.
Useful when benchmark failed at the end (e.g., BERTScore error).

Usage:
    python scripts/recalculate_benchmark.py benchmark_results/benchmark_details_XXXXX.csv
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_bert_scores(predictions, ground_truths, batch_size=64):
    """Compute BERTScore in batches"""
    try:
        from bert_score import BERTScorer
        
        logger.info("[METRICS] Loading BERTScore model...")
        scorer = BERTScorer(
            lang="vi",
            rescale_with_baseline=False,  # No baseline for Vietnamese
            model_type="bert-base-multilingual-cased"
        )
        logger.info("[METRICS] BERTScore ready")
        
        all_P, all_R, all_F1 = [], [], []
        
        for i in range(0, len(predictions), batch_size):
            batch_pred = predictions[i:i+batch_size]
            batch_gold = ground_truths[i:i+batch_size]
            
            P, R, F1 = scorer.score(batch_pred, batch_gold)
            all_P.extend(P.tolist())
            all_R.extend(R.tolist())
            all_F1.extend(F1.tolist())
            
            logger.info(f"[METRICS] BERTScore: {min(i+batch_size, len(predictions))}/{len(predictions)}")
        
        return np.mean(all_P), np.mean(all_R), np.mean(all_F1)
        
    except Exception as e:
        logger.error(f"[METRICS] BERTScore failed: {e}")
        return 0.0, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Recalculate benchmark metrics from CSV")
    parser.add_argument("csv_path", type=str, help="Path to benchmark_details CSV file")
    parser.add_argument("--with-bert", action="store_true", help="Include BERTScore calculation")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return
    
    logger.info(f"[RECALC] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"[RECALC] Loaded {len(df)} samples")
    
    # Check columns
    logger.info(f"[RECALC] Columns: {list(df.columns)}")
    
    # Compute retrieval metrics
    ret_cols = ['mrr', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 'precision@1', 'precision@3', 'precision@5']
    ret_metrics = {}
    for col in ret_cols:
        if col in df.columns:
            ret_metrics[col] = df[col].mean()
    
    # Compute generation metrics
    gen_cols = ['exact_match', 'f1_score', 'bleu_1', 'bleu_2', 'bleu_4', 'rouge_1', 'rouge_2', 'rouge_l', 'length_ratio', 'extractive']
    gen_metrics = {}
    for col in gen_cols:
        if col in df.columns:
            gen_metrics[col] = df[col].mean()
    
    # Compute timing
    if 'retrieval_time' in df.columns:
        ret_metrics['avg_time_ms'] = df['retrieval_time'].mean() * 1000
    if 'generation_time' in df.columns:
        gen_metrics['avg_time_ms'] = df['generation_time'].mean() * 1000
    
    # BERTScore if requested
    bert_p, bert_r, bert_f1 = 0.0, 0.0, 0.0
    if args.with_bert:
        predictions = df['generated_answer'].fillna('').tolist()
        ground_truths = df['ground_truth'].fillna('').tolist()
        
        # Filter empty
        valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if p.strip() and g.strip()]
        if valid_pairs:
            preds, golds = zip(*valid_pairs)
            bert_p, bert_r, bert_f1 = compute_bert_scores(list(preds), list(golds))
    
    # Print results
    print("\n" + "="*70)
    print("📊 RECALCULATED BENCHMARK RESULTS")
    print("="*70)
    print(f"\n📁 File: {csv_path}")
    print(f"📝 Samples: {len(df)}")
    
    print("\n" + "-"*70)
    print("🔍 RETRIEVAL METRICS")
    print("-"*70)
    for key, val in ret_metrics.items():
        if 'time' in key:
            print(f"  {key:15}: {val:.1f}")
        else:
            print(f"  {key:15}: {val:.4f}")
    
    print("\n" + "-"*70)
    print("✍️  GENERATION METRICS")
    print("-"*70)
    for key, val in gen_metrics.items():
        if 'time' in key:
            print(f"  {key:15}: {val:.1f}")
        elif 'ratio' in key:
            print(f"  {key:15}: {val:.2f}")
        elif 'extractive' in key:
            print(f"  {key:15}: {val:.2%}")
        else:
            print(f"  {key:15}: {val:.4f}")
    
    if args.with_bert:
        print(f"\n  BERTScore P:    {bert_p:.4f}")
        print(f"  BERTScore R:    {bert_r:.4f}")
        print(f"  BERTScore F1:   {bert_f1:.4f}")
    
    print("\n" + "="*70)
    
    # Save updated JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "source_file": str(csv_path),
        "total_samples": len(df),
        "retrieval": ret_metrics,
        "generation": gen_metrics,
    }
    if args.with_bert:
        output["generation"]["bert_score_p"] = bert_p
        output["generation"]["bert_score_r"] = bert_r
        output["generation"]["bert_score_f1"] = bert_f1
    
    output_path = csv_path.parent / f"recalculated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[RECALC] Saved to {output_path}")


if __name__ == "__main__":
    main()
