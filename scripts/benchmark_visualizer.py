#!/usr/bin/env python3
"""
Benchmark Visualizer for ALQAC Evaluation

Generates beautiful charts and HTML report from benchmark results:
- Radar chart for overall metrics
- Bar charts for retrieval vs generation
- Distribution plots for per-sample performance
- Heatmaps for error analysis
- Interactive HTML report

Usage:
    from scripts.benchmark_visualizer import BenchmarkVisualizer
    visualizer = BenchmarkVisualizer(result, samples, output_dir)
    visualizer.generate_all()
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Colors
COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed', 
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'retrieval': '#3b82f6',
    'generation': '#8b5cf6',
    'background': '#f8fafc'
}


class BenchmarkVisualizer:
    """
    Generates visualizations from benchmark results
    """
    
    def __init__(
        self,
        result: Any,  # BenchmarkResult
        samples: List[Any],  # List[BenchmarkSample]
        output_dir: str = "benchmark_results"
    ):
        self.result = result
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures directory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all(self):
        """Generate all visualizations and HTML report"""
        print("📊 Generating visualizations...")
        
        # Generate charts
        self.plot_metrics_radar()
        self.plot_retrieval_bars()
        self.plot_generation_bars()
        self.plot_time_distribution()
        self.plot_metric_distributions()
        self.plot_recall_curve()
        self.plot_score_heatmap()
        self.plot_error_analysis()
        
        # Generate HTML report
        self.generate_html_report()
        
        print(f"✅ All visualizations saved to {self.output_dir}")
    
    def plot_metrics_radar(self):
        """Radar chart showing overall performance"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
        
        # Retrieval metrics radar
        ret = self.result.retrieval
        ret_metrics = {
            'MRR': ret.mrr,
            'R@1': ret.recall_at_1,
            'R@3': ret.recall_at_3,
            'R@5': ret.recall_at_5,
            'R@10': ret.recall_at_10,
            'P@1': ret.precision_at_1,
            'P@3': ret.precision_at_3,
            'P@5': ret.precision_at_5,
        }
        
        self._draw_radar(axes[0], ret_metrics, "🔍 Retrieval Metrics", COLORS['retrieval'])
        
        # Generation metrics radar
        gen = self.result.generation
        gen_metrics = {
            'EM': gen.exact_match,
            'F1': gen.f1_score,
            'BLEU-1': gen.bleu_1,
            'BLEU-2': gen.bleu_2,
            'ROUGE-1': gen.rouge_1,
            'ROUGE-L': gen.rouge_l,
            'BERT-F1': gen.bert_score_f1,
            'Extractive': gen.extractive_rate,
        }
        
        self._draw_radar(axes[1], gen_metrics, "✍️ Generation Metrics", COLORS['generation'])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "radar_metrics.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def _draw_radar(self, ax, metrics: Dict[str, float], title: str, color: str):
        """Draw a single radar chart"""
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Complete the loop
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]
        
        # Plot
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=2, marker='o', markersize=6)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1], size=9)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        
        # Add value annotations
        for angle, value, label in zip(angles[:-1], values[:-1], labels[:-1]):
            ax.annotate(
                f'{value:.2f}',
                xy=(angle, value),
                xytext=(angle, value + 0.08),
                ha='center',
                fontsize=8,
                fontweight='bold'
            )
    
    def plot_retrieval_bars(self):
        """Bar chart for retrieval metrics"""
        ret = self.result.retrieval
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = {
            'MRR': ret.mrr,
            'Recall@1': ret.recall_at_1,
            'Recall@3': ret.recall_at_3,
            'Recall@5': ret.recall_at_5,
            'Recall@10': ret.recall_at_10,
            'Precision@1': ret.precision_at_1,
            'Precision@3': ret.precision_at_3,
            'Precision@5': ret.precision_at_5,
        }
        
        x = np.arange(len(metrics))
        bars = ax.bar(x, list(metrics.values()), color=COLORS['retrieval'], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title('🔍 Retrieval Performance Metrics', fontsize=14, fontweight='bold', pad=15)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (0.8)')
        ax.legend(loc='upper right')
        
        # Add grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "retrieval_bars.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_generation_bars(self):
        """Bar chart for generation metrics"""
        gen = self.result.generation
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = {
            'Exact Match': gen.exact_match,
            'F1 Score': gen.f1_score,
            'BLEU-1': gen.bleu_1,
            'BLEU-2': gen.bleu_2,
            'BLEU-4': gen.bleu_4,
            'ROUGE-1': gen.rouge_1,
            'ROUGE-2': gen.rouge_2,
            'ROUGE-L': gen.rouge_l,
            'BERT P': gen.bert_score_p,
            'BERT R': gen.bert_score_r,
            'BERT F1': gen.bert_score_f1,
        }
        
        x = np.arange(len(metrics))
        colors = [COLORS['generation'] if i < 8 else COLORS['info'] for i in range(len(metrics))]
        bars = ax.bar(x, list(metrics.values()), color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                fontsize=9,
                fontweight='bold'
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title('✍️ Generation Performance Metrics', fontsize=14, fontweight='bold', pad=15)
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax.legend(loc='upper right')
        
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "generation_bars.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_time_distribution(self):
        """Distribution of retrieval and generation times"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ret_times = [s.retrieval_time * 1000 for s in self.samples]  # Convert to ms
        gen_times = [s.generation_time * 1000 for s in self.samples]
        
        # Retrieval time
        sns.histplot(ret_times, bins=30, kde=True, ax=axes[0], color=COLORS['retrieval'], alpha=0.7)
        axes[0].axvline(x=np.mean(ret_times), color='red', linestyle='--', label=f'Mean: {np.mean(ret_times):.1f}ms')
        axes[0].axvline(x=np.median(ret_times), color='green', linestyle='--', label=f'Median: {np.median(ret_times):.1f}ms')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('🔍 Retrieval Time Distribution', fontweight='bold')
        axes[0].legend()
        
        # Generation time
        if any(t > 0 for t in gen_times):
            sns.histplot([t for t in gen_times if t > 0], bins=30, kde=True, ax=axes[1], color=COLORS['generation'], alpha=0.7)
            valid_gen = [t for t in gen_times if t > 0]
            axes[1].axvline(x=np.mean(valid_gen), color='red', linestyle='--', label=f'Mean: {np.mean(valid_gen):.1f}ms')
            axes[1].axvline(x=np.median(valid_gen), color='green', linestyle='--', label=f'Median: {np.median(valid_gen):.1f}ms')
            axes[1].legend()
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('✍️ Generation Time Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "time_distribution.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_metric_distributions(self):
        """Distribution of per-sample metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract per-sample metrics
        mrr_scores = [s.retrieval_metrics.get('mrr', 0) for s in self.samples]
        recall5_scores = [s.retrieval_metrics.get('recall@5', 0) for s in self.samples]
        em_scores = [s.generation_metrics.get('exact_match', 0) for s in self.samples]
        f1_scores = [s.generation_metrics.get('f1_score', 0) for s in self.samples]
        bleu1_scores = [s.generation_metrics.get('bleu_1', 0) for s in self.samples]
        rouge1_scores = [s.generation_metrics.get('rouge_1', 0) for s in self.samples]
        
        metrics_data = [
            (mrr_scores, 'MRR', COLORS['retrieval']),
            (recall5_scores, 'Recall@5', COLORS['retrieval']),
            (em_scores, 'Exact Match', COLORS['generation']),
            (f1_scores, 'F1 Score', COLORS['generation']),
            (bleu1_scores, 'BLEU-1', COLORS['info']),
            (rouge1_scores, 'ROUGE-1', COLORS['success']),
        ]
        
        for ax, (data, name, color) in zip(axes.flatten(), metrics_data):
            sns.histplot(data, bins=20, kde=True, ax=ax, color=color, alpha=0.7)
            ax.axvline(x=np.mean(data), color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution (μ={np.mean(data):.3f})', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "metric_distributions.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_recall_curve(self):
        """Recall@K curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ret = self.result.retrieval
        k_values = [1, 3, 5, 10]
        recall_values = [ret.recall_at_1, ret.recall_at_3, ret.recall_at_5, ret.recall_at_10]
        precision_values = [ret.precision_at_1, ret.precision_at_3, ret.precision_at_5, 0]  # No P@10
        
        ax.plot(k_values, recall_values, 'o-', color=COLORS['retrieval'], linewidth=2, markersize=10, label='Recall@K')
        ax.plot(k_values[:3], precision_values[:3], 's--', color=COLORS['warning'], linewidth=2, markersize=10, label='Precision@K')
        
        # Fill area under recall curve
        ax.fill_between(k_values, recall_values, alpha=0.2, color=COLORS['retrieval'])
        
        # Add annotations
        for k, r in zip(k_values, recall_values):
            ax.annotate(f'{r:.3f}', (k, r), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')
        
        ax.set_xlabel('K (Top-K Documents)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('📈 Recall & Precision at Different K Values', fontsize=14, fontweight='bold')
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "recall_curve.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_score_heatmap(self):
        """Heatmap of metric correlations"""
        # Extract per-sample data
        data = []
        for s in self.samples:
            row = {
                'MRR': s.retrieval_metrics.get('mrr', 0),
                'Recall@5': s.retrieval_metrics.get('recall@5', 0),
                'Precision@3': s.retrieval_metrics.get('precision@3', 0),
                'EM': s.generation_metrics.get('exact_match', 0),
                'F1': s.generation_metrics.get('f1_score', 0),
                'BLEU-1': s.generation_metrics.get('bleu_1', 0),
                'ROUGE-1': s.generation_metrics.get('rouge_1', 0),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
                cbar_kws={'shrink': 0.8}
            )
            
            ax.set_title('🔗 Metric Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / "correlation_heatmap.png", bbox_inches='tight', dpi=150)
            plt.close()
    
    def plot_error_analysis(self):
        """Error analysis: samples with low scores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Retrieval failures
        mrr_scores = [(i, s.retrieval_metrics.get('mrr', 0)) for i, s in enumerate(self.samples)]
        sorted_mrr = sorted(mrr_scores, key=lambda x: x[1])[:20]  # Bottom 20
        
        indices, scores = zip(*sorted_mrr) if sorted_mrr else ([], [])
        axes[0].barh(range(len(indices)), scores, color=COLORS['danger'], alpha=0.7)
        axes[0].set_yticks(range(len(indices)))
        axes[0].set_yticklabels([f'Sample {i}' for i in indices])
        axes[0].set_xlabel('MRR Score')
        axes[0].set_title('🔴 Top 20 Retrieval Failures (Lowest MRR)', fontweight='bold')
        axes[0].invert_yaxis()
        
        # Generation failures  
        f1_scores = [(i, s.generation_metrics.get('f1_score', 0)) for i, s in enumerate(self.samples)]
        sorted_f1 = sorted(f1_scores, key=lambda x: x[1])[:20]
        
        indices, scores = zip(*sorted_f1) if sorted_f1 else ([], [])
        axes[1].barh(range(len(indices)), scores, color=COLORS['warning'], alpha=0.7)
        axes[1].set_yticks(range(len(indices)))
        axes[1].set_yticklabels([f'Sample {i}' for i in indices])
        axes[1].set_xlabel('F1 Score')
        axes[1].set_title('🟠 Top 20 Generation Failures (Lowest F1)', fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "error_analysis.png", bbox_inches='tight', dpi=150)
        plt.close()
    
    def generate_html_report(self):
        """Generate interactive HTML report"""
        ret = self.result.retrieval
        gen = self.result.generation
        
        html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALQAC Benchmark Report - Legal RAG System</title>
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #7c3aed;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-secondary: #64748b;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 3rem 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header p {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        .meta-info {{
            display: flex;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--bg);
        }}
        
        .card-header h3 {{
            font-size: 1.1rem;
            color: var(--text);
        }}
        
        .card-icon {{
            font-size: 1.5rem;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}
        
        .metric {{
            text-align: center;
            padding: 0.75rem;
            background: var(--bg);
            border-radius: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .metric-label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}
        
        .score-bar {{
            height: 8px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        
        .score-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease-out;
        }}
        
        .score-bar-fill.good {{ background: var(--success); }}
        .score-bar-fill.warning {{ background: var(--warning); }}
        .score-bar-fill.danger {{ background: var(--danger); }}
        
        .chart-container {{
            background: var(--card-bg);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        
        .chart-container h3 {{
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 0.5rem;
        }}
        
        .highlight {{
            background: linear-gradient(135deg, var(--success) 0%, #34d399 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .highlight h2 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        .summary-table th, .summary-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg);
        }}
        
        .summary-table th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .summary-table tr:hover {{
            background: var(--bg);
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            header h1 {{ font-size: 1.8rem; }}
            .meta-info {{ flex-direction: column; gap: 0.5rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 ALQAC Benchmark Report</h1>
            <p>Comprehensive evaluation of Legal RAG System on Vietnamese Legal QA Dataset</p>
            <div class="meta-info">
                <div class="meta-item">📅 {self.result.timestamp}</div>
                <div class="meta-item">📁 {self.result.dataset}</div>
                <div class="meta-item">📝 {self.result.evaluated_samples} / {self.result.total_samples} samples</div>
                <div class="meta-item">🔧 Top-K: {self.result.config.get('top_k', 10)}</div>
            </div>
        </header>
        
        <!-- Overall Score -->
        <div class="highlight">
            <h2>🎯 Overall Performance Score</h2>
            <p style="font-size: 3rem; font-weight: bold;">
                {((ret.mrr + ret.recall_at_5 + gen.f1_score + gen.exact_match) / 4 * 100):.1f}%
            </p>
            <p>Combined metric average (MRR + Recall@5 + F1 + EM) / 4</p>
        </div>
        
        <!-- Metric Cards -->
        <div class="grid">
            <!-- Retrieval Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">🔍</span>
                    <h3>Retrieval Metrics</h3>
                </div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">{ret.mrr:.3f}</div>
                        <div class="metric-label">MRR</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if ret.mrr > 0.7 else 'warning' if ret.mrr > 0.5 else 'danger'}" style="width: {ret.mrr*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{ret.recall_at_1:.3f}</div>
                        <div class="metric-label">Recall@1</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if ret.recall_at_1 > 0.7 else 'warning' if ret.recall_at_1 > 0.5 else 'danger'}" style="width: {ret.recall_at_1*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{ret.recall_at_5:.3f}</div>
                        <div class="metric-label">Recall@5</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if ret.recall_at_5 > 0.7 else 'warning' if ret.recall_at_5 > 0.5 else 'danger'}" style="width: {ret.recall_at_5*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{ret.recall_at_10:.3f}</div>
                        <div class="metric-label">Recall@10</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if ret.recall_at_10 > 0.7 else 'warning' if ret.recall_at_10 > 0.5 else 'danger'}" style="width: {ret.recall_at_10*100}%"></div>
                        </div>
                    </div>
                </div>
                <p style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
                    ⏱️ Avg retrieval time: <strong>{ret.avg_retrieval_time*1000:.1f}ms</strong>
                </p>
            </div>
            
            <!-- Generation Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">✍️</span>
                    <h3>Generation Metrics</h3>
                </div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">{gen.exact_match:.3f}</div>
                        <div class="metric-label">Exact Match</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if gen.exact_match > 0.6 else 'warning' if gen.exact_match > 0.4 else 'danger'}" style="width: {gen.exact_match*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.f1_score:.3f}</div>
                        <div class="metric-label">F1 Score</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if gen.f1_score > 0.6 else 'warning' if gen.f1_score > 0.4 else 'danger'}" style="width: {gen.f1_score*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.bleu_1:.3f}</div>
                        <div class="metric-label">BLEU-1</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if gen.bleu_1 > 0.5 else 'warning' if gen.bleu_1 > 0.3 else 'danger'}" style="width: {gen.bleu_1*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.rouge_l:.3f}</div>
                        <div class="metric-label">ROUGE-L</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if gen.rouge_l > 0.5 else 'warning' if gen.rouge_l > 0.3 else 'danger'}" style="width: {gen.rouge_l*100}%"></div>
                        </div>
                    </div>
                </div>
                <p style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
                    ⏱️ Avg generation time: <strong>{gen.avg_generation_time*1000:.1f}ms</strong>
                </p>
            </div>
            
            <!-- BERTScore Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">🧠</span>
                    <h3>Semantic Metrics (BERTScore)</h3>
                </div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">{gen.bert_score_p:.3f}</div>
                        <div class="metric-label">Precision</div>
                        <div class="score-bar">
                            <div class="score-bar-fill good" style="width: {gen.bert_score_p*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.bert_score_r:.3f}</div>
                        <div class="metric-label">Recall</div>
                        <div class="score-bar">
                            <div class="score-bar-fill good" style="width: {gen.bert_score_r*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.bert_score_f1:.3f}</div>
                        <div class="metric-label">F1</div>
                        <div class="score-bar">
                            <div class="score-bar-fill {'good' if gen.bert_score_f1 > 0.6 else 'warning'}" style="width: {gen.bert_score_f1*100}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{gen.extractive_rate:.1%}</div>
                        <div class="metric-label">Extractive Rate</div>
                        <div class="score-bar">
                            <div class="score-bar-fill good" style="width: {gen.extractive_rate*100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Additional Stats Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">📈</span>
                    <h3>Additional Statistics</h3>
                </div>
                <table class="summary-table">
                    <tr>
                        <td>Precision@1</td>
                        <td><strong>{ret.precision_at_1:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>Precision@3</td>
                        <td><strong>{ret.precision_at_3:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>Precision@5</td>
                        <td><strong>{ret.precision_at_5:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>BLEU-2</td>
                        <td><strong>{gen.bleu_2:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>BLEU-4</td>
                        <td><strong>{gen.bleu_4:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>ROUGE-1</td>
                        <td><strong>{gen.rouge_1:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>ROUGE-2</td>
                        <td><strong>{gen.rouge_2:.3f}</strong></td>
                    </tr>
                    <tr>
                        <td>Answer Length Ratio</td>
                        <td><strong>{gen.answer_length_ratio:.2f}x</strong></td>
                    </tr>
                </table>
            </div>
        </div>
        
        <!-- Charts Section -->
        <h2 style="margin-bottom: 1.5rem;">📊 Detailed Visualizations</h2>
        
        <div class="chart-container">
            <h3>🎯 Radar Chart - Overall Metrics</h3>
            <img src="figures/radar_metrics.png" alt="Radar Metrics" onerror="this.style.display='none'">
        </div>
        
        <div class="grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="chart-container">
                <h3>🔍 Retrieval Performance</h3>
                <img src="figures/retrieval_bars.png" alt="Retrieval Bars" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h3>✍️ Generation Performance</h3>
                <img src="figures/generation_bars.png" alt="Generation Bars" onerror="this.style.display='none'">
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📈 Recall & Precision Curve</h3>
            <img src="figures/recall_curve.png" alt="Recall Curve" onerror="this.style.display='none'">
        </div>
        
        <div class="grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="chart-container">
                <h3>⏱️ Time Distribution</h3>
                <img src="figures/time_distribution.png" alt="Time Distribution" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h3>🔗 Metric Correlations</h3>
                <img src="figures/correlation_heatmap.png" alt="Correlation Heatmap" onerror="this.style.display='none'">
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📊 Metric Distributions</h3>
            <img src="figures/metric_distributions.png" alt="Metric Distributions" onerror="this.style.display='none'">
        </div>
        
        <div class="chart-container">
            <h3>🔴 Error Analysis</h3>
            <img src="figures/error_analysis.png" alt="Error Analysis" onerror="this.style.display='none'">
        </div>
        
        <footer>
            <p>Generated by ALQAC Benchmark Tool | Legal RAG System Evaluation</p>
            <p>{self.result.timestamp}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        report_path = self.output_dir / f"benchmark_report_{self.timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML Report saved: {report_path}")


# ==============================================================================
# Standalone execution
# ==============================================================================

if __name__ == "__main__":
    print("This module is designed to be imported by benchmark_alqac.py")
    print("Usage: from scripts.benchmark_visualizer import BenchmarkVisualizer")
