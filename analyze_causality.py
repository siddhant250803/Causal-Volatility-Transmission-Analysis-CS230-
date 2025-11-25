"""
Analyze and visualize causal relationships between stocks.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict
import argparse

from config import Config
from models import CausalAttentionModel
from data import StockDataLoader


class CausalityAnalyzer:
    """Analyze causal relationships learned by the model."""
    
    def __init__(self, stock_name: str, config: Config):
        """
        Initialize analyzer.
        
        Args:
            stock_name: Name of the target stock
            config: Configuration object
        """
        self.stock_name = stock_name
        self.config = config
        self.device = config.DEVICE
        
        # Load model and stock info
        self.model = None
        self.stock_names = []
        self.target_stock_idx = None
        self.load_model_and_info()
        
    def load_model_and_info(self):
        """Load trained model and stock information."""
        # Load stock info
        info_path = f'checkpoints/{self.stock_name}_stock_info.json'
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Stock info not found at {info_path}. Please train the model first.")
        
        with open(info_path, 'r') as f:
            stock_info = json.load(f)
        
        self.stock_names = stock_info['stock_names']
        self.target_stock_idx = stock_info['target_stock_idx']
        
        # Initialize and load model
        self.model = CausalAttentionModel(
            n_stocks=len(self.stock_names),
            lookback=self.config.LOOKBACK_WINDOW,
            d_model=self.config.D_MODEL,
            d_k=self.config.D_K,
            d_v=self.config.D_V,
            max_lag=self.config.MAX_LAG,
            dropout=self.config.DROPOUT
        ).to(self.device)
        
        # Load checkpoint
        checkpoint_path = f'checkpoints/{self.stock_name}_best.pt'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        
        # weights_only=False needed for PyTorch 2.6+ when checkpoint contains numpy arrays
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model for {self.stock_name}")
        print(f"Analyzing relationships with {len(self.stock_names)} stocks")
        
    def get_causal_relationships(self, threshold: float = None) -> pd.DataFrame:
        """
        Extract causal relationships from the model.
        
        Args:
            threshold: Minimum gate value to consider a relationship significant
            
        Returns:
            DataFrame with causal relationships
        """
        if threshold is None:
            threshold = self.config.CAUSAL_THRESHOLD
        
        # Get causal graph
        causal_graph = self.model.get_causal_graph(self.target_stock_idx)
        gates = causal_graph['gates']
        lags = causal_graph['lags']
        
        # Create DataFrame
        relationships = []
        for i, stock_name in enumerate(self.stock_names):
            if i == self.target_stock_idx:
                continue  # Skip self
            
            if gates[i] >= threshold:
                relationships.append({
                    'source_stock': stock_name,
                    'target_stock': self.stock_name,
                    'causal_strength': gates[i],
                    'lag_minutes': lags[i] * 5,  # Convert to minutes (5-min intervals)
                    'lag_intervals': int(np.round(lags[i]))
                })
        
        df = pd.DataFrame(relationships)
        if len(df) > 0:
            df = df.sort_values('causal_strength', ascending=False)
        
        return df
    
    def plot_causal_network(self, top_k: int = None, save: bool = True):
        """
        Plot the causal network as a directed graph.
        
        Args:
            top_k: Number of top relationships to display
            save: Whether to save the plot
        """
        if top_k is None:
            top_k = self.config.TOP_K_INFLUENCES
        
        relationships = self.get_causal_relationships()
        
        if len(relationships) == 0:
            print("No significant causal relationships found.")
            return
        
        # Take top k relationships
        relationships = relationships.head(top_k)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        y_pos = np.arange(len(relationships))
        colors = plt.cm.RdYlGn(relationships['causal_strength'].values)
        
        bars = ax.barh(y_pos, relationships['causal_strength'].values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(relationships['source_stock'].values)
        ax.set_xlabel('Causal Strength (Gate Value)', fontsize=12)
        ax.set_title(f'Top {len(relationships)} Stocks Influencing {self.stock_name}', 
                     fontsize=14, fontweight='bold')
        
        # Add lag information as text
        for i, (idx, row) in enumerate(relationships.iterrows()):
            lag_text = f"{row['lag_minutes']:.1f} min"
            ax.text(row['causal_strength'] + 0.01, i, lag_text, 
                   va='center', fontsize=9)
        
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.config.PLOT_DIR, exist_ok=True)
            plot_path = f"{self.config.PLOT_DIR}{self.stock_name}_causal_network.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
        
    def plot_lag_distribution(self, save: bool = True):
        """
        Plot the distribution of lags for significant relationships.
        
        Args:
            save: Whether to save the plot
        """
        relationships = self.get_causal_relationships()
        
        if len(relationships) == 0:
            print("No significant causal relationships found.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of lags
        ax1.hist(relationships['lag_minutes'].values, bins=20, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Lag (minutes)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution of Lags for {self.stock_name}', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter: causal strength vs lag
        scatter = ax2.scatter(relationships['lag_minutes'].values, 
                             relationships['causal_strength'].values,
                             c=relationships['causal_strength'].values,
                             cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Lag (minutes)', fontsize=12)
        ax2.set_ylabel('Causal Strength', fontsize=12)
        ax2.set_title('Causal Strength vs. Lag', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Causal Strength', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.config.PLOT_DIR, exist_ok=True)
            plot_path = f"{self.config.PLOT_DIR}{self.stock_name}_lag_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
        
    def plot_heatmap(self, save: bool = True):
        """
        Plot heatmap of causal strengths.
        
        Args:
            save: Whether to save the plot
        """
        causal_graph = self.model.get_causal_graph(self.target_stock_idx)
        gates = causal_graph['gates']
        
        # Create matrix (simplified - just show influence on target stock)
        # For full implementation, you'd train models for all stocks
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Reshape gates for heatmap
        gates_matrix = gates.reshape(1, -1)
        
        # Create heatmap
        im = ax.imshow(gates_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_yticks([0])
        ax.set_yticklabels([self.stock_name])
        ax.set_xticks(np.arange(len(self.stock_names))[::5])  # Show every 5th stock
        ax.set_xticklabels([self.stock_names[i] for i in range(0, len(self.stock_names), 5)], 
                          rotation=45, ha='right')
        
        ax.set_xlabel('Source Stocks', fontsize=12)
        ax.set_title(f'Causal Influence on {self.stock_name}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Causal Strength', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            os.makedirs(self.config.PLOT_DIR, exist_ok=True)
            plot_path = f"{self.config.PLOT_DIR}{self.stock_name}_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print(f"CAUSAL ANALYSIS REPORT: {self.stock_name}")
        print("="*80)
        
        relationships = self.get_causal_relationships()
        
        if len(relationships) == 0:
            print("\nNo significant causal relationships found.")
            print(f"(Threshold: {self.config.CAUSAL_THRESHOLD})")
            return
        
        print(f"\nFound {len(relationships)} significant causal relationships")
        print(f"(Threshold: {self.config.CAUSAL_THRESHOLD})")
        
        # Top influencers
        print(f"\n{'─'*80}")
        print(f"TOP {min(10, len(relationships))} INFLUENCING STOCKS:")
        print(f"{'─'*80}")
        print(relationships.head(10).to_string(index=False))
        
        # Statistics
        print(f"\n{'─'*80}")
        print("STATISTICS:")
        print(f"{'─'*80}")
        print(f"Mean causal strength: {relationships['causal_strength'].mean():.4f}")
        print(f"Max causal strength:  {relationships['causal_strength'].max():.4f}")
        print(f"Mean lag:             {relationships['lag_minutes'].mean():.2f} minutes")
        print(f"Median lag:           {relationships['lag_minutes'].median():.2f} minutes")
        print(f"Lag range:            {relationships['lag_minutes'].min():.2f} - "
              f"{relationships['lag_minutes'].max():.2f} minutes")
        
        # Save to CSV
        os.makedirs('results', exist_ok=True)
        csv_path = f'results/{self.stock_name}_causal_relationships.csv'
        relationships.to_csv(csv_path, index=False)
        print(f"\nFull results saved to: {csv_path}")
        
        print("\n" + "="*80)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze causal relationships for a stock')
    parser.add_argument('--stock', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top relationships to show')
    parser.add_argument('--threshold', type=float, default=0.1, help='Minimum causal strength threshold')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.TOP_K_INFLUENCES = args.top_k
    config.CAUSAL_THRESHOLD = args.threshold
    
    # Create analyzer
    try:
        analyzer = CausalityAnalyzer(args.stock, config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate report
    analyzer.generate_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_causal_network(save=True)
    analyzer.plot_lag_distribution(save=True)
    analyzer.plot_heatmap(save=True)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

