"""
Enhanced visualization utilities for causal network analysis.
Provides publication-quality figures with improved aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnhancedVisualizer:
    """Enhanced visualization class for causal networks."""
    
    def __init__(self, plot_dir='plots/', dpi=300):
        """
        Initialize visualizer.
        
        Args:
            plot_dir: Directory to save plots
            dpi: Resolution for saved figures
        """
        self.plot_dir = plot_dir
        self.dpi = dpi
        os.makedirs(plot_dir, exist_ok=True)
        
    def plot_network_graph(self, relationships_df: pd.DataFrame, 
                          stock_name: str,
                          threshold: float = 0.1,
                          layout: str = 'spring',
                          save_path: Optional[str] = None):
        """
        Create an enhanced network graph with directed edges.
        
        Args:
            relationships_df: DataFrame with causal relationships
            stock_name: Name of target stock
            threshold: Minimum strength to display
            layout: Graph layout ('spring', 'circular', 'hierarchical')
            save_path: Optional custom save path
        """
        df = relationships_df[relationships_df['causal_strength'] >= threshold].copy()
        
        if len(df) == 0:
            print(f"No relationships above threshold {threshold}")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        all_stocks = set(df['source_stock']) | set(df['target_stock'])
        G.add_nodes_from(all_stocks)
        
        # Add edges with weights
        for _, row in df.iterrows():
            G.add_edge(row['source_stock'], row['target_stock'],
                      weight=row['causal_strength'],
                      lag=row['lag_minutes'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            # Put target at center
            pos = self._hierarchical_layout(G, stock_name)
        else:
            pos = nx.spring_layout(G)
        
        # Node properties
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            in_degree = G.in_degree(node, weight='weight')
            out_degree = G.out_degree(node, weight='weight')
            
            # Size based on total connections
            size = 2000 + (in_degree + out_degree) * 3000
            node_sizes.append(size)
            
            # Color based on role
            if node == stock_name:
                node_colors.append('#FF6B6B')  # Target: red
            elif out_degree > in_degree:
                node_colors.append('#4ECDC4')  # Source: teal
            else:
                node_colors.append('#95E1D3')  # Receiver: light teal
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=2,
                              ax=ax)
        
        # Draw edges with varying widths and colors
        edges = G.edges(data=True)
        for edge in edges:
            source, target, data = edge
            weight = data['weight']
            lag = data['lag']
            
            # Edge properties
            width = 1 + weight * 5
            alpha = 0.3 + weight * 0.6
            
            # Color by lag
            color = plt.cm.viridis(lag / df['lag_minutes'].max())
            
            # Draw curved edge
            rad = 0.2
            arrow = FancyArrowPatch(pos[source], pos[target],
                                   arrowstyle='-|>',
                                   mutation_scale=20,
                                   lw=width,
                                   alpha=alpha,
                                   color=color,
                                   connectionstyle=f"arc3,rad={rad}",
                                   zorder=1)
            ax.add_patch(arrow)
        
        # Draw labels
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels,
                               font_size=11,
                               font_weight='bold',
                               font_color='black',
                               ax=ax)
        
        # Title and legend
        ax.set_title(f'Causal Network: Volatility Transmission to {stock_name}\n'
                    f'({len(df)} significant relationships, threshold={threshold:.2f})',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='#FF6B6B', label=f'Target ({stock_name})'),
            mpatches.Patch(color='#4ECDC4', label='Primary sources'),
            mpatches.Patch(color='#95E1D3', label='Secondary sources'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        
        # Add statistics box
        stats_text = (
            f"Network Statistics:\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Nodes: {G.number_of_nodes()}\n"
            f"Edges: {G.number_of_edges()}\n"
            f"Avg strength: {df['causal_strength'].mean():.3f}\n"
            f"Max strength: {df['causal_strength'].max():.3f}\n"
            f"Avg lag: {df['lag_minutes'].mean():.1f} min\n"
            f"Lag range: {df['lag_minutes'].min():.0f}-{df['lag_minutes'].max():.0f} min"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', 
                        alpha=0.8, edgecolor='black', linewidth=1.5),
               family='monospace')
        
        ax.axis('off')
        ax.margins(0.1)
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = f"{self.plot_dir}{stock_name}_enhanced_network.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Enhanced network graph saved to: {save_path}")
        plt.close()
    
    def _hierarchical_layout(self, G, center_node):
        """Create hierarchical layout with target at center."""
        pos = {}
        
        # Center node at origin
        pos[center_node] = np.array([0.0, 0.0])
        
        # Other nodes in circles
        other_nodes = [n for n in G.nodes() if n != center_node]
        n = len(other_nodes)
        
        for i, node in enumerate(other_nodes):
            angle = 2 * np.pi * i / n
            radius = 2.0
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        return pos
    
    def plot_strength_lag_analysis(self, relationships_df: pd.DataFrame,
                                   stock_name: str,
                                   save_path: Optional[str] = None):
        """
        Create comprehensive strength-lag analysis plot.
        
        Args:
            relationships_df: DataFrame with causal relationships
            stock_name: Name of target stock
            save_path: Optional custom save path
        """
        df = relationships_df.copy()
        
        if len(df) == 0:
            print("No relationships to plot")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main scatter plot (large)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        scatter = ax_main.scatter(df['lag_minutes'], df['causal_strength'],
                                 s=df['causal_strength'] * 500,
                                 c=df['causal_strength'],
                                 cmap='plasma',
                                 alpha=0.6,
                                 edgecolors='black',
                                 linewidth=1.5)
        
        # Add stock labels for top relationships
        top_n = min(10, len(df))
        for idx in df.nlargest(top_n, 'causal_strength').index:
            ax_main.annotate(df.loc[idx, 'source_stock'],
                           (df.loc[idx, 'lag_minutes'], df.loc[idx, 'causal_strength']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        
        ax_main.set_xlabel('Lag (minutes)', fontsize=13, fontweight='bold')
        ax_main.set_ylabel('Causal Strength', fontsize=13, fontweight='bold')
        ax_main.set_title(f'Causal Strength vs. Lag for {stock_name}',
                         fontsize=15, fontweight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('Causal Strength', fontsize=11, fontweight='bold')
        
        # 2. Lag distribution histogram
        ax_lag = fig.add_subplot(gs[2, 0:2])
        ax_lag.hist(df['lag_minutes'], bins=30, color='steelblue',
                   alpha=0.7, edgecolor='black', linewidth=1.2)
        ax_lag.axvline(df['lag_minutes'].mean(), color='red',
                      linestyle='--', linewidth=2, label=f"Mean: {df['lag_minutes'].mean():.1f} min")
        ax_lag.axvline(df['lag_minutes'].median(), color='green',
                      linestyle='--', linewidth=2, label=f"Median: {df['lag_minutes'].median():.1f} min")
        ax_lag.set_xlabel('Lag (minutes)', fontsize=12, fontweight='bold')
        ax_lag.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax_lag.set_title('Lag Distribution', fontsize=13, fontweight='bold')
        ax_lag.legend(fontsize=10)
        ax_lag.grid(True, alpha=0.3, axis='y')
        
        # 3. Strength distribution
        ax_strength = fig.add_subplot(gs[0, 2])
        ax_strength.hist(df['causal_strength'], bins=20, orientation='horizontal',
                        color='coral', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax_strength.axhline(df['causal_strength'].mean(), color='red',
                           linestyle='--', linewidth=2)
        ax_strength.set_ylabel('Causal Strength', fontsize=11, fontweight='bold')
        ax_strength.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax_strength.set_title('Strength Distribution', fontsize=12, fontweight='bold')
        ax_strength.grid(True, alpha=0.3, axis='x')
        
        # 4. Box plot by lag quartiles
        ax_box = fig.add_subplot(gs[1, 2])
        df['lag_quartile'] = pd.qcut(df['lag_minutes'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        df.boxplot(column='causal_strength', by='lag_quartile', ax=ax_box)
        ax_box.set_xlabel('Lag Quartile', fontsize=11, fontweight='bold')
        ax_box.set_ylabel('Causal Strength', fontsize=11, fontweight='bold')
        ax_box.set_title('Strength by Lag Quartile', fontsize=12, fontweight='bold')
        plt.sca(ax_box)
        plt.xticks(rotation=0)
        
        # 5. Statistics table
        ax_stats = fig.add_subplot(gs[2, 2])
        ax_stats.axis('off')
        
        stats_text = f"""
        STATISTICS SUMMARY
        ═══════════════════
        
        Relationships: {len(df)}
        
        Causal Strength:
          Mean:   {df['causal_strength'].mean():.4f}
          Median: {df['causal_strength'].median():.4f}
          Std:    {df['causal_strength'].std():.4f}
          Max:    {df['causal_strength'].max():.4f}
        
        Lag (minutes):
          Mean:   {df['lag_minutes'].mean():.1f}
          Median: {df['lag_minutes'].median():.1f}
          Std:    {df['lag_minutes'].std():.1f}
          Range:  {df['lag_minutes'].min():.0f} - {df['lag_minutes'].max():.0f}
        """
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                              alpha=0.8, edgecolor='black', linewidth=1.5))
        
        fig.suptitle(f'Comprehensive Causal Analysis: {stock_name}',
                    fontsize=17, fontweight='bold', y=0.995)
        
        # Save
        if save_path is None:
            save_path = f"{self.plot_dir}{stock_name}_strength_lag_analysis.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Strength-lag analysis saved to: {save_path}")
        plt.close()
    
    def plot_heatmap_matrix(self, causal_matrix: pd.DataFrame,
                           title: str = "Causal Strength Matrix",
                           save_path: Optional[str] = None):
        """
        Create enhanced heatmap of causal strength matrix.
        
        Args:
            causal_matrix: DataFrame with causal strengths (source × target)
            title: Plot title
            save_path: Optional custom save path
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create mask for diagonal
        mask = np.zeros_like(causal_matrix, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Create heatmap
        sns.heatmap(causal_matrix,
                   mask=mask,
                   annot=True if len(causal_matrix) <= 10 else False,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   cbar_kws={'label': 'Causal Strength', 'shrink': 0.8},
                   linewidths=0.5,
                   linecolor='gray',
                   square=True,
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Target Stock →', fontsize=13, fontweight='bold')
        ax.set_ylabel('← Source Stock', fontsize=13, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = f"{self.plot_dir}causal_matrix_heatmap.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Heatmap saved to: {save_path}")
        plt.close()
    
    def plot_training_history(self, history: dict, stock_name: str,
                             save_path: Optional[str] = None):
        """
        Plot training history with multiple metrics.
        
        Args:
            history: Dictionary with training metrics
            stock_name: Name of stock
            save_path: Optional custom save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History: {stock_name}',
                    fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2, markersize=4)
        ax.plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2, markersize=4)
        if 'best_epoch' in history:
            ax.axvline(history['best_epoch'], color='red', linestyle='--',
                      label=f"Best ({history['best_epoch']})", alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE plot
        if 'train_rmse' in history:
            ax = axes[0, 1]
            ax.plot(epochs, history['train_rmse'], 'o-', label='Train', linewidth=2)
            ax.plot(epochs, history['val_rmse'], 's-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
            ax.set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # R² plot
        if 'train_r2' in history:
            ax = axes[1, 0]
            ax.plot(epochs, history['train_r2'], 'o-', label='Train', linewidth=2)
            ax.plot(epochs, history['val_r2'], 's-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax.set_title('R² Score', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'learning_rate' in history:
            ax = axes[1, 1]
            ax.plot(epochs, history['learning_rate'], 'o-', color='purple', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = f"{self.plot_dir}{stock_name}_training_history.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Training history saved to: {save_path}")
        plt.close()


