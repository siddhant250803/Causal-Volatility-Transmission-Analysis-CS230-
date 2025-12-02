"""
Create network visualization of causal relationships between stocks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_causal_network(csv_path='results/first_5_stocks_all_relationships.csv',
                            threshold=0.1, output_path='plots/causal_network_graph.png'):
    """
    Create a network visualization of causal relationships.
    
    Args:
        csv_path: Path to combined relationships CSV
        threshold: Minimum causal strength to display
        output_path: Where to save the plot
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print("Run analyze_first_5_stocks.py first to generate data.")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df[df['causal_strength'] >= threshold]
    
    if len(df) == 0:
        print(f"No relationships found above threshold {threshold}")
        print(f"Try lowering the threshold. Min strength in data: {df['causal_strength'].min():.4f}")
        return
    
    print(f"Creating network visualization with {len(df)} relationships...")
    
    # Get unique stocks
    stocks = sorted(set(df['source_stock'].unique()) | set(df['target_stock'].unique()))
    n_stocks = len(stocks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Position stocks in a circle
    angles = np.linspace(0, 2*np.pi, n_stocks, endpoint=False)
    positions = {stock: (np.cos(angle), np.sin(angle)) 
                for stock, angle in zip(stocks, angles)}
    
    # Draw edges (arrows for causal relationships)
    for _, row in df.iterrows():
        source = row['source_stock']
        target = row['target_stock']
        strength = row['causal_strength']
        
        if source not in positions or target not in positions:
            continue
        
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        
        # Arrow properties based on strength
        width = strength * 3  # Scale arrow width
        alpha = min(0.3 + strength, 0.9)  # Transparency based on strength
        
        # Draw arrow (slightly curved)
        dx = x2 - x1
        dy = y2 - y1
        
        # Add curvature
        curve = 0.2
        mid_x = (x1 + x2) / 2 + curve * (-dy)
        mid_y = (y1 + y2) / 2 + curve * dx
        
        # Draw curved arrow using bezier-like path
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=width, alpha=alpha,
                                 color='steelblue', 
                                 connectionstyle=f"arc3,rad={curve}"))
    
    # Draw nodes
    for stock, (x, y) in positions.items():
        # Count incoming and outgoing edges
        incoming = df[df['target_stock'] == stock]['causal_strength'].sum()
        outgoing = df[df['source_stock'] == stock]['causal_strength'].sum()
        total_influence = incoming + outgoing
        
        # Node size based on total influence
        node_size = 300 + total_influence * 1000
        
        # Color based on net influence (more influenced vs more influential)
        net = outgoing - incoming
        if net > 0:
            color = 'lightcoral'  # More influential (source)
        else:
            color = 'lightblue'   # More influenced (target)
        
        ax.scatter(x, y, s=node_size, c=color, alpha=0.7, 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Label
        ax.text(x*1.15, y*1.15, stock, fontsize=12, fontweight='bold',
               ha='center', va='center', zorder=6)
    
    # Formatting
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title and legend
    ax.set_title(f'Causal Network: Stock Volatility Transmission\n'
                f'(threshold={threshold}, {len(df)} relationships)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='steelblue', lw=2, label='Causal influence'),
        plt.scatter([], [], s=300, c='lightcoral', edgecolors='black', 
                   label='More influential (net)'),
        plt.scatter([], [], s=300, c='lightblue', edgecolors='black',
                   label='More influenced (net)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics box
    stats_text = (f"Total relationships: {len(df)}\n"
                 f"Avg strength: {df['causal_strength'].mean():.3f}\n"
                 f"Avg lag: {df['lag_minutes'].mean():.1f} min")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Network visualization saved to: {output_path}")
    
    plt.show()


def create_heatmap(csv_path='results/first_5_stocks_matrix.csv',
                  output_path='plots/causal_heatmap_5stocks.png'):
    """Create a heatmap of the causal strength matrix."""
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    # Load matrix
    matrix = pd.read_csv(csv_path, index_col=0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               cbar_kws={'label': 'Causal Strength'}, ax=ax,
               linewidths=0.5, linecolor='gray')
    
    ax.set_title('Causal Strength Matrix: Source → Target',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Target Stock', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Stock', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize causal network')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Minimum causal strength to display (default: 0.1)')
    parser.add_argument('--csv', type=str, 
                       default='results/first_5_stocks_all_relationships.csv',
                       help='Path to relationships CSV')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Network graph
    visualize_causal_network(
        csv_path=args.csv,
        threshold=args.threshold
    )
    
    # Heatmap
    create_heatmap()
    
    print("\n" + "="*80)
    print("✓ Visualizations complete!")
    print("="*80 + "\n")

