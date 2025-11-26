"""
Quick script to analyze causal relationships between first 5 stocks.
Trains a model for each stock and generates a comprehensive report.
"""

import os
import pandas as pd
from config import Config
from data import StockDataLoader
from run_analysis import train_for_stock, analyze_stock

def analyze_first_5_stocks(num_stocks=5, epochs=20, skip_granger=False):
    """
    Train and analyze models for first 5 stocks.
    
    Args:
        num_stocks: Number of stocks to analyze (default: 5)
        epochs: Training epochs per stock (default: 20)
        skip_granger: Skip Granger causality tests for speed (default: False)
    """
    print("="*80)
    print(f"ANALYZING CAUSAL RELATIONSHIPS: FIRST {num_stocks} STOCKS")
    print("="*80)
    
    # Step 1: Get list of first N stocks
    config = Config()
    data_loader = StockDataLoader(config.DATA_PATH, config)
    print("\nLoading data to identify stocks...")
    data_loader.load_data(max_stocks=num_stocks)
    
    stock_list = data_loader.stock_names[:num_stocks]
    
    print(f"\nAnalyzing stocks: {', '.join(stock_list)}")
    print(f"Total: {len(stock_list)} stocks")
    print(f"Epochs per stock: {epochs}")
    print(f"Granger testing: {'Disabled (faster)' if skip_granger else 'Enabled'}")
    
    # Step 2: Train model for each stock
    print("\n" + "="*80)
    print("PHASE 1: TRAINING MODELS")
    print("="*80)
    
    for i, stock in enumerate(stock_list, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(stock_list)}] Training model for {stock}")
        print(f"{'─'*80}")
        
        success = train_for_stock(
            stock_name=stock,
            num_stocks=num_stocks,  # Use only first N stocks
            epochs=epochs,
            force_retrain=True  # Always retrain for clean results
        )
        
        if not success:
            print(f"⚠️  Failed to train {stock}, skipping...")
            continue
    
    # Step 3: Analyze causal relationships
    print("\n" + "="*80)
    print("PHASE 2: ANALYZING CAUSAL RELATIONSHIPS")
    print("="*80)
    
    all_relationships = []
    
    for i, stock in enumerate(stock_list, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(stock_list)}] Analyzing relationships for {stock}")
        print(f"{'─'*80}")
        
        success = analyze_stock(
            stock_name=stock,
            top_k=num_stocks,  # Show all stocks in the group
            threshold=0.05,    # Lower threshold to see more relationships
            run_granger=not skip_granger
        )
        
        if not success:
            print(f"⚠️  Failed to analyze {stock}, skipping...")
            continue
        
        # Load the relationships CSV
        csv_path = f'results/{stock}_causal_relationships.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['target_stock'] = stock  # Add target for combined view
            all_relationships.append(df)
    
    # Step 4: Create comprehensive summary
    print("\n" + "="*80)
    print("PHASE 3: GENERATING SUMMARY")
    print("="*80)
    
    if all_relationships:
        combined = pd.concat(all_relationships, ignore_index=True)
        
        # Save combined results
        combined_path = 'results/first_5_stocks_all_relationships.csv'
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Combined results saved to: {combined_path}")
        
        # Create relationship matrix
        print("\n" + "─"*80)
        print("CAUSAL STRENGTH MATRIX (Source → Target)")
        print("─"*80)
        
        # Pivot to create matrix
        matrix = combined.pivot_table(
            index='source_stock',
            columns='target_stock',
            values='causal_strength',
            fill_value=0
        )
        
        # Format and display
        print("\n" + matrix.to_string())
        
        # Save matrix
        matrix_path = 'results/first_5_stocks_matrix.csv'
        matrix.to_csv(matrix_path)
        print(f"\n✓ Matrix saved to: {matrix_path}")
        
        # Summary statistics
        print("\n" + "─"*80)
        print("SUMMARY STATISTICS")
        print("─"*80)
        
        for stock in stock_list:
            stock_data = combined[combined['target_stock'] == stock]
            if len(stock_data) > 0:
                print(f"\n{stock}:")
                print(f"  Influenced by: {len(stock_data)} stocks")
                print(f"  Mean strength: {stock_data['causal_strength'].mean():.4f}")
                print(f"  Max strength:  {stock_data['causal_strength'].max():.4f}")
                print(f"  Mean lag:      {stock_data['lag_minutes'].mean():.1f} min")
                
                # Top influencer
                top = stock_data.nlargest(1, 'causal_strength').iloc[0]
                print(f"  Top influencer: {top['source_stock']} "
                      f"(strength={top['causal_strength']:.4f}, "
                      f"lag={top['lag_minutes']:.1f}min)")
        
        # Network statistics
        print("\n" + "─"*80)
        print("NETWORK STATISTICS")
        print("─"*80)
        print(f"Total relationships found: {len(combined)}")
        print(f"Average causal strength: {combined['causal_strength'].mean():.4f}")
        print(f"Average lag: {combined['lag_minutes'].mean():.1f} minutes")
        
        # Most influential stocks (highest outgoing strength)
        outgoing = combined.groupby('source_stock')['causal_strength'].agg(['sum', 'mean', 'count'])
        outgoing = outgoing.sort_values('sum', ascending=False)
        print(f"\nMost influential stocks (by total outgoing strength):")
        print(outgoing.head())
        
        # Most influenced stocks (highest incoming strength)
        incoming = combined.groupby('target_stock')['causal_strength'].agg(['sum', 'mean', 'count'])
        incoming = incoming.sort_values('sum', ascending=False)
        print(f"\nMost influenced stocks (by total incoming strength):")
        print(incoming.head())
        
    else:
        print("\n⚠️  No relationships found. Check training logs for errors.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - results/first_5_stocks_all_relationships.csv")
    print(f"  - results/first_5_stocks_matrix.csv")
    print(f"  - results/[STOCK]_causal_relationships.csv (individual)")
    print(f"  - plots/[STOCK]_*.png (visualizations)")
    print("\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze causal relationships between first N stocks'
    )
    parser.add_argument('--num_stocks', type=int, default=5,
                       help='Number of stocks to analyze (default: 5)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs per stock (default: 20)')
    parser.add_argument('--skip_granger', action='store_true',
                       help='Skip Granger causality tests for speed')
    
    args = parser.parse_args()
    
    analyze_first_5_stocks(
        num_stocks=args.num_stocks,
        epochs=args.epochs,
        skip_granger=args.skip_granger
    )

