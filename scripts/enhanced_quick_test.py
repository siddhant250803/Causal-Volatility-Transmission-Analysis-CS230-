"""
Enhanced quick test script for first 5 stocks with:
- Improved visualizations
- Optional hyperparameter tuning
- Parallel training support
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from src.config import Config
from src.data import StockDataLoader, VolatilityDataset
from src.train import Trainer
from src.analyze_causality import CausalityAnalyzer
from src.utils import (
    EnhancedVisualizer,
    HyperparameterSearch,
    get_default_search_space,
    GrangerCausalityTester,
    compute_realized_volatility_for_granger
)


def enhanced_quick_test(num_stocks: int = 5,
                       epochs: int = 20,
                       skip_granger: bool = False,
                       use_hp_tuning: bool = False,
                       hp_trials: int = 10):
    """
    Enhanced quick test with improved visualizations and optional HP tuning.
    
    Args:
        num_stocks: Number of stocks to analyze
        epochs: Training epochs per stock
        skip_granger: Skip Granger causality tests
        use_hp_tuning: Run hyperparameter tuning before training
        hp_trials: Number of HP tuning trials if enabled
    """
    print("="*80)
    print(f"ENHANCED QUICK TEST: {num_stocks} STOCKS")
    print("="*80)
    print(f"Settings:")
    print(f"  Stocks: {num_stocks}")
    print(f"  Epochs: {epochs}")
    print(f"  Granger testing: {'DISABLED' if skip_granger else 'ENABLED'}")
    print(f"  Hyperparameter tuning: {'ENABLED' if use_hp_tuning else 'DISABLED'}")
    if use_hp_tuning:
        print(f"  HP trials: {hp_trials}")
    print("="*80 + "\n")
    
    # Initialize config
    config = Config()
    config.NUM_EPOCHS = epochs
    
    # Load data
    print("Step 1: Loading and preprocessing data...")
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=num_stocks)
    
    stock_names = data_loader.stock_names
    print(f"  Loaded {len(stock_names)} stocks: {', '.join(stock_names)}")
    
    # Preprocess
    data_loader.compute_realized_volatility()
    n_timesteps = len(data_loader.returns)
    train_end_idx = int(n_timesteps * config.TRAIN_SPLIT)
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    print("\nStep 2: Creating sequences...")
    X_returns, X_volatility, y = data_loader.create_sequences()
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Initialize visualizer
    visualizer = EnhancedVisualizer()
    
    # Store results
    all_results = []
    all_relationships = []
    
    # Process each stock
    for i, stock_name in enumerate(stock_names):
        print(f"\n{'='*80}")
        print(f"STOCK {i+1}/{len(stock_names)}: {stock_name}")
        print(f"{'='*80}")
        
        target_stock_idx = i
        
        # Create datasets
        train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_stock_idx)
        val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_stock_idx)
        test_dataset = VolatilityDataset(*test_data, target_stock_idx=target_stock_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Hyperparameter tuning (optional)
        best_config = config
        if use_hp_tuning:
            print(f"\n{'─'*80}")
            print("HYPERPARAMETER TUNING")
            print(f"{'─'*80}")
            
            search_space = get_default_search_space()
            hp_search = HyperparameterSearch(config, search_space, n_trials=hp_trials)
            
            best_config, best_result = hp_search.run_search(
                train_loader=train_loader,
                val_loader=val_loader,
                n_stocks=len(stock_names),
                target_stock_idx=target_stock_idx,
                stock_name=stock_name,
                max_epochs_per_trial=epochs,
                save_results=True
            )
            
            print(f"\nBest configuration found!")
            print(f"  Val loss: {best_result['best_val_loss']:.6f}")
            print(f"  R²: {best_result['r2']:.4f}")
            
            # Plot tuning results
            hp_search.plot_search_results(stock_name)
        
        # Train model
        print(f"\n{'─'*80}")
        print("TRAINING")
        print(f"{'─'*80}")
        
        trainer = Trainer(best_config, target_stock_idx, stock_name)
        trainer.setup_model(n_stocks=len(stock_names))
        history = trainer.train(train_loader, val_loader)
        
        # Test
        test_losses, test_metrics = trainer.validate(test_loader)
        
        # Save stock info
        stock_info = {
            'stock_names': stock_names,
            'target_stock': stock_name,
            'target_stock_idx': target_stock_idx
        }
        os.makedirs('checkpoints', exist_ok=True)
        with open(f'checkpoints/{stock_name}_stock_info.json', 'w') as f:
            import json
            json.dump(stock_info, f, indent=2)
        
        # Plot enhanced training history
        if history:
            visualizer.plot_training_history(history, stock_name)
        
        # Analyze causality
        print(f"\n{'─'*80}")
        print("CAUSALITY ANALYSIS")
        print(f"{'─'*80}")
        
        analyzer = CausalityAnalyzer(stock_name, best_config)
        relationships = analyzer.get_causal_relationships()
        
        if len(relationships) > 0:
            print(f"\nFound {len(relationships)} significant relationships")
            print(f"\nTop 5 influences:")
            print(relationships.head(5).to_string(index=False))
            
            # Enhanced visualizations
            print("\nGenerating enhanced visualizations...")
            
            # 1. Enhanced network graph
            visualizer.plot_network_graph(
                relationships, stock_name,
                threshold=0.0,
                layout='hierarchical'
            )
            
            # 2. Comprehensive strength-lag analysis
            visualizer.plot_strength_lag_analysis(relationships, stock_name)
            
            # 3. Traditional plots (for comparison)
            analyzer.plot_causal_network(save=True)
            analyzer.plot_lag_distribution(save=True)
            analyzer.plot_heatmap(save=True)
            
            # Store for combined analysis
            all_relationships.append(relationships)
        else:
            print("No significant relationships found")
        
        # Granger causality (if enabled)
        if not skip_granger and len(relationships) > 0:
            print(f"\n{'─'*80}")
            print("GRANGER CAUSALITY VALIDATION")
            print(f"{'─'*80}")
            
            volatility = compute_realized_volatility_for_granger(
                data_loader.returns,
                window=config.LOOKBACK_WINDOW
            )
            
            tester = GrangerCausalityTester(max_lag=config.MAX_LAG)
            granger_results = tester.test_all_sources_to_target(
                volatility, target_stock_idx, stock_names
            )
            
            significant = granger_results[granger_results['granger_causes']]
            print(f"Found {len(significant)} Granger-causal relationships")
            
            # Save Granger results
            os.makedirs('results', exist_ok=True)
            granger_results.to_csv(f'results/{stock_name}_granger.csv', index=False)
        
        # Store results
        all_results.append({
            'stock': stock_name,
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'n_relationships': len(relationships)
        })
        
        print(f"\n✓ {stock_name} complete!")
    
    # Combined analysis
    print(f"\n{'='*80}")
    print("COMBINED ANALYSIS")
    print(f"{'='*80}\n")
    
    # Create combined results
    results_df = pd.DataFrame(all_results)
    print("Performance Summary:")
    print(results_df.to_string(index=False))
    print(f"\nAverage R²: {results_df['test_r2'].mean():.4f}")
    print(f"Average RMSE: {results_df['test_rmse'].mean():.6f}")
    
    # Combine all relationships
    if all_relationships:
        combined_relationships = pd.concat(all_relationships, ignore_index=True)
        
        # Create causal matrix
        causal_matrix = pd.DataFrame(
            np.zeros((len(stock_names), len(stock_names))),
            index=stock_names,
            columns=stock_names
        )
        
        for _, row in combined_relationships.iterrows():
            source = row['source_stock']
            target = row['target_stock']
            strength = row['causal_strength']
            causal_matrix.loc[source, target] = strength
        
        # Enhanced heatmap
        visualizer.plot_heatmap_matrix(
            causal_matrix,
            title=f"Causal Strength Matrix: {num_stocks} Stocks"
        )
        
        # Save combined results
        os.makedirs('results', exist_ok=True)
        combined_relationships.to_csv('results/enhanced_quick_test_relationships.csv', index=False)
        causal_matrix.to_csv('results/enhanced_quick_test_matrix.csv')
        results_df.to_csv('results/enhanced_quick_test_summary.csv', index=False)
        
        print("\n✓ Combined analysis complete!")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ ENHANCED QUICK TEST COMPLETE!")
    print(f"{'='*80}")
    print("\nGenerated outputs:")
    print("  📊 checkpoints/        : Trained models")
    print("  📈 plots/              : Enhanced visualizations")
    print("  📄 results/            : CSV analysis results")
    if use_hp_tuning:
        print("  🔍 hyperparameter_search/ : HP tuning results")
    print()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Enhanced quick test with improved visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with enhanced visuals
  python enhanced_quick_test.py
  
  # With hyperparameter tuning
  python enhanced_quick_test.py --hp_tuning --hp_trials 15
  
  # Quick validation (no Granger, fewer epochs)
  python enhanced_quick_test.py --epochs 10 --no_granger
  
  # Full featured test
  python enhanced_quick_test.py --epochs 30 --hp_tuning --hp_trials 20
        """
    )
    
    parser.add_argument('--num_stocks', type=int, default=5,
                       help='Number of stocks to analyze (default: 5)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs per stock (default: 20)')
    parser.add_argument('--no_granger', action='store_true',
                       help='Skip Granger causality testing')
    parser.add_argument('--hp_tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--hp_trials', type=int, default=10,
                       help='Number of HP tuning trials (default: 10)')
    
    args = parser.parse_args()
    
    # Run enhanced quick test
    enhanced_quick_test(
        num_stocks=args.num_stocks,
        epochs=args.epochs,
        skip_granger=args.no_granger,
        use_hp_tuning=args.hp_tuning,
        hp_trials=args.hp_trials
    )


if __name__ == '__main__':
    main()


