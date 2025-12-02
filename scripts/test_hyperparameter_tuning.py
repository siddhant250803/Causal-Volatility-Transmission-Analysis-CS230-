"""
Test hyperparameter tuning with the first 5 stocks.
Quick validation of the hyperparameter search functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import argparse

from src.config import Config
from src.data import StockDataLoader, VolatilityDataset
from src.utils.hyperparameter_tuning import HyperparameterSearch, get_default_search_space
from src.utils.enhanced_visualizations import EnhancedVisualizer


def test_hyperparameter_tuning(num_stocks: int = 5,
                               n_trials: int = 10,
                               max_epochs_per_trial: int = 20,
                               custom_search_space: dict = None):
    """
    Test hyperparameter tuning with a small number of stocks.
    
    Args:
        num_stocks: Number of stocks to use for testing
        n_trials: Number of hyperparameter trials
        max_epochs_per_trial: Maximum epochs per trial
        custom_search_space: Optional custom search space
    """
    print("="*80)
    print(f"HYPERPARAMETER TUNING TEST: {num_stocks} STOCKS")
    print("="*80)
    print(f"Trials: {n_trials}")
    print(f"Max epochs per trial: {max_epochs_per_trial}\n")
    
    # Initialize config
    config = Config()
    config.NUM_EPOCHS = max_epochs_per_trial
    
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
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Test hyperparameter tuning for first stock
    target_stock_name = stock_names[0]
    target_stock_idx = 0
    
    print(f"\nStep 3: Running hyperparameter search for {target_stock_name}...")
    
    # Create datasets
    train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_stock_idx)
    val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_stock_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Define search space
    if custom_search_space is None:
        search_space = get_default_search_space()
    else:
        search_space = custom_search_space
    
    print(f"\nSearch space:")
    for param, spec in search_space.items():
        if spec['type'] == 'choice':
            print(f"  {param}: {spec['values']}")
        else:
            print(f"  {param}: [{spec['low']}, {spec['high']}] ({spec['type']})")
    
    # Run search
    hp_search = HyperparameterSearch(config, search_space, n_trials=n_trials)
    
    best_config, best_result = hp_search.run_search(
        train_loader=train_loader,
        val_loader=val_loader,
        n_stocks=len(stock_names),
        target_stock_idx=target_stock_idx,
        stock_name=target_stock_name,
        max_epochs_per_trial=max_epochs_per_trial,
        save_results=True
    )
    
    # Plot results
    print("\nStep 4: Plotting search results...")
    hp_search.plot_search_results(target_stock_name)
    
    # Show improvement
    baseline_result = hp_search.results[0]
    print(f"\n{'='*80}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"\nBaseline (first trial):")
    print(f"  Val loss: {baseline_result['best_val_loss']:.6f}")
    print(f"  RMSE: {baseline_result['rmse']:.6f}")
    print(f"  R²: {baseline_result['r2']:.4f}")
    
    print(f"\nBest configuration (trial {best_result['trial']}):")
    print(f"  Val loss: {best_result['best_val_loss']:.6f}")
    print(f"  RMSE: {best_result['rmse']:.6f}")
    print(f"  R²: {best_result['r2']:.4f}")
    
    improvement = (baseline_result['best_val_loss'] - best_result['best_val_loss']) / baseline_result['best_val_loss'] * 100
    r2_improvement = (best_result['r2'] - baseline_result['r2']) / abs(baseline_result['r2']) * 100 if baseline_result['r2'] != 0 else 0
    
    print(f"\nImprovement:")
    print(f"  Val loss: {improvement:+.2f}%")
    print(f"  R² score: {r2_improvement:+.2f}%")
    
    print(f"\n{'='*80}")
    print("✓ HYPERPARAMETER TUNING TEST COMPLETE!")
    print(f"{'='*80}\n")
    
    # Test with all 5 stocks if requested
    if num_stocks == 5:
        print("\n" + "="*80)
        print("TESTING WITH ALL 5 STOCKS")
        print("="*80 + "\n")
        
        # Run a smaller search for remaining stocks
        small_trials = max(3, n_trials // 3)
        
        for idx, stock_name in enumerate(stock_names[1:], start=1):
            print(f"\n{'─'*80}")
            print(f"Stock {idx + 1}/5: {stock_name} ({small_trials} trials)")
            print(f"{'─'*80}")
            
            train_dataset = VolatilityDataset(*train_data, target_stock_idx=idx)
            val_dataset = VolatilityDataset(*val_data, target_stock_idx=idx)
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
            
            hp_search = HyperparameterSearch(config, search_space, n_trials=small_trials)
            
            best_config, best_result = hp_search.run_search(
                train_loader=train_loader,
                val_loader=val_loader,
                n_stocks=len(stock_names),
                target_stock_idx=idx,
                stock_name=stock_name,
                max_epochs_per_trial=max_epochs_per_trial // 2,  # Shorter for speed
                save_results=True
            )
            
            print(f"Best R² for {stock_name}: {best_result['r2']:.4f}")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETE!")
    print("="*80)
    print("\nCheck the following directories:")
    print("  - hyperparameter_search/  : Search results and plots")
    print()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Test hyperparameter tuning with first 5 stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  python test_hyperparameter_tuning.py
  
  # More thorough search
  python test_hyperparameter_tuning.py --n_trials 20 --max_epochs 30
  
  # Quick validation
  python test_hyperparameter_tuning.py --n_trials 5 --max_epochs 10
        """
    )
    
    parser.add_argument('--num_stocks', type=int, default=5,
                       help='Number of stocks to use (default: 5)')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of hyperparameter trials (default: 10)')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='Maximum epochs per trial (default: 20)')
    
    args = parser.parse_args()
    
    # Run test
    test_hyperparameter_tuning(
        num_stocks=args.num_stocks,
        n_trials=args.n_trials,
        max_epochs_per_trial=args.max_epochs
    )


if __name__ == '__main__':
    main()


