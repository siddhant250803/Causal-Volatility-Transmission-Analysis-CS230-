"""
Interactive script to train model and analyze causal relationships for any stock.
This is the main entry point for using the framework.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

from config import Config
from data import StockDataLoader, VolatilityDataset
from models import CausalAttentionModel
from utils import CausalRegularizedLoss, calculate_metrics
from train import Trainer
from analyze_causality import CausalityAnalyzer


def list_available_stocks(data_path: str, max_stocks: int = 50):
    """
    List available stocks in the dataset.
    
    Args:
        data_path: Path to data file
        max_stocks: Maximum number of stocks to load
    """
    config = Config()
    data_loader = StockDataLoader(data_path, config)
    print("Loading data to get stock list...")
    data_loader.load_data(max_stocks=max_stocks)
    
    print(f"\n{'='*80}")
    print(f"AVAILABLE STOCKS ({len(data_loader.stock_names)} total)")
    print(f"{'='*80}\n")
    
    # Print in columns
    stocks = data_loader.stock_names
    cols = 6
    for i in range(0, len(stocks), cols):
        row = stocks[i:i+cols]
        print("  ".join(f"{s:8s}" for s in row))
    
    print()
    return data_loader.stock_names


def train_for_stock(stock_name: str, num_stocks: int = 50, epochs: int = None, force_retrain: bool = False):
    """
    Train model for a specific stock.
    
    Args:
        stock_name: Name of the stock ticker
        num_stocks: Number of stocks to include in analysis
        epochs: Number of training epochs (None = use config default)
        force_retrain: If True, retrain even if checkpoint exists
    """
    # Initialize config
    config = Config()
    if epochs is not None:
        config.NUM_EPOCHS = epochs
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL FOR: {stock_name}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Step 1: Loading and preprocessing data...")
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=num_stocks)
    
    # Find stock index
    if stock_name not in data_loader.stock_names:
        print(f"Error: Stock '{stock_name}' not found in dataset.")
        print("Use --list to see available stocks.")
        return False
    
    target_stock_idx = data_loader.stock_names.index(stock_name)
    
    # Preprocess
    data_loader.compute_realized_volatility()
    
    # Compute train split index for proper normalization (avoid data leakage)
    n_timesteps = len(data_loader.returns)
    train_end_idx = int(n_timesteps * config.TRAIN_SPLIT)
    
    # Normalize using only training data statistics
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    # Create sequences
    print("\nStep 2: Creating training sequences...")
    X_returns, X_volatility, y = data_loader.create_sequences()
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Create datasets
    print("\nStep 3: Setting up datasets...")
    train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_stock_idx)
    val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_stock_idx)
    test_dataset = VolatilityDataset(*test_data, target_stock_idx=target_stock_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train model
    print("\nStep 4: Training model...")
    trainer = Trainer(config, target_stock_idx, stock_name)
    trainer.setup_model(n_stocks=len(data_loader.stock_names))
    
    # Check if checkpoint exists and is compatible
    checkpoint_path = f'checkpoints/{stock_name}_best.pt'
    should_train = force_retrain
    
    if os.path.exists(checkpoint_path) and not force_retrain:
        print(f"\n✓ Found existing checkpoint: {checkpoint_path}")
        # Check if checkpoint is compatible (same number of stocks)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
            checkpoint_n_stocks = checkpoint['model_state_dict']['attention.lags'].shape[0]
            
            if checkpoint_n_stocks == len(data_loader.stock_names):
                print("Loading trained model (use --force_retrain to retrain)...")
                trainer.load_checkpoint('best')
            else:
                print(f"⚠️  Checkpoint has {checkpoint_n_stocks} stocks, but you requested {len(data_loader.stock_names)} stocks.")
                print("Checkpoint incompatible. Training new model...")
                should_train = True
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("Training new model...")
            should_train = True
    else:
        should_train = True
    
    if should_train:
        if force_retrain:
            print("Force retraining enabled. Training new model...")
        else:
            print("No existing checkpoint found. Training new model...")
        trainer.train(train_loader, val_loader)
    
    # Test
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    test_losses, test_metrics = trainer.validate(test_loader)
    print(f"Test MSE:  {test_losses['mse_loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE:  {test_metrics['mae']:.6f}")
    print(f"Test R²:   {test_metrics['r2']:.4f}")
    
    # Save stock names for later analysis
    stock_info = {
        'stock_names': data_loader.stock_names,
        'target_stock': stock_name,
        'target_stock_idx': target_stock_idx
    }
    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{stock_name}_stock_info.json', 'w') as f:
        json.dump(stock_info, f, indent=2)
    
    print(f"\n✓ Training complete! Model saved to checkpoints/{stock_name}_best.pt")
    return True


def analyze_stock(stock_name: str, top_k: int = 10, threshold: float = 0.1, 
                  run_granger: bool = True):
    """
    Analyze causal relationships for a stock.
    
    Args:
        stock_name: Name of the stock ticker
        top_k: Number of top relationships to show
        threshold: Minimum causal strength threshold
        run_granger: Whether to run Granger causality tests
    """
    config = Config()
    config.TOP_K_INFLUENCES = top_k
    config.CAUSAL_THRESHOLD = threshold
    
    print(f"\n{'='*80}")
    print(f"ANALYZING CAUSAL RELATIONSHIPS FOR: {stock_name}")
    print(f"{'='*80}\n")
    
    # Create analyzer
    try:
        analyzer = CausalityAnalyzer(stock_name, config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nPlease train the model first using:")
        print(f"  python run_analysis.py --train --stock {stock_name}")
        return False
    
    # Generate attention-based report
    analyzer.generate_report()
    
    # Run Granger causality tests if requested
    if run_granger:
        print("\n" + "="*80)
        print("GRANGER CAUSALITY ANALYSIS")
        print("="*80)
        
        from utils import GrangerCausalityTester, compute_realized_volatility_for_granger
        
        # Load data for Granger testing
        print("\nLoading data for Granger causality tests...")
        data_loader = StockDataLoader(config.DATA_PATH, config)
        data_loader.load_data(max_stocks=len(analyzer.stock_names))
        
        # Get the target stock index
        target_idx = analyzer.target_stock_idx
        
        # Compute volatility
        volatility = compute_realized_volatility_for_granger(
            data_loader.returns, 
            window=config.LOOKBACK_WINDOW
        )
        
        # Run Granger tests
        print(f"Running Granger causality tests (this may take a minute)...")
        tester = GrangerCausalityTester(max_lag=config.MAX_LAG)
        granger_results = tester.test_all_sources_to_target(
            volatility,
            target_idx,
            analyzer.stock_names
        )
        
        # Compare with attention results (use all gates for comparison)
        attention_results = analyzer.get_causal_relationships(threshold=0, use_relative=False)  # Get all
        comparison = tester.compare_with_attention_gates(granger_results, attention_results)
        
        # Print Granger results
        significant = granger_results[granger_results['granger_causes']]
        print(f"\n✓ Found {len(significant)} stocks with significant Granger causality (p < 0.05)")
        
        if len(significant) > 0:
            print(f"\nTop 10 Granger-causal relationships:")
            print(significant.head(10)[['source_stock', 'p_value', 'best_lag', 'f_statistic']].to_string(index=False))
        
        # Print comparison
        print(f"\n{'─'*80}")
        print("COMPARISON: Attention Gates vs. Granger Causality")
        print(f"{'─'*80}")
        
        attention_significant = (comparison['causal_strength'].fillna(0) > threshold).sum()
        granger_significant = comparison['granger_causes'].fillna(False).sum()
        both_agree = comparison['methods_agree'].sum()
        
        print(f"Attention method found:  {attention_significant} significant relationships")
        print(f"Granger method found:    {granger_significant} significant relationships")
        print(f"Both methods agree on:   {both_agree} relationships")
        
        if both_agree > 0:
            print(f"\nStocks validated by BOTH methods:")
            validated = comparison[comparison['methods_agree']]
            print(validated[['source_stock', 'p_value', 'causal_strength']].to_string(index=False))
        
        # Save results
        os.makedirs('results', exist_ok=True)
        granger_results.to_csv(f'results/{stock_name}_granger_causality.csv', index=False)
        comparison.to_csv(f'results/{stock_name}_comparison.csv', index=False)
        print(f"\n✓ Granger results saved to: results/{stock_name}_granger_causality.csv")
        print(f"✓ Comparison saved to: results/{stock_name}_comparison.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_causal_network(save=True)
    analyzer.plot_lag_distribution(save=True)
    analyzer.plot_heatmap(save=True)
    
    print("\n✓ Analysis complete!")
    print(f"  - Report saved to: results/{stock_name}_causal_relationships.csv")
    print(f"  - Plots saved to: plots/")
    
    return True


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train and analyze causal volatility transmission for stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available stocks
  python run_analysis.py --list
  
  # Train model for AAPL
  python run_analysis.py --train --stock AAPL
  
  # Analyze causal relationships for AAPL
  python run_analysis.py --analyze --stock AAPL
  
  # Train and analyze in one go
  python run_analysis.py --train --analyze --stock AAPL
  
  # Use custom parameters
  python run_analysis.py --train --stock AAPL --num_stocks 100 --epochs 30
  python run_analysis.py --analyze --stock AAPL --top_k 15 --threshold 0.15
        """
    )
    
    # Actions
    parser.add_argument('--list', action='store_true', help='List available stocks')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--analyze', action='store_true', help='Analyze causal relationships')
    
    # Parameters
    parser.add_argument('--stock', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--num_stocks', type=int, default=50, 
                       help='Number of stocks to include (default: 50)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retrain even if checkpoint exists')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Number of top relationships to show (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                       help='Minimum causal strength threshold (default: 0.1)')
    parser.add_argument('--no_granger', action='store_true',
                       help='Skip Granger causality testing (faster)')
    
    args = parser.parse_args()
    
    # List stocks
    if args.list:
        list_available_stocks('HF_Returns_Stocks.csv', max_stocks=args.num_stocks)
        return
    
    # Require stock name for train/analyze
    if (args.train or args.analyze) and not args.stock:
        parser.error("--stock is required when using --train or --analyze")
    
    # Train
    if args.train:
        success = train_for_stock(args.stock, args.num_stocks, args.epochs, args.force_retrain)
        if not success:
            return
    
    # Analyze
    if args.analyze:
        if not args.train:
            # Check if model exists
            if not os.path.exists(f'checkpoints/{args.stock}_best.pt'):
                print(f"\nWarning: No trained model found for {args.stock}")
                print("Training model first...")
                success = train_for_stock(args.stock, args.num_stocks, args.epochs, False)
                if not success:
                    return
        
        analyze_stock(args.stock, args.top_k, args.threshold, run_granger=not args.no_granger)
    
    # If no action specified, show help
    if not (args.list or args.train or args.analyze):
        parser.print_help()


if __name__ == '__main__':
    main()

