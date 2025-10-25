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


def train_for_stock(stock_name: str, num_stocks: int = 50, epochs: int = None):
    """
    Train model for a specific stock.
    
    Args:
        stock_name: Name of the stock ticker
        num_stocks: Number of stocks to include in analysis
        epochs: Number of training epochs (None = use config default)
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
    data_loader.normalize_data()
    
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


def analyze_stock(stock_name: str, top_k: int = 10, threshold: float = 0.1):
    """
    Analyze causal relationships for a stock.
    
    Args:
        stock_name: Name of the stock ticker
        top_k: Number of top relationships to show
        threshold: Minimum causal strength threshold
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
    
    # Generate report
    analyzer.generate_report()
    
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
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Number of top relationships to show (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                       help='Minimum causal strength threshold (default: 0.1)')
    
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
        success = train_for_stock(args.stock, args.num_stocks, args.epochs)
        if not success:
            return
    
    # Analyze
    if args.analyze:
        if not args.train:
            # Check if model exists
            if not os.path.exists(f'checkpoints/{args.stock}_best.pt'):
                print(f"\nWarning: No trained model found for {args.stock}")
                print("Training model first...")
                success = train_for_stock(args.stock, args.num_stocks, args.epochs)
                if not success:
                    return
        
        analyze_stock(args.stock, args.top_k, args.threshold)
    
    # If no action specified, show help
    if not (args.list or args.train or args.analyze):
        parser.print_help()


if __name__ == '__main__':
    main()

