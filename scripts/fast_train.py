#!/usr/bin/env python3
"""
FAST Training Script for Causal Volatility Model.

This script is optimized for speed and efficiency:
- Batch processing of multiple stocks simultaneously
- Mixed precision training (AMP)
- Shared data loading (load once, train many)
- Progress tracking and ETA
- Training visualization

Usage:
    python scripts/fast_train.py --num_stocks 50 --epochs 30
    python scripts/fast_train.py --all --epochs 20 --batch_size 128
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.data import StockDataLoader, VolatilityDataset
from src.efficient_trainer import EfficientTrainer, TrainingVisualizer
from src.models.efficient_attention import EfficientCausalModel


def prepare_shared_data(config: Config, num_stocks: int = None,
                       tickers: list = None) -> dict:
    """
    Load and preprocess data once for all stocks.
    This is the key optimization - avoid reloading data for each stock.
    
    Args:
        config: Configuration object
        num_stocks: Max number of stocks (ignored if tickers provided)
        tickers: List of specific ticker symbols to load
    """
    print("\n" + "="*70)
    print("LOADING DATA (shared across all stocks)")
    print("="*70)
    
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=num_stocks, tickers=tickers)
    data_loader.compute_realized_volatility()
    
    n_timesteps = len(data_loader.returns)
    train_end_idx = int(n_timesteps * config.TRAIN_SPLIT)
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    X_returns, X_volatility, y = data_loader.create_sequences()
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'stock_names': data_loader.stock_names,
        'n_stocks': len(data_loader.stock_names)
    }


def train_single_stock_fast(stock_idx: int, stock_name: str, 
                           shared_data: dict, config: Config,
                           visualize: bool = False,
                           n_layers: int = 2,
                           turbo: bool = False) -> dict:
    """
    Train model for a single stock using efficient trainer.
    
    This function assumes data is already loaded and preprocessed.
    """
    try:
        # Create datasets
        train_dataset = VolatilityDataset(
            *shared_data['train'], 
            target_stock_idx=stock_idx
        )
        val_dataset = VolatilityDataset(
            *shared_data['val'],
            target_stock_idx=stock_idx
        )
        test_dataset = VolatilityDataset(
            *shared_data['test'],
            target_stock_idx=stock_idx
        )
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # In-process for speed
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize trainer (quiet mode unless visualizing)
        trainer = EfficientTrainer(
            config, 
            stock_idx, 
            stock_name,
            use_amp=torch.cuda.is_available(),
            visualize=visualize,
            viz_update_freq=10
        )
        trainer.setup_model(n_stocks=shared_data['n_stocks'], n_layers=n_layers)
        
        # Train (with tqdm disabled for cleaner output in batch mode)
        # Turbo mode uses aggressive early stopping
        early_patience = 3 if turbo else config.EARLY_STOPPING_PATIENCE
        result = trainer.train(train_loader, val_loader, early_stopping_patience=early_patience)
        
        # Test
        test_loss, test_metrics = trainer.validate(test_loader)
        
        # Extract causal relationships
        causal_graph = trainer.get_causal_graph(val_loader)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Create causal relationships DataFrame
        relationships = []
        gates = causal_graph['gates']
        lags = causal_graph['lags']
        
        for i, source_name in enumerate(shared_data['stock_names']):
            if i != stock_idx:  # Exclude self-relationships
                gate_value = gates[i, stock_idx] if gates.ndim > 1 else gates[i]
                lag_value = lags[i, stock_idx] if lags.ndim > 1 else lags[i]
                
                if gate_value > 0.1:  # Threshold for significance
                    relationships.append({
                        'source_stock': source_name,
                        'target_stock': stock_name,
                        'causal_strength': float(gate_value),
                        'lag_minutes': float(lag_value) * 5,  # Convert to minutes
                        'lag_intervals': float(lag_value)
                    })
        
        if relationships:
            rel_df = pd.DataFrame(relationships)
            rel_df = rel_df.sort_values('causal_strength', ascending=False)
            rel_df.to_csv(f'results/{stock_name}_causal_relationships.csv', index=False)
        
        # Save stock info
        os.makedirs('checkpoints', exist_ok=True)
        stock_info = {
            'stock_names': shared_data['stock_names'],
            'target_stock': stock_name,
            'target_stock_idx': stock_idx,
            'test_metrics': test_metrics,
            'n_relationships': len(relationships)
        }
        with open(f'checkpoints/{stock_name}_stock_info.json', 'w') as f:
            json.dump(stock_info, f, indent=2)
        
        return {
            'stock_name': stock_name,
            'stock_idx': stock_idx,
            'test_mse': test_loss['mse_loss'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'n_relationships': len(relationships),
            'training_time': result['total_time'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        import traceback
        return {
            'stock_name': stock_name,
            'stock_idx': stock_idx,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def fast_train_all_stocks(num_stocks: int = None, 
                         epochs: int = 30,
                         batch_size: int = 64,
                         visualize_first: bool = True,
                         save_all_visualizations: bool = False,
                         tickers: list = None,
                         n_layers: int = 2,
                         turbo: bool = False):
    """
    Fast training for all stocks with shared data loading.
    
    Key optimizations:
    1. Load data once for all stocks
    2. Sequential training (avoids multiprocessing overhead)
    3. Progress tracking with ETA
    4. Memory-efficient batch processing
    
    Args:
        num_stocks: Number of stocks (ignored if tickers provided)
        epochs: Number of training epochs
        batch_size: Batch size for training
        visualize_first: Visualize first stock training
        save_all_visualizations: Save visualizations for all stocks
        tickers: List of specific ticker symbols
        n_layers: Number of transformer layers
        turbo: Enable turbo mode (faster early stopping)
    """
    print("\n" + "="*70)
    print("⚡ FAST TRAINING MODE")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize config
    config = Config()
    config.NUM_EPOCHS = epochs
    config.BATCH_SIZE = batch_size
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Transformer layers: {n_layers}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Mixed Precision: {'Yes' if torch.cuda.is_available() else 'No (CPU)'}")
    print(f"  Turbo mode: {'ON 🚀' if turbo else 'OFF'}")
    if tickers:
        print(f"  Tickers: {len(tickers)} specified")
    
    # Load data once
    shared_data = prepare_shared_data(config, num_stocks, tickers=tickers)
    n_stocks = shared_data['n_stocks']
    stock_names = shared_data['stock_names']
    
    print(f"\n{'='*70}")
    print(f"TRAINING {n_stocks} STOCKS SEQUENTIALLY")
    print(f"{'='*70}\n")
    
    # Track results
    results = []
    times = []
    
    # Training loop with progress bar
    pbar = tqdm(enumerate(stock_names), total=n_stocks, desc="Training stocks")
    
    for idx, stock_name in pbar:
        stock_start = time.time()
        
        # Visualize first stock only by default (for demo purposes)
        visualize = visualize_first and idx == 0
        if save_all_visualizations:
            visualize = True
        
        # Train this stock
        result = train_single_stock_fast(
            idx, stock_name, shared_data, config, visualize=visualize,
            n_layers=n_layers, turbo=turbo
        )
        results.append(result)
        
        stock_time = time.time() - stock_start
        times.append(stock_time)
        
        # Update progress bar
        if result['success']:
            pbar.set_postfix({
                'stock': stock_name,
                'R²': f"{result['test_r2']:.3f}",
                'time': f"{stock_time:.0f}s"
            })
        else:
            pbar.set_postfix({'stock': stock_name, 'status': 'FAILED'})
        
        # Print ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        eta = avg_time * (n_stocks - idx - 1)
        
        if idx % 5 == 0 and idx > 0:
            print(f"\n  Progress: {idx+1}/{n_stocks} | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta/60:.1f}min")
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per stock: {total_time/n_stocks:.1f} seconds")
    print(f"Successful: {len(successful)}/{n_stocks}")
    print(f"Failed: {len(failed)}/{n_stocks}")
    
    if successful:
        results_df = pd.DataFrame(successful)
        print(f"\n📊 Performance Summary:")
        print(f"  Avg RMSE: {results_df['test_rmse'].mean():.6f}")
        print(f"  Avg R²: {results_df['test_r2'].mean():.4f}")
        print(f"  Best R²: {results_df['test_r2'].max():.4f} ({results_df.loc[results_df['test_r2'].idxmax(), 'stock_name']})")
        
        total_relationships = results_df['n_relationships'].sum()
        print(f"\n🔗 Causal Relationships:")
        print(f"  Total found: {total_relationships}")
        print(f"  Avg per stock: {total_relationships/len(successful):.1f}")
    
    if failed:
        print(f"\n⚠️  Failed stocks:")
        for r in failed[:5]:  # Show first 5
            print(f"  - {r['stock_name']}: {r['error']}")
    
    # Save summary
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"results/fast_training_{timestamp}.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n✓ Results saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fast training for causal volatility model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on first 30 stocks with 20 epochs
  python scripts/fast_train.py --num_stocks 30 --epochs 20

  # Train all stocks with larger batch size
  python scripts/fast_train.py --all --batch_size 128 --epochs 15

  # Train on specific tickers
  python scripts/fast_train.py --tickers AAPL,MSFT,GOOG,JPM,BAC --epochs 25

  # Quick test on 10 stocks
  python scripts/fast_train.py --num_stocks 10 --epochs 10 --visualize_all
        """
    )
    
    parser.add_argument('--num_stocks', type=int, default=30,
                       help='Number of stocks to train (default: 30)')
    parser.add_argument('--all', action='store_true',
                       help='Train on all available stocks')
    parser.add_argument('--tickers', type=str, default=None,
                       help='Comma-separated list of ticker symbols (e.g., AAPL,MSFT,JPM)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--visualize_all', action='store_true',
                       help='Save training visualizations for all stocks')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable all visualizations')
    parser.add_argument('--turbo', action='store_true',
                       help='TURBO MODE: Maximize speed (larger batch, simpler model, no viz)')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of transformer layers (default: 2, use 1 for speed)')
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    num_stocks = None if (args.all or tickers) else args.num_stocks
    
    # TURBO MODE overrides
    epochs = args.epochs
    batch_size = args.batch_size
    n_layers = args.n_layers
    visualize = not args.no_visualize
    visualize_all = args.visualize_all
    
    if args.turbo:
        print("\n🚀 TURBO MODE ACTIVATED!")
        print("   - Batch size: 256")
        print("   - Layers: 1")
        print("   - Visualizations: OFF")
        print("   - Early stopping: 3 epochs\n")
        batch_size = 256
        n_layers = 1
        visualize = False
        visualize_all = False
    
    fast_train_all_stocks(
        num_stocks=num_stocks,
        epochs=epochs,
        batch_size=batch_size,
        visualize_first=visualize,
        save_all_visualizations=visualize_all,
        tickers=tickers,
        n_layers=n_layers,
        turbo=args.turbo
    )


if __name__ == '__main__':
    main()

