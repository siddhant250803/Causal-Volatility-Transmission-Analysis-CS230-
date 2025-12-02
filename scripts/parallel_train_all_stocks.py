"""
Parallel training script for all stocks.
Optimized for AWS environments with multiple cores.
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.data import StockDataLoader, VolatilityDataset
from src.models import CausalAttentionModel
from src.utils import CausalRegularizedLoss, calculate_metrics
from src.train import Trainer


def train_single_stock(stock_idx: int, stock_name: str, data_dict: dict, config_dict: dict):
    """
    Train model for a single stock. Designed to be called in parallel.
    
    Args:
        stock_idx: Index of target stock
        stock_name: Name of target stock
        data_dict: Dictionary containing preprocessed data
        config_dict: Configuration dictionary
        
    Returns:
        Dictionary with training results
    """
    try:
        # Recreate config from dict
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        print(f"\n[{stock_name}] Starting training (PID: {os.getpid()})")
        
        # Recreate datasets from data dict
        train_dataset = VolatilityDataset(
            data_dict['train_returns'],
            data_dict['train_volatility'],
            data_dict['train_targets'],
            target_stock_idx=stock_idx
        )
        
        val_dataset = VolatilityDataset(
            data_dict['val_returns'],
            data_dict['val_volatility'],
            data_dict['val_targets'],
            target_stock_idx=stock_idx
        )
        
        test_dataset = VolatilityDataset(
            data_dict['test_returns'],
            data_dict['test_volatility'],
            data_dict['test_targets'],
            target_stock_idx=stock_idx
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Train model
        trainer = Trainer(config, stock_idx, stock_name)
        trainer.setup_model(n_stocks=data_dict['n_stocks'])
        trainer.train(train_loader, val_loader)
        
        # Test
        test_losses, test_metrics = trainer.validate(test_loader)
        
        # Save stock info
        stock_info = {
            'stock_names': data_dict['stock_names'],
            'target_stock': stock_name,
            'target_stock_idx': stock_idx
        }
        os.makedirs('checkpoints', exist_ok=True)
        with open(f'checkpoints/{stock_name}_stock_info.json', 'w') as f:
            json.dump(stock_info, f, indent=2)
        
        result = {
            'stock_name': stock_name,
            'stock_idx': stock_idx,
            'test_mse': test_losses['mse_loss'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'success': True,
            'error': None
        }
        
        print(f"[{stock_name}] ✓ Training complete - R²: {test_metrics['r2']:.4f}")
        return result
        
    except Exception as e:
        print(f"[{stock_name}] ✗ Training failed: {e}")
        return {
            'stock_name': stock_name,
            'stock_idx': stock_idx,
            'success': False,
            'error': str(e)
        }


def parallel_train_all_stocks(num_stocks: int = None,
                              epochs: int = None,
                              max_workers: int = None,
                              batch_size: int = None):
    """
    Train models for all stocks in parallel.
    
    Args:
        num_stocks: Number of stocks to analyze (None = all)
        epochs: Number of training epochs per stock
        max_workers: Maximum parallel workers (None = auto-detect)
        batch_size: Batch size for training
    """
    print("="*80)
    print("PARALLEL TRAINING: ALL STOCKS")
    print("="*80)
    
    # Initialize config
    config = Config()
    if epochs is not None:
        config.NUM_EPOCHS = epochs
    if batch_size is not None:
        config.BATCH_SIZE = batch_size
    
    # Determine number of workers
    if max_workers is None:
        # Use 75% of available cores
        max_workers = max(1, int(mp.cpu_count() * 0.75))
    
    print(f"\nConfiguration:")
    print(f"  Max workers: {max_workers}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")
    
    # Load data (this is shared preparation)
    print(f"\nStep 1: Loading and preprocessing data...")
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=num_stocks)
    
    stock_names = data_loader.stock_names
    n_stocks = len(stock_names)
    
    print(f"  Loaded {n_stocks} stocks")
    
    # Preprocess data
    data_loader.compute_realized_volatility()
    
    n_timesteps = len(data_loader.returns)
    train_end_idx = int(n_timesteps * config.TRAIN_SPLIT)
    data_loader.normalize_data(train_end_idx=train_end_idx)
    
    print(f"\nStep 2: Creating sequences...")
    X_returns, X_volatility, y = data_loader.create_sequences()
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Create data dictionary (will be passed to workers)
    data_dict = {
        'train_returns': train_data[0],
        'train_volatility': train_data[1],
        'train_targets': train_data[2],
        'val_returns': val_data[0],
        'val_volatility': val_data[1],
        'val_targets': val_data[2],
        'test_returns': test_data[0],
        'test_volatility': test_data[1],
        'test_targets': test_data[2],
        'stock_names': stock_names,
        'n_stocks': n_stocks
    }
    
    # Convert config to dict for serialization
    config_dict = {
        attr: getattr(config, attr)
        for attr in dir(config)
        if not attr.startswith('_') and not callable(getattr(config, attr))
    }
    
    print(f"\nStep 3: Training models in parallel...")
    print(f"  Training {n_stocks} stocks with {max_workers} parallel workers\n")
    
    # Parallel training
    results = []
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(train_single_stock, idx, name, data_dict, config_dict): name
            for idx, name in enumerate(stock_names)
        }
        
        # Process results as they complete
        with tqdm(total=n_stocks, desc="Training progress") as pbar:
            for future in as_completed(futures):
                stock_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[{stock_name}] ✗ Unexpected error: {e}")
                    results.append({
                        'stock_name': stock_name,
                        'success': False,
                        'error': str(e)
                    })
                pbar.update(1)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total time: {duration/60:.1f} minutes ({duration/3600:.2f} hours)")
    print(f"Average time per stock: {duration/n_stocks:.1f} seconds")
    print(f"\nSuccessful: {len(successful)}/{n_stocks}")
    print(f"Failed: {len(failed)}/{n_stocks}")
    
    if successful:
        results_df = pd.DataFrame(successful)
        print(f"\nPerformance Summary:")
        print(f"  Avg RMSE: {results_df['test_rmse'].mean():.6f}")
        print(f"  Avg R²: {results_df['test_r2'].mean():.4f}")
        print(f"  Best R²: {results_df['test_r2'].max():.4f} ({results_df.loc[results_df['test_r2'].idxmax(), 'stock_name']})")
        print(f"  Worst R²: {results_df['test_r2'].min():.4f} ({results_df.loc[results_df['test_r2'].idxmin(), 'stock_name']})")
    
    if failed:
        print(f"\nFailed stocks:")
        for r in failed:
            print(f"  - {r['stock_name']}: {r['error']}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/parallel_training_{timestamp}.csv"
    
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train models for all stocks in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--num_stocks', type=int, default=None,
                       help='Number of stocks to train (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: from config)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum parallel workers (default: auto = 75%% of cores)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (default: from config)')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Run parallel training
    parallel_train_all_stocks(
        num_stocks=args.num_stocks,
        epochs=args.epochs,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()


