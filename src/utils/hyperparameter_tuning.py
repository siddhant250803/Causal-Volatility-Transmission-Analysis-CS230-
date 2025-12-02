"""
Hyperparameter tuning using randomized search.
Supports parallel execution for AWS environments.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from copy import deepcopy
import random

from src.config import Config
from src.models import CausalAttentionModel
from src.utils import CausalRegularizedLoss, calculate_metrics


class HyperparameterSearch:
    """Randomized hyperparameter search with early stopping."""
    
    def __init__(self, base_config: Config, search_space: Dict, n_trials: int = 20):
        """
        Initialize hyperparameter search.
        
        Args:
            base_config: Base configuration to modify
            search_space: Dictionary defining search space for each parameter
            n_trials: Number of random trials to run
        """
        self.base_config = base_config
        self.search_space = search_space
        self.n_trials = n_trials
        self.results = []
        
    def sample_config(self) -> Config:
        """Sample a random configuration from the search space."""
        config = deepcopy(self.base_config)
        
        sampled_params = {}
        for param, spec in self.search_space.items():
            if spec['type'] == 'uniform':
                value = np.random.uniform(spec['low'], spec['high'])
            elif spec['type'] == 'log_uniform':
                log_low = np.log10(spec['low'])
                log_high = np.log10(spec['high'])
                value = 10 ** np.random.uniform(log_low, log_high)
            elif spec['type'] == 'int_uniform':
                value = np.random.randint(spec['low'], spec['high'] + 1)
            elif spec['type'] == 'choice':
                value = np.random.choice(spec['values'])
            else:
                raise ValueError(f"Unknown type: {spec['type']}")
            
            sampled_params[param] = value
            setattr(config, param, value)
        
        return config, sampled_params
    
    def evaluate_config(self, config: Config, 
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       n_stocks: int,
                       target_stock_idx: int,
                       max_epochs: int = 30) -> Dict:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration to evaluate
            train_loader: Training data loader
            val_loader: Validation data loader
            n_stocks: Number of stocks
            target_stock_idx: Target stock index
            max_epochs: Maximum training epochs
            
        Returns:
            Dictionary with evaluation results
        """
        # Initialize model
        model = CausalAttentionModel(
            n_stocks=n_stocks,
            lookback=config.LOOKBACK_WINDOW,
            d_model=config.D_MODEL,
            d_k=config.D_K,
            d_v=config.D_V,
            max_lag=config.MAX_LAG,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = CausalRegularizedLoss(
            lambda_gate=config.LAMBDA_GATE,
            gamma_tv=config.GAMMA_TV,
            eta_irm=config.ETA_IRM,
            beta_lag_diversity=config.BETA_LAG_DIVERSITY
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping for tuning
        
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                returns_seq, volatility_seq, target_volatility = batch
                returns_seq = returns_seq.to(config.DEVICE)
                volatility_seq = volatility_seq.to(config.DEVICE)
                target_volatility = target_volatility.to(config.DEVICE)
                
                optimizer.zero_grad()
                pred, causal_graph = model(returns_seq, volatility_seq, target_stock_idx)
                
                loss_dict = criterion(pred, target_volatility, causal_graph)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    returns_seq, volatility_seq, target_volatility = batch
                    returns_seq = returns_seq.to(config.DEVICE)
                    volatility_seq = volatility_seq.to(config.DEVICE)
                    target_volatility = target_volatility.to(config.DEVICE)
                    
                    pred, causal_graph = model(returns_seq, volatility_seq, target_stock_idx)
                    loss_dict = criterion(pred, target_volatility, causal_graph)
                    val_loss += loss_dict['total_loss'].item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Calculate final metrics on validation set
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                returns_seq, volatility_seq, target_volatility = batch
                returns_seq = returns_seq.to(config.DEVICE)
                volatility_seq = volatility_seq.to(config.DEVICE)
                
                pred, _ = model(returns_seq, volatility_seq, target_stock_idx)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target_volatility.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = calculate_metrics(all_preds, all_targets)
        
        return {
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'epochs_trained': len(train_losses),
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        }
    
    def run_search(self, train_loader: DataLoader,
                  val_loader: DataLoader,
                  n_stocks: int,
                  target_stock_idx: int,
                  stock_name: str,
                  max_epochs_per_trial: int = 30,
                  save_results: bool = True) -> Tuple[Config, Dict]:
        """
        Run randomized hyperparameter search.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_stocks: Number of stocks
            target_stock_idx: Target stock index
            stock_name: Name of target stock
            max_epochs_per_trial: Maximum epochs per trial
            save_results: Whether to save results to file
            
        Returns:
            Tuple of (best_config, best_results)
        """
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER SEARCH: {stock_name}")
        print(f"{'='*80}")
        print(f"Search space: {len(self.search_space)} parameters")
        print(f"Number of trials: {self.n_trials}")
        print(f"Max epochs per trial: {max_epochs_per_trial}\n")
        
        best_val_loss = float('inf')
        best_config = None
        best_result = None
        
        for trial in range(self.n_trials):
            print(f"\n{'─'*80}")
            print(f"Trial {trial + 1}/{self.n_trials}")
            print(f"{'─'*80}")
            
            # Sample configuration
            config, sampled_params = self.sample_config()
            
            print("Sampled parameters:")
            for param, value in sampled_params.items():
                print(f"  {param}: {value:.6f}" if isinstance(value, float) else f"  {param}: {value}")
            
            # Evaluate configuration
            try:
                result = self.evaluate_config(
                    config, train_loader, val_loader,
                    n_stocks, target_stock_idx, max_epochs_per_trial
                )
                
                result['trial'] = trial + 1
                result['params'] = sampled_params
                result['timestamp'] = datetime.now().isoformat()
                
                self.results.append(result)
                
                print(f"\nResults:")
                print(f"  Val loss: {result['best_val_loss']:.6f}")
                print(f"  RMSE: {result['rmse']:.6f}")
                print(f"  R²: {result['r2']:.4f}")
                print(f"  Epochs: {result['epochs_trained']}")
                
                # Track best
                if result['best_val_loss'] < best_val_loss:
                    best_val_loss = result['best_val_loss']
                    best_config = config
                    best_result = result
                    print(f"  ⭐ NEW BEST!")
                
            except Exception as e:
                print(f"  ✗ Trial failed: {e}")
                continue
        
        # Summary
        print(f"\n{'='*80}")
        print("SEARCH COMPLETE")
        print(f"{'='*80}\n")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best trial: {best_result['trial']}")
        print(f"\nBest parameters:")
        for param, value in best_result['params'].items():
            print(f"  {param}: {value:.6f}" if isinstance(value, float) else f"  {param}: {value}")
        
        # Save results
        if save_results:
            os.makedirs('hyperparameter_search', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"hyperparameter_search/{stock_name}_{timestamp}.json"
            
            with open(results_path, 'w') as f:
                json.dump({
                    'stock_name': stock_name,
                    'n_trials': self.n_trials,
                    'search_space': self.search_space,
                    'best_result': best_result,
                    'all_results': self.results
                }, f, indent=2, default=str)
            
            print(f"\n✓ Results saved to: {results_path}")
        
        return best_config, best_result
    
    def plot_search_results(self, stock_name: str, save_path: Optional[str] = None):
        """
        Plot hyperparameter search results.
        
        Args:
            stock_name: Name of stock
            save_path: Optional custom save path
        """
        import matplotlib.pyplot as plt
        
        if len(self.results) == 0:
            print("No results to plot")
            return
        
        df = pd.DataFrame([{
            'trial': r['trial'],
            'val_loss': r['best_val_loss'],
            'rmse': r['rmse'],
            'r2': r['r2'],
            **r['params']
        } for r in self.results])
        
        # Create figure
        n_params = len(self.search_space)
        n_rows = (n_params + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        fig.suptitle(f'Hyperparameter Search Results: {stock_name}',
                    fontsize=16, fontweight='bold')
        
        # Plot each parameter vs validation loss
        for idx, (param, spec) in enumerate(self.search_space.items()):
            ax = axes[idx]
            
            scatter = ax.scatter(df[param], df['val_loss'],
                               c=df['r2'], cmap='RdYlGn',
                               s=100, alpha=0.6, edgecolors='black')
            
            ax.set_xlabel(param, fontsize=11, fontweight='bold')
            ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
            ax.set_title(f'{param} vs Val Loss', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if spec['type'] in ['log_uniform']:
                ax.set_xscale('log')
            
            plt.colorbar(scatter, ax=ax, label='R²')
        
        # Hide extra subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = f"hyperparameter_search/{stock_name}_search_plot.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Search plot saved to: {save_path}")
        plt.close()


def get_default_search_space() -> Dict:
    """Get default search space for hyperparameters."""
    return {
        'LEARNING_RATE': {
            'type': 'log_uniform',
            'low': 1e-4,
            'high': 1e-2
        },
        'D_MODEL': {
            'type': 'choice',
            'values': [32, 64, 128, 256]
        },
        'D_K': {
            'type': 'choice',
            'values': [16, 32, 64]
        },
        'DROPOUT': {
            'type': 'uniform',
            'low': 0.0,
            'high': 0.3
        },
        'LAMBDA_GATE': {
            'type': 'log_uniform',
            'low': 1e-5,
            'high': 1e-3
        },
        'GAMMA_TV': {
            'type': 'log_uniform',
            'low': 1e-5,
            'high': 1e-3
        },
        'BATCH_SIZE': {
            'type': 'choice',
            'values': [16, 32, 64]
        }
    }


