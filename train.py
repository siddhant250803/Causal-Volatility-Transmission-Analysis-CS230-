"""
Training script for causal volatility transmission model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json

from config import Config
from data import StockDataLoader, VolatilityDataset
from models import CausalAttentionModel
from utils import CausalRegularizedLoss, calculate_metrics


class Trainer:
    """Trainer for the causal attention model."""
    
    def __init__(self, config: Config, target_stock_idx: int, stock_name: str):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            target_stock_idx: Index of target stock to predict
            stock_name: Name of target stock
        """
        self.config = config
        self.target_stock_idx = target_stock_idx
        self.stock_name = stock_name
        self.device = config.DEVICE
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_model(self, n_stocks: int):
        """
        Setup model, optimizer, and loss function.
        
        Args:
            n_stocks: Number of stocks in the dataset
        """
        self.model = CausalAttentionModel(
            n_stocks=n_stocks,
            lookback=self.config.LOOKBACK_WINDOW,
            d_model=self.config.D_MODEL,
            d_k=self.config.D_K,
            d_v=self.config.D_V,
            max_lag=self.config.MAX_LAG,
            dropout=self.config.DROPOUT
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        self.criterion = CausalRegularizedLoss(
            lambda_gate=self.config.LAMBDA_GATE,
            gamma_tv=self.config.GAMMA_TV,
            eta_irm=self.config.ETA_IRM
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc='Training')
        for X_returns, X_volatility, y in pbar:
            X_returns = X_returns.to(self.device)
            X_volatility = X_volatility.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, attention_info = self.model(
                X_returns, X_volatility, self.target_stock_idx, return_attention=True
            )
            
            # Compute loss
            loss_dict = self.criterion(
                predictions, y, self.model, 
                attention_info['attention_weights'] if attention_info else None
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_dict['mse_loss']:.4f}"})
        
        # Average losses
        avg_losses = {
            key: np.mean([d[key] for d in epoch_losses]) 
            for key in epoch_losses[0].keys() if key != 'total_loss'
        }
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_returns, X_volatility, y in val_loader:
                X_returns = X_returns.to(self.device)
                X_volatility = X_volatility.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                predictions, _ = self.model(X_returns, X_volatility, self.target_stock_idx)
                
                # Compute loss
                loss_dict = self.criterion(predictions, y, self.model)
                val_losses.append(loss_dict)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Average losses
        avg_losses = {
            key: np.mean([d[key] for d in val_losses]) 
            for key in val_losses[0].keys() if key != 'total_loss'
        }
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return avg_losses, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nTraining model for stock: {self.stock_name} (index {self.target_stock_idx})")
        print("=" * 80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_losses)
            
            # Print results
            print(f"Train MSE: {train_losses['mse_loss']:.6f} | "
                  f"Val MSE: {val_losses['mse_loss']:.6f} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"Val R²: {val_metrics['r2']:.4f}")
            
            # Early stopping
            if val_losses['mse_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['mse_loss']
                self.patience_counter = 0
                self.save_checkpoint('best')
                print("✓ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        self.load_checkpoint('best')
        print("\nTraining complete!")
        
    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save model checkpoint."""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'target_stock_idx': self.target_stock_idx,
            'stock_name': self.stock_name
        }
        path = f'checkpoints/{self.stock_name}_{name}.pt'
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, name: str = 'checkpoint'):
        """Load model checkpoint."""
        path = f'checkpoints/{self.stock_name}_{name}.pt'
        if os.path.exists(path):
            # weights_only=False needed for PyTorch 2.6+ when checkpoint contains numpy arrays
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"No checkpoint found at {path}")


def main():
    """Main training function."""
    # Initialize config
    config = Config()
    print(f"Using device: {config.DEVICE}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data_loader = StockDataLoader(config.DATA_PATH, config)
    data_loader.load_data(max_stocks=50)  # Start with 50 stocks for faster testing
    data_loader.compute_realized_volatility()
    data_loader.normalize_data()
    
    # Create sequences
    X_returns, X_volatility, y = data_loader.create_sequences()
    
    # Split data
    train_data, val_data, test_data = data_loader.split_data(X_returns, X_volatility, y)
    
    # Select target stock for testing (e.g., first stock)
    target_stock_idx = 0
    target_stock_name = data_loader.stock_names[target_stock_idx]
    
    print(f"\n" + "="*80)
    print(f"TRAINING FOR TARGET STOCK: {target_stock_name}")
    print("="*80)
    
    # Create datasets
    train_dataset = VolatilityDataset(*train_data, target_stock_idx=target_stock_idx)
    val_dataset = VolatilityDataset(*val_data, target_stock_idx=target_stock_idx)
    test_dataset = VolatilityDataset(*test_data, target_stock_idx=target_stock_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Train model
    trainer = Trainer(config, target_stock_idx, target_stock_name)
    trainer.setup_model(n_stocks=len(data_loader.stock_names))
    trainer.train(train_loader, val_loader)
    
    # Test
    print("\n" + "="*80)
    print("TESTING")
    print("="*80)
    test_losses, test_metrics = trainer.validate(test_loader)
    print(f"Test MSE: {test_losses['mse_loss']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    # Save stock names for later analysis
    stock_info = {
        'stock_names': data_loader.stock_names,
        'target_stock': target_stock_name,
        'target_stock_idx': target_stock_idx
    }
    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{target_stock_name}_stock_info.json', 'w') as f:
        json.dump(stock_info, f, indent=2)
    
    print("\nTraining complete! Use analyze_causality.py to explore causal relationships.")


if __name__ == '__main__':
    main()

