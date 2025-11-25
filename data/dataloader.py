"""
Data loading and preprocessing for high-frequency stock returns.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from config import Config


class StockDataLoader:
    """Loads and preprocesses high-frequency stock returns data."""
    
    def __init__(self, data_path: str, config: Config):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing stock returns
            config: Configuration object
        """
        self.config = config
        self.data_path = data_path
        self.data = None
        self.stock_names = None
        self.dates = None
        self.times = None
        self.returns = None
        self.volatility = None
        self.normalized_returns = None
        self.normalized_volatility = None
        
    def load_data(self, max_stocks: Optional[int] = None):
        """
        Load data from CSV file.
        
        Args:
            max_stocks: Maximum number of stocks to load (None for all)
        """
        print("Loading data...")
        # Read CSV, skip the first row which is a description
        # Use latin-1 encoding to handle copyright symbol in header
        self.data = pd.read_csv(self.data_path, skiprows=1, low_memory=False, encoding='latin-1')
        
        # Extract date and time columns
        self.dates = self.data['Date'].values
        self.times = self.data['Time'].values
        
        # Extract stock names (all columns except Date and Time)
        self.stock_names = [col for col in self.data.columns if col not in ['Date', 'Time']]
        
        # Limit number of stocks if specified
        if max_stocks is not None:
            self.stock_names = self.stock_names[:max_stocks]
        
        # Extract returns data
        self.returns = self.data[self.stock_names].values.astype(np.float32)
        
        # Replace missing values with 0
        self.returns[np.abs(self.returns - self.config.MISSING_VALUE) < 1e-10] = 0
        
        # Handle NaN values (replace with 0)
        nan_count = np.isnan(self.returns).sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values, replacing with 0")
            self.returns = np.nan_to_num(self.returns, nan=0.0)
        
        print(f"Loaded data: {len(self.dates)} time steps, {len(self.stock_names)} stocks")
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        
    def compute_realized_volatility(self, window: int = None):
        """
        Compute realized volatility over a rolling window.
        
        Args:
            window: Rolling window size (default: from config)
        """
        if window is None:
            window = self.config.LOOKBACK_WINDOW
            
        print(f"Computing realized volatility with window={window}...")
        
        n_timesteps, n_stocks = self.returns.shape
        self.volatility = np.zeros_like(self.returns)
        
        # Compute rolling standard deviation
        for i in range(window, n_timesteps):
            # RMS of returns over window
            self.volatility[i] = np.sqrt(np.mean(self.returns[i-window:i]**2, axis=0))
        
        #MIA: maybe exclude those in training part?
        # Set initial values to mean volatility to avoid zeros
        for i in range(window):
            self.volatility[i] = self.volatility[window]
            
        print(f"Volatility computed. Mean: {np.mean(self.volatility):.6f}, Std: {np.std(self.volatility):.6f}")
        
    def normalize_data(self):
        """Normalize returns and volatility per stock using z-score."""
        print("Normalizing data...")
        
        # Normalize returns per stock
        self.normalized_returns = np.zeros_like(self.returns)
        for i in range(len(self.stock_names)):
            mean = np.mean(self.returns[:, i])
            std = np.std(self.returns[:, i])
            if std > 0:
                self.normalized_returns[:, i] = (self.returns[:, i] - mean) / std
            else:
                self.normalized_returns[:, i] = 0
                
        # Normalize volatility per stock
        self.normalized_volatility = np.zeros_like(self.volatility)
        for i in range(len(self.stock_names)):
            mean = np.mean(self.volatility[:, i])
            std = np.std(self.volatility[:, i])
            if std > 0:
                self.normalized_volatility[:, i] = (self.volatility[:, i] - mean) / std
            else:
                self.normalized_volatility[:, i] = 0
                
        print("Normalization complete.")
        
    def create_sequences(self, lookback: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for training.
        
        Args:
            lookback: Number of past intervals to include
            
        Returns:
            X_returns: (n_samples, lookback, n_stocks) - historical returns
            X_volatility: (n_samples, n_stocks) - current volatility
            y: (n_samples, n_stocks) - target volatility (next interval)
        """
        if lookback is None:
            lookback = self.config.LOOKBACK_WINDOW
            
        print(f"Creating sequences with lookback={lookback}...")
        
        n_timesteps, n_stocks = self.normalized_returns.shape
        n_samples = n_timesteps - lookback
        
        X_returns = np.zeros((n_samples, lookback, n_stocks), dtype=np.float32)
        X_volatility = np.zeros((n_samples, n_stocks), dtype=np.float32)
        y = np.zeros((n_samples, n_stocks), dtype=np.float32)
        
        for i in range(n_samples):
            X_returns[i] = self.normalized_returns[i:i+lookback]
            X_volatility[i] = self.normalized_volatility[i+lookback-1]
            y[i] = self.normalized_volatility[i+lookback]
            
        print(f"Created {n_samples} sequences.")
        return X_returns, X_volatility, y
    
    def split_data(self, X_returns, X_volatility, y) -> Tuple:
        """
        Split data into train, validation, and test sets chronologically.
        
        Returns:
            Tuple of (train, val, test) datasets
        """
        n_samples = len(X_returns)
        train_end = int(n_samples * self.config.TRAIN_SPLIT)
        val_end = int(n_samples * (self.config.TRAIN_SPLIT + self.config.VAL_SPLIT))
        
        # Train
        X_ret_train = X_returns[:train_end]
        X_vol_train = X_volatility[:train_end]
        y_train = y[:train_end]
        
        # Validation
        X_ret_val = X_returns[train_end:val_end]
        X_vol_val = X_volatility[train_end:val_end]
        y_val = y[train_end:val_end]
        
        # Test
        X_ret_test = X_returns[val_end:]
        X_vol_test = X_volatility[val_end:]
        y_test = y[val_end:]
        
        print(f"Data split: Train={len(X_ret_train)}, Val={len(X_ret_val)}, Test={len(X_ret_test)}")
        
        return (X_ret_train, X_vol_train, y_train), \
               (X_ret_val, X_vol_val, y_val), \
               (X_ret_test, X_vol_test, y_test)


class VolatilityDataset(Dataset):
    """PyTorch Dataset for volatility prediction."""
    
    def __init__(self, X_returns, X_volatility, y, target_stock_idx: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            X_returns: Historical returns (n_samples, lookback, n_stocks)
            X_volatility: Current volatility (n_samples, n_stocks)
            y: Target volatility (n_samples, n_stocks)
            target_stock_idx: If specified, only predict for this stock
        """
        self.X_returns = torch.FloatTensor(X_returns)
        self.X_volatility = torch.FloatTensor(X_volatility)
        self.y = torch.FloatTensor(y)
        self.target_stock_idx = target_stock_idx
        
    def __len__(self):
        return len(self.X_returns)
    
    def __getitem__(self, idx):
        if self.target_stock_idx is not None:
            # Return only the target stock's prediction
            return (self.X_returns[idx], 
                   self.X_volatility[idx], 
                   self.y[idx, self.target_stock_idx])
        else:
            return (self.X_returns[idx], 
                   self.X_volatility[idx], 
                   self.y[idx])

