"""
Attention-based causal inference model for volatility transmission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LearnedLagAttention(nn.Module):
    """Attention mechanism with learned lag parameters."""
    
    def __init__(self, n_stocks: int, d_model: int, d_k: int, d_v: int, max_lag: int):
        """
        Initialize lagged attention mechanism.
        
        Args:
            n_stocks: Number of stocks
            d_model: Dimension of model embeddings
            d_k: Dimension of query/key vectors
            d_v: Dimension of value vectors
            max_lag: Maximum lag to consider
        """
        super().__init__()
        self.n_stocks = n_stocks
        self.d_k = d_k
        self.d_v = d_v
        self.max_lag = max_lag
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        
        # Learned lag parameters (initialized uniformly across range)
        self.lags = nn.Parameter(torch.rand(n_stocks) * max_lag)
        
        # Causal gates (controls which stocks influence others)
        # Initialize at 0.0 so sigmoid(0.0) = 0.5 (neutral, let training determine)
        self.causal_gates = nn.Parameter(torch.randn(n_stocks) * 0.1)
        
    def forward(self, query_stock_emb: torch.Tensor, 
                all_stock_embs: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of lagged attention.
        
        Args:
            query_stock_emb: Embedding for target stock (batch_size, d_model)
            all_stock_embs: Embeddings for all stocks (batch_size, lookback, n_stocks, d_model)
            return_attention: Whether to return attention weights
            
        Returns:
            context: Aggregated context vector (batch_size, d_v)
            attention_weights: Attention weights if return_attention=True
        """
        batch_size, lookback, n_stocks, d_model = all_stock_embs.shape
        
        # Compute query for target stock
        q = self.W_q(query_stock_emb)  # (batch_size, d_k)
        
        # Apply causal gates (sigmoid to keep in [0, 1])
        gates = torch.sigmoid(self.causal_gates)  # (n_stocks,)
        
        # Normalize lags to [0, max_lag]
        normalized_lags = torch.sigmoid(self.lags) * self.max_lag  # (n_stocks,)
        
        # Compute keys and values for each stock with learned lags
        # Use soft (differentiable) temporal indexing instead of hard rounding
        context_list = []
        attention_weights_list = []
        
        for stock_idx in range(n_stocks):
            # Get the lag for this stock (continuous value)
            lag = normalized_lags[stock_idx]
            
            # Soft indexing: interpolate between adjacent time steps
            # This makes the operation differentiable
            lag_floor = torch.floor(lag)
            lag_ceil = torch.ceil(lag)
            
            # Clamp to valid range
            lag_floor = torch.clamp(lag_floor, 0, lookback - 1).long()
            lag_ceil = torch.clamp(lag_ceil, 0, lookback - 1).long()
            
            # Interpolation weight
            alpha = lag - lag_floor.float()
            
            # Get embeddings at floor and ceil
            emb_floor = all_stock_embs[:, lag_floor, stock_idx, :]  # (batch_size, d_model)
            emb_ceil = all_stock_embs[:, lag_ceil, stock_idx, :]    # (batch_size, d_model)
            
            # Linearly interpolate (differentiable)
            stock_emb = (1 - alpha) * emb_floor + alpha * emb_ceil  # (batch_size, d_model)
            
            # Compute key and value
            k = self.W_k(stock_emb)  # (batch_size, d_k)
            v = self.W_v(stock_emb)  # (batch_size, d_v)
            
            # Compute attention score
            score = torch.sum(q * k, dim=-1) / np.sqrt(self.d_k)  # (batch_size,)
            
            # Apply causal gate
            score = score * gates[stock_idx]
            
            context_list.append((score, v))
            attention_weights_list.append(score)
        
        # Stack and apply softmax
        scores = torch.stack([s for s, _ in context_list], dim=1)  # (batch_size, n_stocks)
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, n_stocks)
        
        # Aggregate context
        values = torch.stack([v for _, v in context_list], dim=1)  # (batch_size, n_stocks, d_v)
        context = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)  # (batch_size, d_v)
        
        if return_attention:
            return context, attention_weights
        else:
            return context, None


class CausalAttentionModel(nn.Module):
    """
    Attention-based causal inference model for volatility prediction.
    Predicts volatility for a single target stock using information from all stocks.
    """
    
    def __init__(self, n_stocks: int, lookback: int, d_model: int = 64, 
                 d_k: int = 32, d_v: int = 32, max_lag: int = 12, dropout: float = 0.1):
        """
        Initialize the model.
        
        Args:
            n_stocks: Number of stocks
            lookback: Number of historical time steps
            d_model: Dimension of embeddings
            d_k: Dimension of query/key
            d_v: Dimension of value
            max_lag: Maximum lag to consider
            dropout: Dropout rate
        """
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.d_model = d_model
        
        # Embedding layers for stock returns
        self.stock_embedding = nn.Linear(1, d_model)
        # Target stock history + volatility (use MLP for larger lookback)
        self.target_embedding = nn.Sequential(
            nn.Linear(lookback + 1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        # Lagged attention mechanism
        self.attention = LearnedLagAttention(n_stocks, d_model, d_k, d_v, max_lag)
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(d_v + d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, X_returns: torch.Tensor, X_volatility: torch.Tensor, 
                target_stock_idx: int, return_attention: bool = False):
        """
        Forward pass.
        
        Args:
            X_returns: Historical returns (batch_size, lookback, n_stocks)
            X_volatility: Current volatility (batch_size, n_stocks)
            target_stock_idx: Index of target stock to predict
            return_attention: Whether to return attention weights and gates
            
        Returns:
            predictions: Predicted volatility (batch_size, 1)
            attention_info: Dict with attention weights and gates if return_attention=True
        """
        batch_size = X_returns.shape[0]
        
        # Embed all stocks' historical returns
        # (batch_size, lookback, n_stocks, 1) -> (batch_size, lookback, n_stocks, d_model)
        all_stock_embs = self.stock_embedding(X_returns.unsqueeze(-1))
        
        # Embed target stock information (its own history + current volatility)
        target_history = X_returns[:, :, target_stock_idx]  # (batch_size, lookback)
        target_vol = X_volatility[:, target_stock_idx:target_stock_idx+1]  # (batch_size, 1)
        target_info = torch.cat([target_history, target_vol], dim=1)  # (batch_size, lookback+1)
        target_emb = self.target_embedding(target_info)  # (batch_size, d_model)
        
        # Apply attention to get context from other stocks
        context, attention_weights = self.attention(target_emb, all_stock_embs, return_attention)
        
        # Combine context with target embedding
        combined = torch.cat([context, target_emb], dim=-1)  # (batch_size, d_v + d_model)
        
        # Predict volatility
        prediction = self.mlp(combined)  # (batch_size, 1)
        
        if return_attention:
            attention_info = {
                'attention_weights': attention_weights,
                'causal_gates': torch.sigmoid(self.attention.causal_gates),
                'lags': torch.sigmoid(self.attention.lags) * self.attention.max_lag
            }
            return prediction, attention_info
        else:
            return prediction, None
    
    def get_causal_graph(self, target_stock_idx: int) -> dict:
        """
        Extract the causal graph for a target stock.
        
        Args:
            target_stock_idx: Index of target stock
            
        Returns:
            Dictionary with causal gates and lags for all stocks
        """
        gates = torch.sigmoid(self.attention.causal_gates).detach().cpu().numpy()
        lags = (torch.sigmoid(self.attention.lags) * self.attention.max_lag).detach().cpu().numpy()
        
        return {
            'gates': gates,
            'lags': lags
        }

