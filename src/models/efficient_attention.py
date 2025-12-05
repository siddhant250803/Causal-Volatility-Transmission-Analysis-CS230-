"""
Efficient Attention Model with FULL LAG SPECTRUM Analysis.

Simplified: Uses attention weights directly as causal influence measure.
No separate gating - the lag spectrum attention IS the causal discovery.

Lag Range: 2 to MAX_LAG+2 intervals (default 2-14 = 10-70 minutes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict


# Minimum lag to ensure causal (not contemporaneous) relationships
MIN_LAG = 2  # 2 intervals = 10 minutes minimum lead time


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Sinusoidal encoding for inductive bias
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos_indices = torch.arange(seq_len, device=x.device)
        learned_pe = self.pos_embedding(pos_indices)
        fixed_pe = self.pe[:seq_len]
        alpha = torch.sigmoid(self.alpha)
        return self.dropout(x + alpha * learned_pe + (1 - alpha) * fixed_pe)


class LagSpectrumAttention(nn.Module):
    """
    Computes attention at EACH lag value separately.
    
    The attention weights ARE the causal influence measure:
    - attn_weights[i, j, lag] = how much stock j at time (t-lag) influences stock i at time t
    """
    
    def __init__(self, d_model: int, n_heads: int, min_lag: int = 2, 
                 max_lag: int = 12, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.n_lags = max_lag - min_lag + 1
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable lag embeddings
        self.lag_embeddings = nn.Parameter(torch.randn(self.n_lags, d_model) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, 
                return_lag_spectrum: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, n_stocks, d_model)
            return_lag_spectrum: if True, return attention weights at each lag
            
        Returns:
            output: (batch, seq_len, n_stocks, d_model)
            lag_spectrum: (batch, n_stocks, n_stocks, n_lags) - attention at each lag
        """
        batch_size, seq_len, n_stocks, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, n_stocks, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, n_stocks, self.n_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, n_stocks, self.n_heads, self.d_k)
        
        lag_attention_weights = []
        lag_outputs = []
        
        for lag_idx, lag in enumerate(range(self.min_lag, self.max_lag + 1)):
            if lag >= seq_len:
                continue
            
            # Query at time t, Key/Value at time (t - lag)
            q_t = q[:, lag:, :, :, :]  # (batch, T-lag, n_stocks, n_heads, d_k)
            k_lag = k[:, :-lag, :, :, :]
            v_lag = v[:, :-lag, :, :, :]
            
            # Add lag embedding to distinguish different lags
            lag_emb = self.lag_embeddings[lag_idx].view(1, 1, 1, self.n_heads, self.d_k)
            k_lag = k_lag + lag_emb
            
            # Attention scores: how much each source stock influences each target stock
            scores = torch.einsum('btihc,btjhc->bthij', q_t, k_lag) / self.scale
            
            # Softmax over source stocks
            attn = F.softmax(scores, dim=-1)  # (batch, T-lag, n_heads, n_stocks_target, n_stocks_source)
            attn = self.dropout(attn)
            
            # Store attention weights (averaged over time and heads)
            # This IS the causal influence at this lag
            avg_attn = attn.mean(dim=(1, 2))  # (batch, n_stocks_target, n_stocks_source)
            lag_attention_weights.append(avg_attn)
            
            # Apply attention
            out = torch.einsum('bthij,btjhc->btihc', attn, v_lag)
            lag_outputs.append(out)
        
        # Combine outputs across lags
        if lag_outputs:
            max_len = max(o.size(1) for o in lag_outputs)
            padded = []
            for out in lag_outputs:
                if out.size(1) < max_len:
                    out = F.pad(out, (0, 0, 0, 0, 0, 0, max_len - out.size(1), 0))
                padded.append(out)
            
            combined = torch.stack(padded, dim=-1).mean(dim=-1)
            combined = combined.reshape(batch_size, -1, n_stocks, d_model)
            
            if combined.size(1) < seq_len:
                combined = F.pad(combined, (0, 0, 0, 0, seq_len - combined.size(1), 0))
            
            output = self.out_proj(combined)
        else:
            output = torch.zeros_like(x)
        
        # Build lag spectrum: (batch, n_stocks_target, n_stocks_source, n_lags)
        lag_spectrum = None
        if return_lag_spectrum and lag_attention_weights:
            lag_spectrum = torch.stack(lag_attention_weights, dim=-1)
        
        return output, lag_spectrum


class LatentSpaceEncoder(nn.Module):
    """Projects stocks into latent space for clustering."""
    
    def __init__(self, d_model: int, latent_dim: int = 32, n_clusters: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim)
        )
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        distances = torch.cdist(latent, self.cluster_centers.unsqueeze(0).expand(x.size(0), -1, -1))
        cluster_probs = F.softmax(-distances, dim=-1)
        return latent, cluster_probs


class EfficientCausalModel(nn.Module):
    """
    Simplified causal model - attention weights ARE the causal measure.
    
    No separate gating network. The lag spectrum attention directly outputs:
    - Predictions (volatility forecast)
    - Causal influence at each lag (from attention weights)
    """
    
    def __init__(self, n_stocks: int, lookback: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, max_lag: int = 12,
                 latent_dim: int = 32, n_clusters: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_lag = max_lag
        self.min_lag = MIN_LAG
        self.n_lags = max_lag - MIN_LAG + 1
        
        # Input
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_len=lookback + 10, dropout=dropout)
        self.stock_embedding = nn.Embedding(n_stocks, d_model)
        
        # Lag spectrum attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': LagSpectrumAttention(d_model, n_heads, min_lag=MIN_LAG, 
                                                   max_lag=max_lag, dropout=dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        
        # Latent space (for clustering similar stocks)
        self.latent_encoder = LatentSpaceEncoder(d_model, latent_dim, n_clusters)
        
        # Prediction head - simplified
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, X_returns: torch.Tensor, X_volatility: torch.Tensor,
                target_stock_idx: int = None,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, lookback, n_stocks = X_returns.shape
        device = X_returns.device
        
        # Input embedding
        x = torch.stack([X_returns, X_volatility], dim=-1)
        x = self.input_proj(x)
        
        # Add stock embeddings
        stock_ids = torch.arange(n_stocks, device=device)
        x = x + self.stock_embedding(stock_ids).unsqueeze(0).unsqueeze(1)
        
        # Positional encoding
        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * n_stocks, lookback, -1)
        x_flat = self.pos_encoding(x_flat)
        x = x_flat.reshape(batch_size, n_stocks, lookback, -1).permute(0, 2, 1, 3)
        
        # Collect lag spectrums from all layers
        all_lag_spectrums = []
        
        for layer in self.layers:
            attn_out, lag_spectrum = layer['attention'](x, return_lag_spectrum=True)
            x = layer['norm1'](x + attn_out)
            x = layer['norm2'](x + layer['ffn'](x))
            
            if lag_spectrum is not None:
                all_lag_spectrums.append(lag_spectrum)
        
        # Temporal pooling: mean over time
        # x shape: (batch, seq_len, n_stocks, d_model)
        stock_reps = x.mean(dim=1)  # (batch, n_stocks, d_model)
        
        # Latent space
        latent, cluster_probs = self.latent_encoder(stock_reps)
        
        # Aggregate lag spectrum (average across layers)
        if all_lag_spectrums:
            # (batch, n_stocks_target, n_stocks_source, n_lags)
            lag_spectrum = torch.stack(all_lag_spectrums, dim=0).mean(dim=0)
        else:
            lag_spectrum = None
        
        # Prediction
        if target_stock_idx is not None:
            target_rep = stock_reps[:, target_stock_idx, :]
            pred = self.pred_head(target_rep)
        else:
            pred = self.pred_head(stock_reps).squeeze(-1)
        
        result = {
            'predictions': pred,
            'lag_spectrum': lag_spectrum,  # This IS the causal influence!
            'latent': latent,
            'cluster_probs': cluster_probs
        }
        
        if return_all:
            result['stock_reps'] = stock_reps
            result['all_lag_spectrums'] = all_lag_spectrums
        
        return result
    
    def get_causal_graph(self, X_returns: torch.Tensor, X_volatility: torch.Tensor) -> Dict:
        """
        Extract causal relationships directly from attention weights.
        
        Returns:
            - lag_spectrum: (n_stocks, n_stocks, n_lags) - influence at each lag
            - causal_strength: (n_stocks, n_stocks) - total influence (sum over lags)
            - peak_lags: (n_stocks, n_stocks) - lag with maximum influence
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(X_returns, X_volatility, return_all=True)
            
            if result['lag_spectrum'] is not None:
                # Average over batch
                lag_spectrum = result['lag_spectrum'].mean(0).cpu().numpy()
                
                # Total causal strength = sum of influence across all lags
                causal_strength = lag_spectrum.sum(axis=-1)
                
                # Peak lag = lag with maximum influence
                peak_lag_indices = np.argmax(lag_spectrum, axis=-1)
                peak_lags = peak_lag_indices + self.min_lag
            else:
                lag_spectrum = np.zeros((self.n_stocks, self.n_stocks, self.n_lags))
                causal_strength = np.zeros((self.n_stocks, self.n_stocks))
                peak_lags = np.full((self.n_stocks, self.n_stocks), self.min_lag)
            
            latent_avg = result['latent'].mean(0).cpu().numpy()
            cluster_probs_avg = result['cluster_probs'].mean(0).cpu().numpy()
            
            lag_values = list(range(self.min_lag, self.max_lag + 1))
            
            return {
                'lag_spectrum': lag_spectrum,
                'causal_strength': causal_strength,  # Replaces 'gates'
                'peak_lags': peak_lags,
                'lag_values': lag_values,
                'lag_values_minutes': [l * 5 for l in lag_values],
                'latent': latent_avg,
                'cluster_probs': cluster_probs_avg,
                'n_lags': self.n_lags,
                'min_lag': self.min_lag,
                'max_lag': self.max_lag
            }


class EfficientLoss(nn.Module):
    """Simplified loss - just MSE + optional regularization."""
    
    def __init__(self, lambda_cluster: float = 0.001):
        super().__init__()
        self.lambda_cluster = lambda_cluster
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                gates: torch.Tensor, latent: torch.Tensor,
                cluster_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        mse_loss = F.mse_loss(predictions.squeeze(), targets.squeeze())
        
        # Encourage diverse cluster usage
        cluster_usage = cluster_probs.mean(dim=[0, 1])
        cluster_entropy = -(cluster_usage * (cluster_usage + 1e-8).log()).sum()
        cluster_loss = -cluster_entropy
        
        total_loss = mse_loss + self.lambda_cluster * cluster_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss.item(),
            'cluster_loss': cluster_loss.item(),
            'gate_loss': 0.0,  # Kept for compatibility
            'contrastive_loss': 0.0  # Kept for compatibility
        }
