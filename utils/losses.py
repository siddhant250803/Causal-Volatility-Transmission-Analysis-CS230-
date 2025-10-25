"""
Loss functions with causal regularization.
"""

import torch
import torch.nn as nn


class CausalRegularizedLoss(nn.Module):
    """
    Loss function combining MSE with causal regularization terms:
    - Group lasso on causal gates (sparsity)
    - Total variation on attention weights (temporal smoothness)
    - Invariant risk minimization (stability across regimes)
    """
    
    def __init__(self, lambda_gate: float = 0.01, gamma_tv: float = 0.001, eta_irm: float = 0.001):
        """
        Initialize loss function.
        
        Args:
            lambda_gate: Weight for group lasso penalty on gates
            gamma_tv: Weight for total variation penalty
            eta_irm: Weight for IRM penalty
        """
        super().__init__()
        self.lambda_gate = lambda_gate
        self.gamma_tv = gamma_tv
        self.eta_irm = eta_irm
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                model, attention_weights: torch.Tensor = None) -> dict:
        """
        Compute regularized loss.
        
        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth values (batch_size,) or (batch_size, 1)
            model: The model (to access causal gates)
            attention_weights: Attention weights for TV penalty (batch_size, n_stocks)
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Ensure targets have correct shape
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        # MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Group lasso penalty on causal gates (L2,1 norm)
        # Encourages sparsity - only few stocks have causal influence
        causal_gates = torch.sigmoid(model.attention.causal_gates)
        gate_penalty = torch.norm(causal_gates, p=2)
        
        # Total variation penalty (if attention weights provided)
        tv_penalty = torch.tensor(0.0, device=predictions.device)
        if attention_weights is not None and self.gamma_tv > 0:
            # Penalize rapid changes in attention across stocks
            # Sort by stock index and compute differences
            tv_penalty = torch.mean(torch.abs(attention_weights[:, 1:] - attention_weights[:, :-1]))
        
        # IRM penalty (simplified version - variance of losses across batch)
        # Encourages consistent performance across different samples/regimes
        irm_penalty = torch.tensor(0.0, device=predictions.device)
        if self.eta_irm > 0:
            # Split batch into two "environments" and compute variance of gradients
            mid = len(predictions) // 2
            if mid > 1:
                loss_env1 = self.mse(predictions[:mid], targets[:mid])
                loss_env2 = self.mse(predictions[mid:], targets[mid:])
                irm_penalty = torch.abs(loss_env1 - loss_env2)
        
        # Total loss
        total_loss = (mse_loss + 
                     self.lambda_gate * gate_penalty + 
                     self.gamma_tv * tv_penalty + 
                     self.eta_irm * irm_penalty)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss.item(),
            'gate_penalty': gate_penalty.item(),
            'tv_penalty': tv_penalty.item(),
            'irm_penalty': irm_penalty.item()
        }

