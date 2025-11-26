"""
Configuration file for the causal volatility transmission framework.
"""

import torch

class Config:
    """Configuration parameters for the model and training."""
    
    # Data parameters
    DATA_PATH = "HF_Returns_Stocks.csv"
    MISSING_VALUE = -1.04e-07  # Placeholder for missing data in the dataset
    LOOKBACK_WINDOW = 12  # Number of 5-min intervals to look back (60 minutes)
    PREDICTION_HORIZON = 1  # Predict 1 interval ahead (5 minutes)
    
    # Model parameters
    D_MODEL = 64  # Dimension of embeddings
    D_K = 32  # Dimension of query/key vectors
    D_V = 32  # Dimension of value vectors
    N_HEADS = 4  # Number of attention heads
    MAX_LAG = 12  # Maximum lag to consider (in 5-min intervals = 60 minutes)
    DROPOUT = 0.1
    
    # Regularization parameters
    LAMBDA_GATE = 0.0001  # Group lasso penalty for causal gates (much lighter - was suppressing learning)
    GAMMA_TV = 0.0001  # Total variation penalty for attention smoothness (reduced)
    ETA_IRM = 0.0001  # Invariant risk minimization weight (reduced)
    BETA_LAG_DIVERSITY = 0.0  # Lag diversity penalty (disabled - let data determine lags)
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 10
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Analysis parameters
    CAUSAL_THRESHOLD = 0.5  # Threshold for considering a causal relationship significant (relative to max)
    TOP_K_INFLUENCES = 15  # Number of top influencing stocks to display
    USE_RELATIVE_THRESHOLD = True  # Use threshold relative to max gate value
    
    # Visualization
    SAVE_PLOTS = True
    PLOT_DIR = "plots/"
    
    def __repr__(self):
        return f"Config(d_model={self.D_MODEL}, batch_size={self.BATCH_SIZE}, device={self.DEVICE})"

