# Critical Fixes Applied - Zero Correlation Issue

## Overview
This document summarizes all critical fixes applied to resolve the zero/extremely low correlation issue in the causal volatility transmission model.

## Date Applied
November 26, 2025

---

## 🔴 CRITICAL BUG FIXES

### 1. **Input Dimension Mismatch in Stock Embedding**
**File:** `models/attention_model.py` (Line 151)

**Problem:**
- Stock embedding layer expected 1D input: `nn.Linear(1, d_model)`
- But received 2D input (returns + volatility): shape `(..., 2)`
- Only first dimension was processed, volatility information was lost

**Fix:**
```python
# OLD: self.stock_embedding = nn.Linear(1, d_model)
# NEW: 
self.stock_embedding = nn.Linear(2, d_model)
```

**Impact:** Now correctly processes both returns and volatility features.

---

### 2. **Input Dimension Mismatch in Target Embedding**
**File:** `models/attention_model.py` (Line 154)

**Problem:**
- Target embedding expected `lookback + 1` features
- But received `lookback * 2` features (history + volatility concatenated)
- With `lookback=84`, passing 168 features to layer expecting 85

**Fix:**
```python
# OLD: nn.Linear(lookback + 1, 128)
# NEW:
nn.Linear(lookback * 2, 128)
```

**Impact:** Correct dimension for combined history and volatility features.

---

### 3. **Data Leakage in Normalization**
**File:** `data/dataloader.py` (Line 100-124)

**Problem:**
- Z-score statistics computed on ALL data (including test set)
- Test set statistics leaked into training normalization
- Model saw future information during training

**Fix:**
```python
def normalize_data(self, train_end_idx: Optional[int] = None):
    """Now accepts train_end_idx to compute stats only on training data"""
    if train_end_idx is None:
        stat_end = len(self.returns)
        print("WARNING: Computing statistics on all data")
    else:
        stat_end = train_end_idx
        print(f"Computing statistics on training data only")
    
    # Compute mean/std only on [:stat_end]
    mean = np.mean(self.returns[:stat_end, i])
    std = np.std(self.returns[:stat_end, i])
```

**Files Updated:**
- `data/dataloader.py` - Modified `normalize_data()` method
- `run_analysis.py` - Added train split calculation before normalization
- `train.py` - Added train split calculation before normalization

**Impact:** Eliminates data leakage, ensures proper train/val/test separation.

---

### 4. **Excessive Regularization Suppressing Learning**
**File:** `config.py` (Lines 25-28)

**Problem:**
- `LAMBDA_GATE = 0.005` with L2 penalty on 100 stocks
- Total penalty dominated MSE loss
- Model learned to minimize gates to zero rather than find true relationships

**Fix:**
```python
# OLD:
LAMBDA_GATE = 0.005
GAMMA_TV = 0.001
ETA_IRM = 0.001
BETA_LAG_DIVERSITY = 0.01

# NEW:
LAMBDA_GATE = 0.0001      # Reduced 50x
GAMMA_TV = 0.0001          # Reduced 10x
ETA_IRM = 0.0001           # Reduced 10x
BETA_LAG_DIVERSITY = 0.0   # Disabled (let data determine lags)
```

**Impact:** Model can now learn meaningful relationships without being forced to zero.

---

### 5. **Volatility Initialization Creating Artificial Patterns**
**File:** `data/dataloader.py` (Lines 95-96)

**Problem:**
- Initial volatility values filled with `volatility[window]`
- Created artificial patterns in early samples
- These contaminated samples included in training

**Fix:**
```python
# OLD: Fill initial values with mean
for i in range(window):
    self.volatility[i] = self.volatility[window]

# NEW: Leave as zero, exclude from training
# (No filling - samples will be skipped)
```

**Also updated `create_sequences()` to skip initial window:**
```python
start_idx = volatility_window
n_samples = n_timesteps - lookback - start_idx
print(f"Skipping first {start_idx} samples (volatility burn-in period)")
```

**Impact:** Removes artificial patterns, cleaner training data.

---

### 6. **Lag Initialization Bias**
**File:** `models/attention_model.py` (Line 38)

**Problem:**
- Lags initialized as `torch.rand(n_stocks) * max_lag`
- Then normalized with `sigmoid(lags) * max_lag`
- Double transformation biased all lags toward maximum (60 minutes)
- Poor gradient flow, no diversity

**Fix:**
```python
# OLD: self.lags = nn.Parameter(torch.rand(n_stocks) * max_lag)
# NEW:
self.lags = nn.Parameter(torch.randn(n_stocks) * 1.0)
# Initialize in logit space for better gradients and diversity
```

**Also improved gate initialization:**
```python
# OLD: self.causal_gates = nn.Parameter(torch.randn(n_stocks) * 0.1)
# NEW:
self.causal_gates = nn.Parameter(torch.randn(n_stocks) * 0.5 + 0.5)
# Start around sigmoid(0.5) ≈ 0.65, allowing learning in both directions
```

**Impact:** Better gradient flow, more diverse learned lags, easier training.

---

### 7. **Lookback Window and Max Lag Mismatch**
**File:** `config.py` (Lines 14, 21)

**Problem:**
- `LOOKBACK_WINDOW = 84` (7 hours)
- `MAX_LAG = 72` (6 hours)
- Need 84 + 72 = 156 historical points for max lag
- But volatility computed with only 84-point window
- Insufficient data for high lags

**Fix:**
```python
# OLD:
LOOKBACK_WINDOW = 84  # 7 hours
MAX_LAG = 72          # 6 hours

# NEW:
LOOKBACK_WINDOW = 12  # 60 minutes (more reasonable)
MAX_LAG = 12          # 60 minutes (matches common financial patterns)
```

**Impact:** Consistent and sufficient data for all lag values.

---

## 📊 EXPECTED IMPROVEMENTS

### Before Fixes:
- Causal strengths: 0.0004 - 0.004 (essentially zero)
- All lags at ~60 minutes (maximum)
- Granger found 100 relationships, neural network found 0
- Model not learning from cross-stock information

### After Fixes:
- ✅ Causal strengths should increase to 0.1 - 0.8 range
- ✅ Diverse lag patterns (0-60 minutes)
- ✅ Better alignment with Granger causality results
- ✅ Improved R² scores (prediction accuracy)
- ✅ Meaningful cross-stock attention patterns

---

## 🔄 HOW TO RE-TRAIN

To benefit from these fixes, you must **retrain your models**:

```bash
# Delete old checkpoints (incompatible due to dimension changes)
rm checkpoints/AAPL_best.pt

# Train with new architecture and parameters
python run_analysis.py --train --stock AAPL --num_stocks 50 --epochs 30

# Analyze results
python run_analysis.py --analyze --stock AAPL --top_k 15
```

**Note:** Old checkpoints are incompatible due to:
1. Changed embedding dimensions (1→2, lookback+1→2*lookback)
2. Different parameter counts
3. Changed lookback window (84→12)

---

## 📝 TECHNICAL DETAILS

### Normalization Flow (Fixed)
```
1. Load raw data
2. Compute volatility
3. Calculate train_end_idx = int(n_timesteps * 0.70)
4. Normalize using ONLY data[:train_end_idx] statistics
5. Apply same normalization to val and test sets
```

### Sequence Creation Flow (Fixed)
```
1. Skip first 'volatility_window' samples (burn-in period)
2. Create sequences from clean data only
3. Ensures no artificial patterns from initialization
```

### Model Input Dimensions (Fixed)
```
Stock features:    (batch, lookback, n_stocks, 2)  ← 2D: returns + volatility
Stock embedding:   Linear(2, 64)                   ← Matches input
Target features:   (batch, lookback*2)              ← History + volatility
Target embedding:  Linear(lookback*2, 128)         ← Matches input
```

---

## 🎯 KEY TAKEAWAYS

1. **Dimension mismatches** completely broke feature processing
2. **Data leakage** gave artificially good train performance but poor generalization
3. **Over-regularization** prevented any meaningful learning
4. **Artificial patterns** from volatility initialization contaminated data
5. **Poor initialization** led to stuck gradients and no diversity

All of these are now **fixed** and the model should train properly.

---

## ✅ VERIFICATION CHECKLIST

After retraining, verify:

- [ ] Training MSE decreases over epochs (not stuck)
- [ ] Validation MSE follows similar pattern (no overfitting)
- [ ] R² score > 0.1 (model explains variance)
- [ ] Causal strengths vary (not all ~0.001)
- [ ] Lag values diverse (not all at max)
- [ ] Some alignment with Granger causality results
- [ ] Attention weights sum to 1.0 and vary across samples

---

## 📧 Questions?

If issues persist after retraining:
1. Check that you deleted old checkpoints
2. Verify data file is correct (HF_Returns_Stocks.csv)
3. Try increasing epochs (--epochs 50)
4. Try fewer stocks first (--num_stocks 30)
5. Check console output for errors during training

**Expected training time:** 5-10 minutes per stock with 50 stocks, 30 epochs

