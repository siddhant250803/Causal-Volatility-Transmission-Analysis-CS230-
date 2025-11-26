# Quick Start: Analyze First 5 Stocks

This guide shows you how to quickly analyze causal relationships between the first 5 stocks in your dataset.

## 🚀 Three Ways to Run

### Option 1: Quick Test with Full Validation (Recommended for First Try)
**Time: ~10-15 minutes**

```bash
python quick_test_5_stocks.py
```

This runs:
- 5 stocks
- 10 epochs each (quick training)
- **Granger testing enabled** (validates results)
- Perfect for testing the fixes!

---

### Option 2: Standard Analysis
**Time: ~15-20 minutes**

```bash
python analyze_first_5_stocks.py
```

Default settings:
- 5 stocks
- 20 epochs each
- Granger testing enabled

---

### Option 3: Custom Analysis
**Time: Varies**

```bash
# Analyze first 10 stocks with 30 epochs
python analyze_first_5_stocks.py --num_stocks 10 --epochs 30

# Fast mode: Skip Granger testing
python analyze_first_5_stocks.py --num_stocks 5 --epochs 15 --skip_granger

# Thorough analysis
python analyze_first_5_stocks.py --num_stocks 5 --epochs 50
```

**Arguments:**
- `--num_stocks N` - Analyze first N stocks (default: 5)
- `--epochs E` - Training epochs per stock (default: 20)
- `--skip_granger` - Skip Granger causality tests (faster)

---

## 📊 What You'll Get

### 1. Individual Results (per stock)
Location: `results/`

For each stock (e.g., AAPL):
- `AAPL_causal_relationships.csv` - All relationships
- `AAPL_comparison.csv` - Attention vs Granger comparison (if enabled)
- `AAPL_granger_causality.csv` - Granger test results (if enabled)

### 2. Combined Results
Location: `results/`

- `first_5_stocks_all_relationships.csv` - All relationships in one file
- `first_5_stocks_matrix.csv` - Source × Target matrix

### 2.5 Granger Causality Results
Location: `results/`

For each stock:
- `[STOCK]_granger_causality.csv` - Statistical validation results
- `[STOCK]_comparison.csv` - Neural network vs Granger comparison
- Shows which relationships are validated by **both methods**

### 3. Visualizations
Location: `plots/`

For each stock:
- `[STOCK]_causal_network.png` - Bar chart of influences
- `[STOCK]_lag_distribution.png` - Lag patterns
- `[STOCK]_heatmap.png` - Visual matrix

### 4. Network Visualization (Optional)

After running the analysis, create network graphs:

```bash
# Network diagram
python visualize_network.py

# With custom threshold
python visualize_network.py --threshold 0.15
```

This creates:
- `plots/causal_network_graph.png` - Network diagram
- `plots/causal_heatmap_5stocks.png` - Combined heatmap

---

## 📋 Example Output

```
================================================================================
ANALYZING CAUSAL RELATIONSHIPS: FIRST 5 STOCKS
================================================================================

Analyzing stocks: AAPL, MSFT, GOOGL, AMZN, NVDA
Total: 5 stocks
Epochs per stock: 20

================================================================================
PHASE 1: TRAINING MODELS
================================================================================

[1/5] Training model for AAPL
Training model for stock: AAPL (index 0)
Epoch 1/20
Train MSE: 0.456789 | Val MSE: 0.467890 | Val RMSE: 0.683883 | Val R²: 0.1234
...
✓ New best model saved!

================================================================================
PHASE 2: ANALYZING CAUSAL RELATIONSHIPS
================================================================================

[1/5] Analyzing relationships for AAPL

Gate statistics:
  Min: 0.001234
  Max: 0.456789
  Mean: 0.123456

Found 4 significant causal relationships

TOP 10 INFLUENCING STOCKS:
source_stock  target_stock  causal_strength  lag_minutes  lag_intervals
        MSFT          AAPL           0.4568        35.2              7
       GOOGL          AAPL           0.3456        42.1              8
        NVDA          AAPL           0.2345        28.5              6
       AMZN          AAPL           0.1567        31.8              6

================================================================================
GRANGER CAUSALITY ANALYSIS
================================================================================

Running Granger causality tests...
✓ Found 4 stocks with significant Granger causality (p < 0.05)

Top 10 Granger-causal relationships:
source_stock      p_value  best_lag  f_statistic
        MSFT  0.00000123         7       45.678
       GOOGL  0.00001234         8       32.456
        NVDA  0.00012345         6       28.901
       AMZN  0.00123456         6       18.234

────────────────────────────────────────────────────────────────────────────────
COMPARISON: Attention Gates vs. Granger Causality
────────────────────────────────────────────────────────────────────────────────
Attention method found:  4 significant relationships
Granger method found:    4 significant relationships
Both methods agree on:   4 relationships

Stocks validated by BOTH methods:
source_stock      p_value  causal_strength
        MSFT  0.00000123           0.4568
       GOOGL  0.00001234           0.3456
        NVDA  0.00012345           0.2345
       AMZN  0.00123456           0.1567

================================================================================
PHASE 3: GENERATING SUMMARY
================================================================================

CAUSAL STRENGTH MATRIX (Source → Target)
              AAPL    MSFT   GOOGL   AMZN   NVDA
AAPL        0.0000  0.3456  0.2345  0.1234  0.4567
MSFT        0.4568  0.0000  0.3210  0.2345  0.3456
GOOGL       0.3456  0.3210  0.0000  0.2567  0.2890
AMZN        0.1567  0.2345  0.2567  0.0000  0.1890
NVDA        0.2345  0.3456  0.2890  0.1890  0.0000

SUMMARY STATISTICS

AAPL:
  Influenced by: 4 stocks
  Mean strength: 0.2984
  Max strength:  0.4568
  Mean lag:      34.4 min
  Top influencer: MSFT (strength=0.4568, lag=35.2min)

[... similar for other stocks ...]

NETWORK STATISTICS
Total relationships found: 20
Average causal strength: 0.2789
Average lag: 33.2 minutes

Most influential stocks (by total outgoing strength):
             sum      mean  count
MSFT      1.5234   0.3809      4
AAPL      1.2345   0.3086      4
NVDA      1.0890   0.2723      4

✓ Analysis complete!
```

---

## ✅ Verification Checklist

After running, verify the fixes worked:

- [ ] Training MSE decreases over epochs (**not stuck**)
- [ ] Causal strengths are **0.1-0.8** range (not ~0.001)
- [ ] Lags are **diverse** (0-60 min, not all at max)
- [ ] R² score **> 0.1** (model explains variance)
- [ ] Different stocks have **different** influence patterns
- [ ] Console shows **no dimension errors**

---

## 🔧 Troubleshooting

### "FileNotFoundError: HF_Returns_Stocks.csv"
Make sure the data file is in the project root directory.

### "Dimension mismatch" errors
Delete old checkpoints: `rm -rf checkpoints/*`
Then run again.

### Very low R² or causal strengths still near zero
- Try more epochs: `--epochs 50`
- Check that old checkpoints were deleted
- Verify you're using the fixed code

### Out of memory
- Reduce number of stocks: `--num_stocks 3`
- Reduce batch size in `config.py`: `BATCH_SIZE = 16`

---

## 📈 What to Look For

### Good Signs (Fixes Working):
✅ Causal strengths vary widely (0.05 - 0.8)
✅ Lags are diverse (not all 60 min)
✅ R² improves over training
✅ Different stocks show different patterns
✅ Network has meaningful structure

### Bad Signs (Issues Remain):
❌ All causal strengths < 0.01
❌ All lags at maximum
❌ R² stays near 0
❌ All stocks look identical
❌ Training loss stuck

---

## 🎯 Next Steps

After analyzing first 5 stocks:

1. **Check results look good** (causal strengths > 0.1, diverse lags)
2. **Scale up** to more stocks:
   ```bash
   python analyze_first_5_stocks.py --num_stocks 20 --epochs 30
   ```
3. **Analyze specific stocks** of interest:
   ```bash
   python run_analysis.py --train --analyze --stock TSLA --num_stocks 50
   ```
4. **Create custom analyses** using the individual CSVs

---

## 💡 Tips

- **Start small** (5 stocks, 10 epochs) to verify fixes
- **Granger testing is included** - validates neural network findings statistically
- **Check plots** in `plots/` directory for visual confirmation
- **Compare matrix** to see which stocks influence each other most
- **Look at lags** - different sectors should have different lag patterns
- **Check comparison.csv** to see agreement between attention and Granger methods
- **Use `--skip_granger`** flag only if you need extra speed for debugging

---

## 📧 Questions?

If you see zero correlation again:
1. Delete ALL checkpoints: `rm -rf checkpoints/*`
2. Verify `config.py` has: `LOOKBACK_WINDOW = 12`, `LAMBDA_GATE = 0.0001`
3. Check you have the latest code (all fixes applied)
4. Try: `python quick_test_5_stocks.py` first

