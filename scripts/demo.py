"""
Demo script to quickly test the framework with minimal data.
Run this after installing dependencies to verify everything works.
"""

import sys
import os

print("="*80)
print("CAUSAL VOLATILITY TRANSMISSION - DEMO")
print("="*80)

# Check if dependencies are installed
try:
    import torch
    import pandas as pd
    import numpy as np
    print("✓ Dependencies installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Check if data file exists
if not os.path.exists('HF_Returns_Stocks.csv'):
    print("✗ Data file 'HF_Returns_Stocks.csv' not found")
    print("  Please ensure the data file is in the project root directory")
    sys.exit(1)
else:
    print("✓ Data file found")

print("\n" + "="*80)
print("QUICK START GUIDE")
print("="*80)

print("""
This framework lets you:
1. Train a model to predict volatility for a target stock
2. Discover which other stocks causally influence the target
3. Visualize the causal relationships with lags

EXAMPLE COMMANDS:

1. List available stocks (first 50):
   python run_analysis.py --list

2. Train model for AAPL:
   python run_analysis.py --train --stock AAPL --num_stocks 20

   Note: Start with fewer stocks (e.g., 20) for faster testing
   
3. Analyze causal relationships:
   python run_analysis.py --analyze --stock AAPL

4. Do both in one command:
   python run_analysis.py --train --analyze --stock AAPL --num_stocks 20

RECOMMENDED FIRST RUN:
  python run_analysis.py --list --num_stocks 20
  python run_analysis.py --train --analyze --stock [PICK_A_STOCK] --num_stocks 20 --epochs 5

This will:
- Use only 20 stocks (faster training)
- Run 5 epochs (quick test)
- Generate full analysis with plots
""")

print("="*80)
print("FRAMEWORK STRUCTURE")
print("="*80)
print("""
The framework is modular:

📁 data/           - Data loading and preprocessing
📁 models/         - Attention-based causal model architecture  
📁 utils/          - Loss functions and metrics
📁 checkpoints/    - Saved trained models
📁 plots/          - Generated visualizations
📁 results/        - CSV reports of causal relationships

Main scripts:
- run_analysis.py     : Interactive interface (USE THIS!)
- train.py            : Direct training script
- analyze_causality.py: Direct analysis script
- config.py           : Configuration parameters
""")

print("="*80)
print("KEY FEATURES")
print("="*80)
print("""
✓ Learns which stocks influence others (causal direction)
✓ Learns strength of influence (0-1 gate values)
✓ Learns time delays of influence (0-60 minute lags)
✓ Interpretable causal graphs with visualizations
✓ Regularized learning (sparse, smooth, stable)
✓ Easy-to-use command-line interface
""")

print("="*80)
print("READY TO START!")
print("="*80)
print("\nRun one of the example commands above to begin.\n")

