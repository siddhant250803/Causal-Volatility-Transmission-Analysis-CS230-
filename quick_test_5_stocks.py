"""
Quick test script for first 5 stocks.
Includes Granger testing for full validation.
"""

from analyze_first_5_stocks import analyze_first_5_stocks

if __name__ == '__main__':
    print("="*80)
    print("QUICK TEST: First 5 Stocks (With Granger Validation)")
    print("="*80)
    print("\nSettings:")
    print("  - 5 stocks")
    print("  - 10 epochs per stock (quick training)")
    print("  - Granger testing: ENABLED (validates results)")
    print("  - Total estimated time: ~10-15 minutes")
    print("\nPress Ctrl+C to cancel...")
    print("="*80 + "\n")
    
    analyze_first_5_stocks(
        num_stocks=5,
        epochs=10,           # Quick training
        skip_granger=False   # Keep Granger testing
    )
    
    print("\n" + "="*80)
    print("✓ Quick test complete!")
    print("="*80)
    print("\nTo run full analysis with more epochs and Granger testing:")
    print("  python analyze_first_5_stocks.py --epochs 30")
    print()

