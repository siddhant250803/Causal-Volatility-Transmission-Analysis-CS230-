"""
Granger Causality testing for validating learned causal relationships.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GrangerCausalityTester:
    """Test Granger causality between stock volatilities."""
    
    def __init__(self, max_lag: int = 12, significance_level: float = 0.05):
        """
        Initialize Granger causality tester.
        
        Args:
            max_lag: Maximum lag to test (in time steps)
            significance_level: P-value threshold for significance
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        
    def test_pairwise_granger(self, 
                               source_data: np.ndarray, 
                               target_data: np.ndarray,
                               max_lag: int = None) -> Dict:
        """
        Test if source Granger-causes target.
        
        Args:
            source_data: Time series for source variable (T,)
            target_data: Time series for target variable (T,)
            max_lag: Maximum lag to test (defaults to self.max_lag)
            
        Returns:
            Dictionary with test results
        """
        if max_lag is None:
            max_lag = self.max_lag
            
        # Prepare data for Granger test
        # statsmodels expects (T, 2) array with [target, source]
        data = np.column_stack([target_data, source_data])
        
        try:
            # Run Granger causality test
            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract p-values and test statistics for each lag
            p_values = []
            f_stats = []
            best_lag = None
            min_p_value = 1.0
            
            for lag in range(1, max_lag + 1):
                # Use F-test results
                f_test = result[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                p_value = f_test[1]
                
                p_values.append(p_value)
                f_stats.append(f_stat)
                
                if p_value < min_p_value:
                    min_p_value = p_value
                    best_lag = lag
            
            # Determine if there's significant Granger causality
            is_significant = min_p_value < self.significance_level
            
            return {
                'is_granger_cause': is_significant,
                'best_lag': best_lag,
                'min_p_value': min_p_value,
                'p_values': p_values,
                'f_stats': f_stats,
                'all_lags_p_values': {lag: p_values[lag-1] for lag in range(1, max_lag + 1)}
            }
            
        except Exception as e:
            # If test fails (e.g., insufficient data, singular matrix)
            return {
                'is_granger_cause': False,
                'best_lag': None,
                'min_p_value': 1.0,
                'p_values': [1.0] * max_lag,
                'f_stats': [0.0] * max_lag,
                'error': str(e)
            }
    
    def test_all_sources_to_target(self,
                                    all_volatilities: np.ndarray,
                                    target_idx: int,
                                    stock_names: List[str]) -> pd.DataFrame:
        """
        Test Granger causality from all stocks to a target stock.
        
        Args:
            all_volatilities: Volatility time series (T, N) for N stocks
            target_idx: Index of target stock
            stock_names: List of stock names
            
        Returns:
            DataFrame with Granger causality results
        """
        T, N = all_volatilities.shape
        target_data = all_volatilities[:, target_idx]
        
        results = []
        
        for source_idx in range(N):
            if source_idx == target_idx:
                continue  # Skip self-causation
            
            source_data = all_volatilities[:, source_idx]
            
            # Test Granger causality
            test_result = self.test_pairwise_granger(source_data, target_data)
            
            results.append({
                'source_stock': stock_names[source_idx],
                'target_stock': stock_names[target_idx],
                'granger_causes': test_result['is_granger_cause'],
                'best_lag': test_result['best_lag'],
                'p_value': test_result['min_p_value'],
                'f_statistic': max(test_result['f_stats']) if test_result['f_stats'] else 0.0
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('p_value')
        
        return df
    
    def compare_with_attention_gates(self,
                                     granger_results: pd.DataFrame,
                                     attention_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compare Granger causality results with attention-based causal gates.
        
        Args:
            granger_results: DataFrame from test_all_sources_to_target
            attention_results: DataFrame from CausalityAnalyzer.get_causal_relationships
            
        Returns:
            Merged DataFrame comparing both methods
        """
        # Merge on source stock
        comparison = granger_results.merge(
            attention_results,
            on='source_stock',
            how='outer',
            suffixes=('_granger', '_attention')
        )
        
        # Add agreement column
        comparison['methods_agree'] = (
            comparison['granger_causes'].fillna(False) & 
            (comparison['causal_strength'].fillna(0) > 0)
        )
        
        # Reorder columns
        cols = ['source_stock', 'granger_causes', 'p_value', 'best_lag', 
                'causal_strength', 'lag_intervals', 'methods_agree']
        comparison = comparison[cols]
        
        comparison = comparison.sort_values('p_value')
        
        return comparison


def compute_realized_volatility_for_granger(returns: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Compute realized volatility for Granger testing.
    
    Args:
        returns: Returns data (T, N)
        window: Window size for volatility computation
        
    Returns:
        Volatility time series (T, N)
    """
    T, N = returns.shape
    volatility = np.zeros_like(returns)
    
    for t in range(window, T):
        volatility[t] = np.std(returns[t-window:t], axis=0)
    
    # Remove initial window period where volatility is undefined
    return volatility[window:]

