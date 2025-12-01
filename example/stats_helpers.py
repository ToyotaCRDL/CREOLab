"""
Statistical helper functions for evaluation pipeline.
"""

from statistics import mean, stdev
import numpy as np
from scipy import stats


def mean_and_95ci(values):
    """
    Calculate mean and 95% confidence interval for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        tuple: (mean, 95% CI margin) where CI margin is ± value
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    
    m = float(mean(values))
    s = stdev(values)
    n = len(values)
    
    if s == 0 or not np.isfinite(s):
        return m, 0.0
    
    # 95% confidence interval using t-distribution
    t = stats.t.ppf(0.975, df=n-1)  # 97.5% for two-tailed 95% CI
    std_err = s / np.sqrt(n)
    
    return m, float(t * std_err)

