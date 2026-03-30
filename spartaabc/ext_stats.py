"""
Extended Statistics Implementations
Users can add new statistics by decorating functions with @registry.register()
"""
import numpy as np
from spartaabc.stat_registry import registry
from spartaabc.external_utils import get_unique_gap_lengths


# ============================================================================
# QUANTILE STATISTICS
# ============================================================================

@registry.register(
    name="UNIQUE_GAP_Q25",
    description="25th percentile of unique gap lengths",
    category="quantile",
    enabled=True
)
def calculate_q25(gap_lengths: np.ndarray) -> float:
    """Calculate Q25 (25th percentile)"""
    if len(gap_lengths) == 0:
        return 0.0
    return float(np.percentile(gap_lengths, 25, method='nearest'))


@registry.register(
    name="UNIQUE_GAP_Q50",
    description="50th percentile (median) of unique gap lengths",
    category="quantile",
    enabled=True
)
def calculate_q50(gap_lengths: np.ndarray) -> float:
    """Calculate Q50 (median)"""
    if len(gap_lengths) == 0:
        return 0.0
    return float(np.percentile(gap_lengths, 50, method='nearest'))


@registry.register(
    name="UNIQUE_GAP_Q75",
    description="75th percentile of unique gap lengths",
    category="quantile",
    enabled=True
)
def calculate_q75(gap_lengths: np.ndarray) -> float:
    """Calculate Q75 (75th percentile)"""
    if len(gap_lengths) == 0:
        return 0.0
    return float(np.percentile(gap_lengths, 75, method='nearest'))


@registry.register(
    name="UNIQUE_GAP_Q95",
    description="95th percentile - captures long tail events",
    category="quantile",
    enabled=True
)
def calculate_q95(gap_lengths: np.ndarray) -> float:
    """Calculate Q95 (95th percentile)"""
    if len(gap_lengths) == 0:
        return 0.0
    return float(np.percentile(gap_lengths, 95, method='nearest'))


# ============================================================================
# DISTRIBUTION STATISTICS
# ============================================================================

@registry.register(
    name="GAP_VARIANCE",
    description="Variance of gap lengths - measures dispersion",
    category="distribution",
    enabled=True
)
def calculate_variance(gap_lengths: np.ndarray) -> float:
    """Calculate variance"""
    if len(gap_lengths) == 0:
        return 0.0
    return float(np.var(gap_lengths))


@registry.register(
    name="GAP_CV",
    description="Coefficient of Variation (σ/μ) - standardized spread",
    category="distribution",
    enabled=True
)
def calculate_cv(gap_lengths: np.ndarray) -> float:
    """Calculate Coefficient of Variation"""
    if len(gap_lengths) == 0:
        return 0.0
    mean = np.mean(gap_lengths)
    if mean == 0:
        return 0.0
    std = np.std(gap_lengths)
    return float(std / mean)


# ============================================================================
# DIVERSITY/ENTROPY STATISTICS
# ============================================================================

@registry.register(
    name="GAP_SHANNON_ENTROPY",
    description="Shannon entropy: H = -∑ P(li) * log2(P(li))",
    category="diversity",
    enabled=True
)
def calculate_shannon_entropy(gap_lengths: np.ndarray) -> float:
    """
    Calculate Shannon entropy of gap length distribution.
    Measures uncertainty/diversity of gap lengths.
    """
    if len(gap_lengths) == 0:
        return 0.0
    
    # Get frequency of each unique length
    unique_vals, counts = np.unique(gap_lengths, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / counts.sum()
    
    # Calculate Shannon entropy (remove zeros to avoid log(0))
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)


# ============================================================================
# RATIO STATISTICS
# ============================================================================

@registry.register(
    name="GAP_MAX_TO_MEAN",
    description="Max gap length / Mean gap length - outlier detection",
    category="ratio",
    enabled=True
)
def calculate_max_to_mean(gap_lengths: np.ndarray) -> float:
    """Calculate max-to-mean ratio"""
    if len(gap_lengths) == 0:
        return 0.0
    mean = np.mean(gap_lengths)
    if mean == 0:
        return 0.0
    max_val = np.max(gap_lengths)
    return float(max_val / mean)


@registry.register(
    name="GAP_DENSITY_RATIO",
    description="Proportion of alignment that is gaps",
    category="ratio",
    enabled=True
)
def calculate_gap_density_ratio(gap_lengths: np.ndarray) -> float:
    """
    Placeholder — real value is computed from the full MSA sequences
    and injected by simulate_data.py and stats_manager.py using
    external_utils.compute_gap_density(). Returns 0.0 as fallback only.
    """
    return 0.0


# ============================================================================
# EXAMPLE: USER-DEFINED STATISTIC
# ============================================================================

@registry.register(
    name="GAP_LONG_RATIO",
    description="Proportion of gaps longer than 5bp",
    category="custom",
    enabled=False  # Disabled by default - users can enable if needed
)
def calculate_long_gap_ratio(gap_lengths: np.ndarray) -> float:
    """
    Example custom statistic: Proportion of long gaps.
    Users can add their own statistics this easily!
    """
    if len(gap_lengths) == 0:
        return 0.0
    long_gaps = np.sum(gap_lengths > 5)
    total_gaps = len(gap_lengths)
    return float(long_gaps / total_gaps)
