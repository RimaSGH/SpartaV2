"""
Statistics Manager
Coordinates C++ base statistics with Python extended statistics.
"""
from typing import Dict, List, Optional
import numpy as np
from spartaabc.stat_registry import StatRegistry, registry


class StatsManager:
    """
    Manager that coordinates C++ base statistics with Python extended statistics.
    
    Workflow:
        1. Initialize with C++ wrapper
        2. Calculate 27 base stats via C++
        3. Get gap lengths from C++
        4. Apply registered Python stats
        5. Merge and return combined results
    """
    
    def __init__(
        self, 
        cpp_wrapper,
        stat_registry: Optional[StatRegistry] = None
    ):
        """
        Initialize the statistics manager.
        
        Args:
            cpp_wrapper: C++ wrapper object (e.g., msastats module)
            stat_registry: Custom registry, or use global default
        """
        self.cpp_wrapper = cpp_wrapper
        self.registry = stat_registry or registry
        
        # Cache for base stat names
        self._base_stat_names = None
    
    def get_base_stat_names(self) -> List[str]:
        """Get names of the 27 base C++ statistics"""
        if self._base_stat_names is None:
            # Assumes C++ wrapper has stats_names() method
            self._base_stat_names = self.cpp_wrapper.stats_names()
        return self._base_stat_names
    
    def calculate_base_stats(self, msa: List[str]) -> Dict[str, float]:
        """
        Calculate the 27 base statistics using C++.
        
        Args:
            msa: List of aligned sequences
        
        Returns:
            Dictionary of base statistics
        """
        # Call C++ wrapper
        stat_values = self.cpp_wrapper.calculate_msa_stats(msa)
        stat_names = self.get_base_stat_names()
        
        # Convert to dictionary
        return dict(zip(stat_names, stat_values))
    
    def get_gap_lengths(self, msa: List[str]) -> np.ndarray:
        """
        Extract gap lengths from MSA.
        
        This is the data bridge between C++ and Python.
        
        Args:
            msa: List of aligned sequences
        
        Returns:
            1D numpy array of unique gap lengths
        """
        # Use Python implementation from external_utils
        from spartaabc.external_utils import get_unique_gap_lengths
        return get_unique_gap_lengths(msa)
    
    def calculate_extended_stats(
        self, 
        gap_lengths: np.ndarray,
        enabled_only: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all registered Python statistics.
        
        Args:
            gap_lengths: 1D numpy array of gap lengths
            enabled_only: Only calculate enabled statistics
        
        Returns:
            Dictionary of extended statistics
        """
        if len(gap_lengths) == 0:
            # Return zeros for all stats if no gaps
            return {
                name: 0.0 
                for name in self.registry.list_stats(enabled_only)
            }
        
        return self.registry.calculate_all(gap_lengths, enabled_only)
    
    def calculate_all_stats(
        self, 
        msa: List[str],
        include_base: bool = True,
        include_extended: bool = True,
        enabled_only: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all statistics (base + extended).
        
        Args:
            msa: List of aligned sequences
            include_base: Include 27 C++ base statistics
            include_extended: Include Python extended statistics
            enabled_only: Only include enabled extended stats
        
        Returns:
            Dictionary with all statistics
        """
        results = {}
        
        # Step 1: Calculate base C++ statistics
        if include_base:
            base_stats = self.calculate_base_stats(msa)
            results.update(base_stats)
        
        # Step 2: Get gap lengths (data bridge)
        if include_extended:
            gap_lengths = self.get_gap_lengths(msa)
            
            # Step 3: Calculate extended Python statistics
            extended_stats = self.calculate_extended_stats(
                gap_lengths, 
                enabled_only
            )
            results.update(extended_stats)
        
        return results
    
    def calculate_all_stats_list(
        self,
        msa: List[str],
        enabled_only: bool = True
    ) -> List[float]:
        """
        Calculate all statistics and return as ordered list.
        Compatible with existing code that expects list of values.
        
        Args:
            msa: List of aligned sequences
            enabled_only: Only include enabled extended stats
        
        Returns:
            List of statistic values (base stats + extended stats)
        """
        # Get base stats from C++
        base_stats = self.cpp_wrapper.calculate_msa_stats(msa)
        
        # Get extended stats
        gap_lengths = self.get_gap_lengths(msa)
        extended_stats_dict = self.calculate_extended_stats(gap_lengths, enabled_only)
        
        # Convert extended stats dict to ordered list
        extended_stat_names = sorted(extended_stats_dict.keys())
        extended_stats = [extended_stats_dict[name] for name in extended_stat_names]
        
        # Combine: base stats (list) + extended stats (list)
        return list(base_stats) + extended_stats
    
    def get_stat_vector(
        self, 
        msa: List[str],
        stat_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get statistics as ordered vector (useful for ABC).
        
        Args:
            msa: List of aligned sequences
            stat_names: Specific stats to include (in order), or None for all
        
        Returns:
            1D numpy array of statistic values
        """
        all_stats = self.calculate_all_stats(msa)
        
        if stat_names is None:
            # Use all stats in deterministic order
            stat_names = sorted(all_stats.keys())
        
        return np.array([all_stats[name] for name in stat_names])
    
    def get_stat_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all available statistics.
        
        Returns:
            Dictionary mapping stat names to their metadata
        """
        info = {}
        
        # Base C++ stats
        for name in self.get_base_stat_names():
            info[name] = {
                "source": "cpp",
                "description": "Base C++ statistic",
                "category": "base"
            }
        
        # Extended Python stats
        for name in self.registry.list_stats():
            stat = self.registry.get_stat(name)
            info[name] = {
                "source": "python",
                "description": stat.description,
                "category": stat.category,
                "enabled": stat.enabled
            }
        
        return info


def calculate_all_extended_stats(sequences: List[str], enabled_only: bool = True) -> List[float]:
    """
    Calculate all extended statistics for a list of MSA sequences.

    Returns values as a flat list in registry registration order,
    which matches the SUMSTATS_LIST ordering in utility.py.

    Args:
        sequences: List of MSA sequence strings (no FASTA headers)
        enabled_only: Only include enabled statistics

    Returns:
        List of float values in registration order
    """
    from spartaabc.external_utils import get_unique_gap_lengths, compute_gap_density
    from spartaabc import ext_stats  # ensure all stats are registered

    stat_names = registry.list_stats(enabled_only=enabled_only)

    gap_lengths = get_unique_gap_lengths(sequences)
    if len(gap_lengths) == 0:
        return [0.0] * len(stat_names)

    extended_stats_dict = registry.calculate_all(gap_lengths, enabled_only=enabled_only)
    if 'GAP_DENSITY_RATIO' in extended_stats_dict:
        extended_stats_dict['GAP_DENSITY_RATIO'] = compute_gap_density(sequences)
    return [extended_stats_dict[name] for name in stat_names]


# Factory function for easy instantiation
def create_stats_manager(cpp_wrapper) -> StatsManager:
    """
    Factory function to create a StatsManager instance.
    
    Args:
        cpp_wrapper: C++ wrapper module (e.g., import msastats)
    
    Returns:
        Configured StatsManager instance
    """
    return StatsManager(cpp_wrapper, registry)
