"""
Statistic Registry System
Provides decorator-based registration for custom statistics.
"""
from typing import Callable, Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class StatDefinition:
    """Definition of a registered statistic"""
    name: str
    func: Callable
    description: str
    category: str = "general"
    enabled: bool = True


class StatRegistry:
    """
    Registry for managing extended statistics.
    Uses decorator pattern for easy registration.
    """
    
    def __init__(self):
        self._stats: Dict[str, StatDefinition] = {}
    
    def register(
        self, 
        name: str, 
        description: str = "",
        category: str = "general",
        enabled: bool = True
    ):
        """
        Decorator to register a statistic function.
        
        Usage:
            @registry.register("Q95", description="95th percentile")
            def calculate_q95(gap_lengths: np.ndarray) -> float:
                return np.percentile(gap_lengths, 95)
        
        Args:
            name: Unique name for the statistic
            description: Human-readable description
            category: Category (e.g., "quantile", "distribution", "diversity")
            enabled: Whether statistic is enabled by default
        """
        def decorator(func: Callable) -> Callable:
            if name in self._stats:
                raise ValueError(f"Statistic '{name}' already registered")
            
            self._stats[name] = StatDefinition(
                name=name,
                func=func,
                description=description or func.__doc__ or "",
                category=category,
                enabled=enabled
            )
            return func
        return decorator
    
    def get_stat(self, name: str) -> StatDefinition:
        """Get a registered statistic by name"""
        if name not in self._stats:
            raise KeyError(f"Statistic '{name}' not found")
        return self._stats[name]
    
    def list_stats(self, enabled_only: bool = False) -> List[str]:
        """List all registered statistic names"""
        if enabled_only:
            return [name for name, stat in self._stats.items() if stat.enabled]
        return list(self._stats.keys())
    
    def get_by_category(self, category: str) -> Dict[str, StatDefinition]:
        """Get all statistics in a category"""
        return {
            name: stat 
            for name, stat in self._stats.items() 
            if stat.category == category
        }
    
    def enable(self, name: str):
        """Enable a statistic"""
        self._stats[name].enabled = True
    
    def disable(self, name: str):
        """Disable a statistic"""
        self._stats[name].enabled = False
    
    def calculate(self, name: str, gap_lengths: np.ndarray) -> float:
        """Calculate a single statistic"""
        stat = self.get_stat(name)
        if not stat.enabled:
            raise RuntimeError(f"Statistic '{name}' is disabled")
        return float(stat.func(gap_lengths))
    
    def calculate_all(
        self, 
        gap_lengths: np.ndarray, 
        enabled_only: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all registered statistics.
        
        Args:
            gap_lengths: 1D numpy array of gap lengths
            enabled_only: Only calculate enabled statistics
        
        Returns:
            Dictionary mapping statistic names to values
        """
        results = {}
        
        for name, stat in self._stats.items():
            if enabled_only and not stat.enabled:
                continue
            
            try:
                results[name] = float(stat.func(gap_lengths))
            except Exception as e:
                # Log error and use 0.0 as fallback
                import logging
                logging.warning(f"Error calculating {name}: {e}")
                results[name] = 0.0
        
        return results
    
    def __len__(self) -> int:
        """Number of registered statistics"""
        return len(self._stats)
    
    def __contains__(self, name: str) -> bool:
        """Check if statistic is registered"""
        return name in self._stats


# Global registry instance
registry = StatRegistry()
