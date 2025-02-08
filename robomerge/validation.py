# robomerge/robomerge/validation.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class QualityMetrics:
    """Container for data quality metrics."""
    completeness: float  # Percentage of valid data points
    temporal_consistency: float  # Measure of timestamp regularity
    action_smoothness: float  # Measure of action trajectory smoothness
    warnings: List[str]  # Any quality warnings

class DataValidator:
    """Validates data quality and consistency."""
    
    def __init__(self, 
                 min_completeness: float = 0.99,
                 max_action_jump: float = 0.5,
                 min_temporal_consistency: float = 0.95):
        self.min_completeness = min_completeness
        self.max_action_jump = max_action_jump
        self.min_temporal_consistency = min_temporal_consistency
    
    def validate_episode(self, 
                        standardized_data: Dict[str, np.ndarray]) -> QualityMetrics:
        """Validate standardized episode data."""
        warnings = []
        
        # Check data completeness
        completeness = self._check_completeness(standardized_data)
        if completeness < self.min_completeness:
            warnings.append(f"Data completeness below threshold: {completeness:.2f}")
        
        # Check temporal consistency
        temporal = self._check_temporal_consistency(
            standardized_data['timestamps'])
        if temporal < self.min_temporal_consistency:
            warnings.append(f"Poor temporal consistency: {temporal:.2f}")
        
        # Check action smoothness
        smoothness = self._check_action_smoothness(
            standardized_data['actions'])
        if smoothness > self.max_action_jump:
            warnings.append(f"Large action discontinuities detected: {smoothness:.2f}")
        
        return QualityMetrics(
            completeness=completeness,
            temporal_consistency=temporal,
            action_smoothness=smoothness,
            warnings=warnings
        )
    
    def _check_completeness(self, data: Dict[str, np.ndarray]) -> float:
        """Check for missing or invalid values."""
        # Check for NaNs and infinities
        valid_fractions = []
        for key, array in data.items():
            if isinstance(array, np.ndarray):
                valid = np.isfinite(array).sum() / array.size
                valid_fractions.append(valid)
        return np.mean(valid_fractions)
    
    def _check_temporal_consistency(self, timestamps: np.ndarray) -> float:
        """Check regularity of timestamp spacing."""
        diffs = np.diff(timestamps)
        mean_diff = np.mean(diffs)
        # Measure consistency as fraction within 10% of mean
        consistency = np.sum(np.abs(diffs - mean_diff) < 0.1 * mean_diff) / len(diffs)
        return consistency
    
    def _check_action_smoothness(self, actions: np.ndarray) -> float:
        """Check for sudden jumps in action values."""
        action_diffs = np.diff(actions, axis=0)
        max_jump = np.max(np.abs(action_diffs))
        return max_jump