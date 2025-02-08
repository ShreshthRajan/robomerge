from typing import Dict, Any, Optional
import numpy as np

from robomerge.fast_prep import FASTBatch, FASTPreprocessor
from .ingestion import DROIDIngestion
from .transform import DataStandardizer
from .validation import DataValidator, QualityMetrics

class RoboMerge:
    """Main pipeline interface for standardizing robot data for FAST.
    
    Handles:
    1. Data ingestion from DROID format
    2. Standardization and normalization
    3. Quality validation
    4. FAST-compatible output formatting
    """
    
    def __init__(self, 
                 target_freq: float = 50.0,
                 min_completeness: float = 0.99,
                 max_action_jump: float = 0.5,
                 min_temporal_consistency: float = 0.95):
        """Initialize RoboMerge pipeline.
        
        Args:
            target_freq: Target control frequency (Hz)
            min_completeness: Minimum required data completeness (0-1)
            max_action_jump: Maximum allowed action discontinuity
            min_temporal_consistency: Minimum temporal consistency (0-1)
        """
        self.ingestion = DROIDIngestion(target_freq)
        self.standardizer = DataStandardizer(target_freq)
        self.validator = DataValidator(
            min_completeness=min_completeness,
            max_action_jump=max_action_jump,
            min_temporal_consistency=min_temporal_consistency
        )
    
    def process_episode(self, 
                       filepath: str, 
                       raise_on_warning: bool = False) -> Dict[str, Any]:
        """Process a single episode from raw data to FAST-ready format.
        
        Args:
            filepath: Path to DROID episode file
            raise_on_warning: Whether to raise error on quality warnings
        
        Returns:
            Dict containing standardized data and metadata
        
        Raises:
            ValueError: If data quality checks fail and raise_on_warning=True
        """
        # Load and validate raw data
        episode = self.ingestion.load_episode(filepath)
        
        # Standardize to common format
        standardized = self.standardizer.standardize_episode(episode)
        
        # Validate data quality
        quality = self.validator.validate_episode(standardized)
        
        # Add metadata
        standardized['metadata'] = {
            'original_frequency': episode.frequency,
            'target_frequency': self.standardizer.target_freq,
            'action_dims': episode.actions.shape[1],
            'normalized': True,
            'quality_metrics': {
                'completeness': quality.completeness,
                'temporal_consistency': quality.temporal_consistency,
                'action_smoothness': quality.action_smoothness
            },
            'quality_warnings': quality.warnings,
            'processing_info': {
                'source_format': 'DROID',
                'pipeline_version': '0.1.0'
            }
        }
        
        if raise_on_warning and quality.warnings:
            raise ValueError(f"Quality warnings: {quality.warnings}")
        
        return standardized
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about pipeline configuration."""
        return {
            'target_frequency': self.standardizer.target_freq,
            'quality_thresholds': {
                'min_completeness': self.validator.min_completeness,
                'max_action_jump': self.validator.max_action_jump,
                'min_temporal_consistency': self.validator.min_temporal_consistency
            },
            'supported_formats': ['DROID'],
            'output_format': 'FAST-compatible'
        }
    
    def process_for_fast(self, 
                        filepath: str,
                        chunk_size: Optional[int] = None,
                        raise_on_warning: bool = False) -> FASTBatch:
        """Process episode directly to FAST-ready format."""
        # Process through standard pipeline
        standardized = self.process_episode(filepath, raise_on_warning)
        
        # Initialize FAST preprocessor
        fast_prep = FASTPreprocessor(
            chunk_size=chunk_size if chunk_size else int(self.standardizer.target_freq)
        )
        
        # Convert to FAST format
        return fast_prep.prepare_episode(standardized)