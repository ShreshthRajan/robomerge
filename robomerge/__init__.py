"""
RoboMerge: Enhanced Robot Data Preprocessing Pipeline
====================================================

A comprehensive preprocessing pipeline for standardizing robot data for 
FAST tokenization, Knowledge Insulation (KI), and Real-Time Chunking (RTC).

Features:
- FAST tokenization for efficient robot action representation
- Knowledge Insulation (KI) support for π₀.₅ + KI training
- Real-Time Chunking (RTC) for low-latency deployment
- Operations Dashboard for enterprise-scale monitoring
- Multi-robot data standardization and quality validation

Compatible with Physical Intelligence's latest VLA architectures.
"""

__version__ = "0.3.0"
__author__ = "Shreshth Rajan"

# Core components
from .ingestion import DROIDIngestion
from .transform import DataStandardizer
from .validation import DataValidator, QualityMetrics
from .fast_prep import FASTPreprocessor, FASTBatch

# Advanced features
from .main import RoboMerge

# KI components (Knowledge Insulation)
try:
    from .ki_prep import KnowledgeInsulationPreprocessor, KIBatch, VLMDataMixer
    __has_ki__ = True
except ImportError:
    __has_ki__ = False

# RTC components (Real-Time Chunking)
try:
    from .rtc_prep import RealTimeChunkingPreprocessor, RTCBatch
    __has_rtc__ = True
except ImportError:
    __has_rtc__ = False

# Operations Dashboard
try:
    from .ops_dashboard import OperationsDashboard, OperatorMetrics, ProcessingQueueItem, QualityAlert
    __has_dashboard__ = True
except ImportError:
    __has_dashboard__ = False

# Convenience imports
__all__ = [
    # Core
    'RoboMerge',
    'DROIDIngestion',
    'DataStandardizer', 
    'DataValidator',
    'QualityMetrics',
    'FASTPreprocessor',
    'FASTBatch',
    
    # KI
    'KnowledgeInsulationPreprocessor',
    'KIBatch', 
    'VLMDataMixer',
    
    # RTC
    'RealTimeChunkingPreprocessor',
    'RTCBatch',
    
    # Dashboard
    'OperationsDashboard',
    'OperatorMetrics',
    'ProcessingQueueItem',
    'QualityAlert'
]

def get_info():
    """Get RoboMerge package information."""
    features = {
        'ki_support': __has_ki__,
        'rtc_support': __has_rtc__, 
        'dashboard_support': __has_dashboard__
    }
    
    return {
        'version': __version__,
        'author': __author__,
        'features': features,
        'supported_formats': ['DROID'],
        'output_formats': ['FAST-compatible', 'KI-compatible', 'RTC-compatible'],
        'description': 'Enhanced robot data preprocessing pipeline with PI methodology support'
    }