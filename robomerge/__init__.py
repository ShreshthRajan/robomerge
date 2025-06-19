"""
RoboMerge: Enhanced Robot Data Preprocessing Pipeline

A comprehensive pipeline for standardizing robot data for FAST, Knowledge Insulation (KI),
and Real-Time Chunking (RTC) following Physical Intelligence's latest methodologies.

Version: 0.3.0
Compatible with: π₀, π₀.₅ + KI, RTC
"""

from .main import RoboMerge
from .fast_prep import FASTBatch, FASTPreprocessor
from .ki_prep import KIBatch, KnowledgeInsulationPreprocessor, VLMDataMixer
from .rtc_prep import RTCBatch, RealTimeChunkingPreprocessor
from .ops_dashboard import OperationsDashboard, OperatorMetrics, QualityAlert

__version__ = "0.3.0"
__author__ = "Shreshth Rajan"
__description__ = "Robot data preprocessing pipeline for Physical Intelligence VLA training and deployment"

__all__ = [
    # Main pipeline
    'RoboMerge',
    
    # FAST components
    'FASTBatch',
    'FASTPreprocessor',
    
    # Knowledge Insulation components
    'KIBatch', 
    'KnowledgeInsulationPreprocessor',
    'VLMDataMixer',
    
    # Real-Time Chunking components
    'RTCBatch',
    'RealTimeChunkingPreprocessor',
    
    # Operations Dashboard components
    'OperationsDashboard',
    'OperatorMetrics',
    'QualityAlert'
]