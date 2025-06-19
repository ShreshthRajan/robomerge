# robomerge/robomerge/rtc_prep.py
"""
Real-Time Chunking preprocessing module for deployment.

Implements RTC methodology from Physical Intelligence's
"Real-Time Action Chunking with Large Models" paper (June 2025).
"""

# Re-export main classes
from .rtc_preprocessor import (
    RTCBatch,
    RealTimeChunkingPreprocessor
)

__all__ = [
    'RTCBatch',
    'RealTimeChunkingPreprocessor'
]