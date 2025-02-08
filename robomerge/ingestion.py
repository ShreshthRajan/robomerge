# robomerge/robomerge/ingestion.py

import h5py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class DROIDEpisode:
    """Structured container for DROID episode data."""
    actions: np.ndarray          # Robot actions
    states: np.ndarray          # Robot states
    timestamps: np.ndarray      # Timestamps for synchronization
    images: Dict[str, np.ndarray]  # Camera images
    frequency: float            # Control frequency
    metadata: Dict[str, Any]    # Additional episode info

class DROIDIngestion:
    """Handles loading and initial processing of DROID data."""
    
    def __init__(self, target_freq: float = 50.0):
        self.target_freq = target_freq
        self.expected_keys = ['actions', 'states', 'timestamps', 'images']
    
    def load_episode(self, filepath: str) -> DROIDEpisode:
        """Load and validate a DROID episode."""
        with h5py.File(filepath, 'r') as f:
            # Validate data completeness
            self._validate_keys(f)
            
            # Load core data
            actions = np.array(f['actions'])
            states = np.array(f['states'])
            timestamps = np.array(f['timestamps'])
            
            # Load images with proper formatting
            images = {
                'wrist_cam': np.array(f['images']['wrist']),
                'external_cam': np.array(f['images']['external'])
            }
            
            # Calculate source frequency
            freq = 1.0 / np.mean(np.diff(timestamps))
            
            return DROIDEpisode(
                actions=actions,
                states=states,
                timestamps=timestamps,
                images=images,
                frequency=freq,
                metadata=self._extract_metadata(f)
            )
    
    def _validate_keys(self, file: h5py.File) -> None:
        """Ensure all required data fields are present."""
        missing = [key for key in self.expected_keys if key not in file]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
    
    def _extract_metadata(self, file: h5py.File) -> Dict[str, Any]:
        """Extract episode metadata."""
        return {
            'episode_length': len(file['timestamps']),
            'action_dims': file['actions'].shape[1],
            'state_dims': file['states'].shape[1]
        }