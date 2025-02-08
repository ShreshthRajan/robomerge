# robomerge/robomerge/fast_prep.py

from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class FASTBatch:
    """FAST-compatible data batch."""
    action_chunks: np.ndarray  # Shape: (N, chunk_size, action_dims)
    observations: Dict[str, np.ndarray]  # Camera images, state, etc.
    metadata: Dict[str, Any]

class FASTPreprocessor:
    """Prepares standardized data for FAST tokenization."""
    
    def __init__(self, 
                 chunk_size: int = 50,  # 1 second at 50Hz
                 chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def prepare_episode(self, 
                       standardized_data: Dict[str, np.ndarray]) -> FASTBatch:
        """Convert standardized episode into FAST-ready format."""
        # Create action chunks
        action_chunks = self._create_chunks(
            standardized_data['actions'],
            self.chunk_size,
            self.chunk_overlap
        )
        
        # Prepare observations
        observations = self._prepare_observations(standardized_data)
        
        # Prepare metadata for FAST
        metadata = self._prepare_metadata(
            standardized_data['metadata'],
            action_chunks.shape
        )
        
        return FASTBatch(
            action_chunks=action_chunks,
            observations=observations,
            metadata=metadata
        )
    
    def _create_chunks(self, 
                      actions: np.ndarray,
                      chunk_size: int,
                      overlap: int) -> np.ndarray:
        """Create overlapping action chunks."""
        num_actions = len(actions)
        stride = chunk_size - overlap
        
        # Calculate number of chunks
        num_chunks = max(1, (num_actions - overlap) // stride)
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * stride
            end_idx = start_idx + chunk_size
            
            # Handle last chunk
            if end_idx > num_actions:
                # Pad with zeros if necessary
                chunk = np.zeros((chunk_size, actions.shape[1]))
                chunk[:num_actions-start_idx] = actions[start_idx:num_actions]
            else:
                chunk = actions[start_idx:end_idx]
            
            chunks.append(chunk)
        
        return np.array(chunks)
    
    def _prepare_observations(self, 
                            data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare observation dictionary for FAST."""
        observations = {}
        
        # Add images if present
        if 'images' in data:
            for cam_name, images in data['images'].items():
                observations[f'image_{cam_name}'] = images
        
        # Add robot state
        if 'states' in data:
            observations['robot_state'] = data['states']
        
        return observations
    
    def _prepare_metadata(self, 
                         orig_metadata: Dict[str, Any],
                         chunk_shape: tuple) -> Dict[str, Any]:
        """Prepare metadata for FAST."""
        return {
            'chunk_info': {
                'num_chunks': chunk_shape[0],
                'chunk_size': self.chunk_size,
                'overlap': self.chunk_overlap
            },
            'original_metadata': orig_metadata
        }
