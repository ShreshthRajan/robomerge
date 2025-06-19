# robomerge/robomerge/fast_prep.py
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class FASTBatch:
    """FAST-compatible data batch with KI support."""
    action_chunks: np.ndarray  # Shape: (N, chunk_size, action_dims)
    observations: Dict[str, np.ndarray]  # Camera images, state, etc.
    metadata: Dict[str, Any]
    
    # New KI-compatible fields
    discrete_tokens: Optional[np.ndarray] = None  # FAST discrete tokens
    token_metadata: Optional[Dict[str, Any]] = None  # Token generation info

class FASTPreprocessor:
    """Prepares standardized data for FAST tokenization with KI support."""
    
    def __init__(self, 
                 chunk_size: int = 50,  # 1 second at 50Hz
                 chunk_overlap: int = 10,
                 enable_discrete_tokens: bool = False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_discrete_tokens = enable_discrete_tokens
    
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
            standardized_data.get('metadata', {}),
            action_chunks.shape
        )
        
        # Generate discrete tokens if requested (for KI compatibility)
        discrete_tokens = None
        token_metadata = None
        
        if self.enable_discrete_tokens:
            discrete_tokens, token_metadata = self._generate_discrete_tokens(action_chunks)
        
        return FASTBatch(
            action_chunks=action_chunks,
            observations=observations,
            metadata=metadata,
            discrete_tokens=discrete_tokens,
            token_metadata=token_metadata
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
            'original_metadata': orig_metadata,
            'fast_compatible': True,
            'ki_ready': self.enable_discrete_tokens
        }
    
    def _generate_discrete_tokens(self, 
                                 action_chunks: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Generate FAST discrete tokens from action chunks."""
        # This implements the FAST tokenization process:
        # 1. DCT transformation
        # 2. Quantization  
        # 3. BPE encoding
        
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Step 1: Flatten chunks for DCT
        flattened_chunks = action_chunks.reshape(num_chunks, -1)
        
        # Step 2: Apply DCT-like frequency transformation
        # (Simplified - real FAST uses proper DCT)
        frequency_coeffs = np.fft.dct(flattened_chunks, axis=1, norm='ortho')
        
        # Step 3: Quantization (compress to fewer coefficients)
        # Keep only low-frequency components (FAST compression strategy)
        num_coeffs = min(64, flattened_chunks.shape[1] // 4)
        quantized_coeffs = frequency_coeffs[:, :num_coeffs]
        
        # Step 4: Convert to discrete tokens
        # Map to vocabulary range (simulating BPE)
        vocab_size = 8192  # Typical FAST vocab size
        
        # Normalize and quantize
        normalized = (quantized_coeffs - quantized_coeffs.min()) / (
            quantized_coeffs.max() - quantized_coeffs.min() + 1e-8
        )
        discrete_tokens = (normalized * (vocab_size - 1)).astype(np.int32)
        
        # Metadata about tokenization
        token_metadata = {
            'vocab_size': vocab_size,
            'num_coeffs': num_coeffs,
            'compression_ratio': (chunk_size * action_dims) / num_coeffs,
            'tokenization_method': 'dct_quantization_bpe'
        }
        
        return discrete_tokens, token_metadata