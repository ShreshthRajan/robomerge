from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time

from .fast_prep import FASTBatch
from .ki_prep import KIBatch

@dataclass
class RTCBatch:
    """Real-Time Chunking compatible data batch for deployment."""
    # Action chunks ready for real-time execution
    action_chunks: np.ndarray  # Smoothed action chunks
    chunk_transitions: np.ndarray  # Transition masks for chunk switching
    
    # Inpainting masks for chunk consistency
    inpaint_masks: np.ndarray  # Which actions to freeze during transitions
    partial_attention_masks: np.ndarray  # Partial attention for overlapping actions
    
    # Timing and latency information
    chunk_timing: np.ndarray  # Expected timing for each chunk
    latency_tolerance: float  # Maximum latency this batch can handle
    
    # Original data for fallback
    original_batch: Optional[KIBatch] = None
    
    # Metadata
    metadata: Dict[str, Any] = None

class RealTimeChunkingPreprocessor:
    """Real-Time Chunking preprocessor following PI's RTC methodology."""
    
    def __init__(self,
                 chunk_size: int = 50,
                 overlap_size: int = 10,
                 max_latency_ms: float = 300.0,
                 consistency_threshold: float = 0.1,
                 partial_attention_decay: float = 0.8):
        """
        Args:
            chunk_size: Size of action chunks (50 = 1 second at 50Hz)
            overlap_size: Overlap between chunks for consistency checking
            max_latency_ms: Maximum inference latency to handle (milliseconds)
            consistency_threshold: Threshold for chunk consistency detection
            partial_attention_decay: Decay factor for partial attention
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_latency_ms = max_latency_ms
        self.consistency_threshold = consistency_threshold
        self.partial_attention_decay = partial_attention_decay
        
        # Calculate timing parameters
        self.control_freq = 50.0  # Hz
        self.timestep_ms = 1000.0 / self.control_freq  # 20ms per timestep
        self.max_latency_steps = int(max_latency_ms / self.timestep_ms)
    
    def prepare_for_rtc(self, 
                       ki_batch: KIBatch,
                       simulated_latency_ms: Optional[float] = None) -> RTCBatch:
        """Convert KI batch to RTC format for real-time deployment."""
        
        # Extract action chunks
        action_chunks = ki_batch.continuous_actions
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Generate inpainting masks for chunk transitions
        inpaint_masks = self._generate_inpainting_masks(action_chunks)
        
        # Create partial attention masks for overlapping regions
        partial_attention_masks = self._generate_partial_attention_masks(action_chunks)
        
        # Generate chunk transition indicators
        chunk_transitions = self._detect_chunk_transitions(action_chunks)
        
        # Calculate chunk timing
        chunk_timing = self._calculate_chunk_timing(num_chunks, simulated_latency_ms)
        
        # Determine latency tolerance
        latency_tolerance = self._calculate_latency_tolerance(action_chunks)
        
        # Apply RTC smoothing to reduce discontinuities
        smoothed_chunks = self._apply_rtc_smoothing(action_chunks, inpaint_masks)
        
        return RTCBatch(
            action_chunks=smoothed_chunks,
            chunk_transitions=chunk_transitions,
            inpaint_masks=inpaint_masks,
            partial_attention_masks=partial_attention_masks,
            chunk_timing=chunk_timing,
            latency_tolerance=latency_tolerance,
            original_batch=ki_batch,
            metadata={
                **ki_batch.metadata,
                'rtc_enabled': True,
                'max_latency_ms': self.max_latency_ms,
                'latency_tolerance_ms': latency_tolerance,
                'chunk_overlap': self.overlap_size,
                'smoothing_applied': True,
                'consistency_threshold': self.consistency_threshold
            }
        )
    
    def _generate_inpainting_masks(self, action_chunks: np.ndarray) -> np.ndarray:
        """Generate inpainting masks following PI's RTC methodology."""
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Create masks for actions that should be "frozen" during transitions
        inpaint_masks = np.zeros((num_chunks, chunk_size), dtype=bool)
        
        # First few actions of each chunk should be frozen (already executed)
        freeze_steps = min(self.max_latency_steps, chunk_size // 3)
        
        for i in range(num_chunks):
            # Freeze the first N actions (they're already being executed)
            inpaint_masks[i, :freeze_steps] = True
            
            # For chunks after the first, also consider overlap with previous chunk
            if i > 0 and self.overlap_size > 0:
                overlap_start = max(0, chunk_size - self.overlap_size)
                # Partially freeze overlapping region based on consistency
                prev_chunk = action_chunks[i-1]
                curr_chunk = action_chunks[i]
                
                # Check consistency in overlap region
                if overlap_start < chunk_size:
                    prev_overlap = prev_chunk[overlap_start:]
                    curr_overlap = curr_chunk[:len(prev_overlap)]
                    
                    # Calculate consistency score
                    consistency = self._calculate_chunk_consistency(prev_overlap, curr_overlap)
                    
                    # If chunks are inconsistent, freeze more of the overlap
                    if consistency < self.consistency_threshold:
                        freeze_overlap = int(len(prev_overlap) * 0.7)  # Freeze 70% of overlap
                        inpaint_masks[i, :freeze_overlap] = True
        
        return inpaint_masks
    
    def _generate_partial_attention_masks(self, action_chunks: np.ndarray) -> np.ndarray:
        """Generate partial attention masks for smooth transitions."""
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Create attention weights (1.0 = full attention, 0.0 = no attention)
        attention_masks = np.ones((num_chunks, chunk_size), dtype=np.float32)
        
        for i in range(num_chunks):
            if i > 0 and self.overlap_size > 0:
                # Apply decaying attention in overlap region
                overlap_start = max(0, chunk_size - self.overlap_size)
                overlap_region = chunk_size - overlap_start
                
                for j in range(overlap_region):
                    # Exponential decay for partial attention
                    decay_factor = self.partial_attention_decay ** (j / overlap_region)
                    attention_masks[i, overlap_start + j] *= decay_factor
        
        return attention_masks
    
    def _detect_chunk_transitions(self, action_chunks: np.ndarray) -> np.ndarray:
        """Detect where chunk transitions are likely to cause issues."""
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        transitions = np.zeros(num_chunks, dtype=bool)
        
        for i in range(1, num_chunks):
            # Check for large changes between end of previous chunk and start of current
            prev_end = action_chunks[i-1, -1]  # Last action of previous chunk
            curr_start = action_chunks[i, 0]   # First action of current chunk
            
            # Calculate discontinuity magnitude
            discontinuity = np.linalg.norm(curr_start - prev_end)
            
            # Mark as problematic transition if discontinuity is large
            if discontinuity > self.consistency_threshold:
                transitions[i] = True
        
        return transitions
    
    def _calculate_chunk_timing(self, 
                               num_chunks: int, 
                               simulated_latency_ms: Optional[float] = None) -> np.ndarray:
        """Calculate expected timing for each chunk execution."""
        
        # Base timing: each chunk takes 1 second to execute (50 timesteps at 50Hz)
        base_timing = np.ones(num_chunks) * 1000.0  # milliseconds
        
        # Add inference latency
        latency = simulated_latency_ms if simulated_latency_ms else self.max_latency_ms / 2
        inference_timing = np.full(num_chunks, latency)
        
        # Total timing per chunk
        chunk_timing = base_timing + inference_timing
        
        return chunk_timing
    
    def _calculate_latency_tolerance(self, action_chunks: np.ndarray) -> float:
        """Calculate how much latency this batch can tolerate."""
        
        # Analyze action dynamics to determine tolerance
        action_velocities = np.diff(action_chunks, axis=1)
        max_velocity = np.max(np.abs(action_velocities))
        
        # Higher velocity actions are less tolerant of latency
        if max_velocity > 0.5:  # Fast movements
            tolerance = self.max_latency_ms * 0.6
        elif max_velocity > 0.2:  # Medium movements
            tolerance = self.max_latency_ms * 0.8
        else:  # Slow movements
            tolerance = self.max_latency_ms
        
        return tolerance
    
    def _apply_rtc_smoothing(self, 
                            action_chunks: np.ndarray, 
                            inpaint_masks: np.ndarray) -> np.ndarray:
        """Apply RTC smoothing to reduce chunk transition discontinuities."""
        
        smoothed_chunks = action_chunks.copy()
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Apply smoothing at chunk boundaries
        for i in range(1, num_chunks):
            # Find the boundary region
            boundary_size = min(5, chunk_size // 4)  # Smooth over 5 timesteps
            
            # Get actions around the boundary
            prev_end = smoothed_chunks[i-1, -boundary_size:]
            curr_start = smoothed_chunks[i, :boundary_size]
            
            # Apply smoothing only where inpainting allows
            for j in range(boundary_size):
                if not inpaint_masks[i, j]:  # Only smooth non-frozen actions
                    # Linear interpolation for smoothing
                    alpha = j / boundary_size
                    smoothed_action = (1 - alpha) * prev_end[-1] + alpha * curr_start[j]
                    smoothed_chunks[i, j] = smoothed_action
        
        return smoothed_chunks
    
    def _calculate_chunk_consistency(self, 
                                   chunk1: np.ndarray, 
                                   chunk2: np.ndarray) -> float:
        """Calculate consistency score between two action chunks."""
        
        # Ensure chunks are same length
        min_len = min(len(chunk1), len(chunk2))
        chunk1_trimmed = chunk1[:min_len]
        chunk2_trimmed = chunk2[:min_len]
        
        # Calculate normalized difference
        diff = np.linalg.norm(chunk1_trimmed - chunk2_trimmed, axis=-1)
        max_diff = np.linalg.norm(chunk1_trimmed, axis=-1) + np.linalg.norm(chunk2_trimmed, axis=-1)
        
        # Avoid division by zero
        max_diff = np.where(max_diff == 0, 1.0, max_diff)
        
        # Consistency score (1.0 = perfect consistency, 0.0 = completely inconsistent)
        consistency = 1.0 - np.mean(diff / max_diff)
        
        return max(0.0, consistency)
    
    def simulate_latency_robustness(self, 
                                   rtc_batch: RTCBatch,
                                   latency_range: List[float] = None) -> Dict[str, float]:
        """Simulate RTC performance under different latency conditions."""
        
        if latency_range is None:
            latency_range = [0, 100, 200, 300, 400, 500]  # milliseconds
        
        results = {}
        
        for latency_ms in latency_range:
            # Calculate performance metrics under this latency
            latency_steps = int(latency_ms / self.timestep_ms)
            
            # Check how many chunks can handle this latency
            handling_ratio = np.mean(latency_ms <= rtc_batch.latency_tolerance)
            
            # Calculate expected smoothness degradation
            smoothness_score = self._calculate_smoothness_under_latency(
                rtc_batch, latency_steps
            )
            
            results[f"{latency_ms}ms"] = {
                'handling_ratio': handling_ratio,
                'smoothness_score': smoothness_score,
                'expected_performance': handling_ratio * smoothness_score
            }
        
        return results
    
    def _calculate_smoothness_under_latency(self, 
                                          rtc_batch: RTCBatch, 
                                          latency_steps: int) -> float:
        """Calculate expected smoothness under given latency."""
        
        # Simulate the effect of latency on chunk transitions
        action_chunks = rtc_batch.action_chunks
        inpaint_masks = rtc_batch.inpaint_masks
        
        # Calculate discontinuities that would occur with this latency
        total_discontinuity = 0.0
        num_transitions = 0
        
        for i in range(1, len(action_chunks)):
            # Simulate late arrival of new chunk
            delayed_start = min(latency_steps, self.chunk_size)
            
            if delayed_start < self.chunk_size:
                # Calculate discontinuity at delayed transition
                prev_action = action_chunks[i-1, delayed_start-1]
                new_action = action_chunks[i, delayed_start]
                
                discontinuity = np.linalg.norm(new_action - prev_action)
                total_discontinuity += discontinuity
                num_transitions += 1
        
        # Convert to smoothness score (lower discontinuity = higher smoothness)
        if num_transitions > 0:
            avg_discontinuity = total_discontinuity / num_transitions
            smoothness_score = 1.0 / (1.0 + avg_discontinuity)
        else:
            smoothness_score = 1.0
        
        return smoothness_score