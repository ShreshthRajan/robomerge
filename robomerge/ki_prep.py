from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from robomerge.fast_prep import FASTBatch, FASTPreprocessor

@dataclass
class KIBatch:
    """Knowledge Insulation compatible data batch."""
    # Discrete tokenization path (for VLM backbone)
    discrete_tokens: np.ndarray  # FAST tokens for cross-entropy loss
    token_attention_mask: np.ndarray  # Which tokens are valid
    
    # Continuous action path (for action expert)
    continuous_actions: np.ndarray  # Flow matching targets
    action_attention_mask: np.ndarray  # Which actions are valid
    
    # Shared observations
    observations: Dict[str, np.ndarray]  # Images, states, etc.
    
    # Gradient isolation markers
    gradient_mask: np.ndarray  # True = stop gradient, False = allow gradient
    
    # VLM auxiliary data
    language_tokens: Optional[np.ndarray] = None  # For language tasks
    planning_tokens: Optional[np.ndarray] = None  # High-level planning
    
    # Metadata
    metadata: Dict[str, Any] = None

class KnowledgeInsulationPreprocessor:
    """Prepares data for π₀.₅ + KI training with dual objectives."""
    
    def __init__(self,
                 chunk_size: int = 50,
                 chunk_overlap: int = 10,
                 enable_web_data_mixing: bool = True,
                 gradient_isolation_ratio: float = 0.7):
        """
        Args:
            chunk_size: Action chunk size (50 = 1 second at 50Hz)
            chunk_overlap: Overlap between chunks
            enable_web_data_mixing: Whether to include VLM web data
            gradient_isolation_ratio: Fraction of action expert gradients to stop
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_web_data_mixing = enable_web_data_mixing
        self.gradient_isolation_ratio = gradient_isolation_ratio
        
        # Initialize FAST preprocessor for discrete tokenization
        self.fast_preprocessor = FASTPreprocessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def prepare_episode_ki(self, 
                          standardized_data: Dict[str, np.ndarray],
                          language_instruction: Optional[str] = None) -> KIBatch:
        """Convert standardized episode into KI-ready format."""
        
        # Step 1: Generate FAST batch (discrete + continuous)
        fast_batch = self.fast_preprocessor.prepare_episode(standardized_data)
        
        # Step 2: Generate discrete tokens from actions
        discrete_tokens = self._generate_fast_tokens(fast_batch.action_chunks)
        
        # Step 3: Prepare continuous actions (already done by FAST)
        continuous_actions = fast_batch.action_chunks
        
        # Step 4: Create attention masks
        token_mask, action_mask = self._create_attention_masks(
            discrete_tokens, continuous_actions
        )
        
        # Step 5: Generate gradient isolation mask
        gradient_mask = self._create_gradient_mask(continuous_actions.shape)
        
        # Step 6: Process language instruction if provided
        language_tokens = self._process_language_instruction(language_instruction)
        
        # Step 7: Add planning tokens (high-level commands)
        planning_tokens = self._generate_planning_tokens(
            standardized_data, language_instruction
        )
        
        return KIBatch(
            discrete_tokens=discrete_tokens,
            token_attention_mask=token_mask,
            continuous_actions=continuous_actions,
            action_attention_mask=action_mask,
            observations=fast_batch.observations,
            gradient_mask=gradient_mask,
            language_tokens=language_tokens,
            planning_tokens=planning_tokens,
            metadata={
                **fast_batch.metadata,
                'ki_enabled': True,
                'gradient_isolation_ratio': self.gradient_isolation_ratio,
                'web_data_mixed': self.enable_web_data_mixing
            }
        )
    
    def _generate_fast_tokens(self, action_chunks: np.ndarray) -> np.ndarray:
        """Generate discrete FAST tokens from continuous actions."""
        # Simulate FAST tokenization (DCT + quantization + BPE)
        # In real implementation, this would use the actual FAST tokenizer
        
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Apply DCT-like transformation (simplified)
        # Real FAST uses DCT, quantization, then BPE encoding
        tokens_per_chunk = 32  # Typical FAST compression ratio
        
        # Flatten and normalize actions
        flattened = action_chunks.reshape(num_chunks, -1)
        
        # Simulate frequency domain transformation + quantization
        # This is a simplified version - real FAST is more sophisticated
        discrete_tokens = np.round(
            (flattened + 1) * 512  # Map [-1,1] to [0,1024]
        ).astype(np.int32)
        
        # Compress to target token count (simulate BPE)
        compressed_tokens = discrete_tokens[:, :tokens_per_chunk]
        
        return compressed_tokens
    
    def _create_attention_masks(self, 
                               tokens: np.ndarray, 
                               actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create attention masks for tokens and actions."""
        # Token mask: all tokens are valid (could add padding logic)
        token_mask = np.ones(tokens.shape, dtype=bool)
        
        # Action mask: all actions are valid (could add sequence length logic)
        action_mask = np.ones(actions.shape[:2], dtype=bool)  # (num_chunks, chunk_size)
        
        return token_mask, action_mask
    
    def _create_gradient_mask(self, action_shape: tuple) -> np.ndarray:
        """Create gradient isolation mask for knowledge insulation."""
        num_chunks, chunk_size, action_dims = action_shape
        
        # Randomly mask gradients according to isolation ratio
        mask = np.random.random((num_chunks, chunk_size)) < self.gradient_isolation_ratio
        
        return mask
    
    def _process_language_instruction(self, instruction: Optional[str]) -> Optional[np.ndarray]:
        """Convert language instruction to tokens."""
        if instruction is None:
            return None
        
        # Simplified tokenization - real implementation would use proper tokenizer
        # This would normally use the VLM's tokenizer
        words = instruction.lower().split()
        
        # Mock tokenization (map words to integers)
        vocab_size = 50000
        tokens = [hash(word) % vocab_size for word in words]
        
        return np.array(tokens, dtype=np.int32)
    
    def _generate_planning_tokens(self, 
                                 data: Dict[str, np.ndarray],
                                 instruction: Optional[str]) -> Optional[np.ndarray]:
        """Generate high-level planning tokens."""
        if not self.enable_web_data_mixing:
            return None
        
        # Generate semantic planning tokens based on task context
        # This simulates high-level robot planning commands
        
        episode_length = len(data['actions'])
        
        # Create planning tokens for different phases of the task
        planning_phases = [
            "approach_object",
            "grasp_object", 
            "manipulate_object",
            "place_object",
            "retreat"
        ]
        
        # Map planning phases to token IDs
        planning_tokens = []
        phase_duration = episode_length // len(planning_phases)
        
        for i, phase in enumerate(planning_phases):
            phase_token = hash(phase) % 10000  # Planning vocab
            # Repeat token for phase duration
            planning_tokens.extend([phase_token] * phase_duration)
        
        # Pad or trim to match episode length
        planning_tokens = planning_tokens[:episode_length]
        if len(planning_tokens) < episode_length:
            planning_tokens.extend([0] * (episode_length - len(planning_tokens)))
        
        return np.array(planning_tokens, dtype=np.int32)

class VLMDataMixer:
    """Handles mixing of robot data with web-scale VLM data."""
    
    def __init__(self, mix_ratio: float = 0.3):
        """
        Args:
            mix_ratio: Proportion of VLM data to mix in (0.3 = 30% VLM, 70% robot)
        """
        self.mix_ratio = mix_ratio
    
    def create_mixed_batch(self, 
                          ki_batches: List[KIBatch],
                          web_data_samples: Optional[List[Dict]] = None) -> List[KIBatch]:
        """Mix robot data with web-scale VLM data."""
        
        if not web_data_samples or self.mix_ratio == 0:
            return ki_batches
        
        mixed_batches = []
        num_web_samples = int(len(ki_batches) * self.mix_ratio)
        
        # Add original robot batches
        mixed_batches.extend(ki_batches)
        
        # Add web data samples converted to KI format
        for i in range(min(num_web_samples, len(web_data_samples))):
            web_sample = web_data_samples[i]
            web_batch = self._convert_web_data_to_ki(web_sample)
            mixed_batches.append(web_batch)
        
        return mixed_batches
    
    def _convert_web_data_to_ki(self, web_sample: Dict) -> KIBatch:
        """Convert web data sample to KI batch format."""
        # This would convert web VLM data (images + text) to KI format
        # For demonstration, creating a minimal web data batch
        
        # Web data doesn't have actions, so create dummy actions
        dummy_actions = np.zeros((1, 50, 7))  # 1 chunk, 50 timesteps, 7 DOF
        dummy_tokens = np.zeros((1, 32), dtype=np.int32)  # 1 chunk, 32 tokens
        
        return KIBatch(
            discrete_tokens=dummy_tokens,
            token_attention_mask=np.ones_like(dummy_tokens, dtype=bool),
            continuous_actions=dummy_actions,
            action_attention_mask=np.ones((1, 50), dtype=bool),
            observations={'web_image': web_sample.get('image', np.zeros((224, 224, 3)))},
            gradient_mask=np.ones((1, 50), dtype=bool),  # Stop all gradients for web data
            language_tokens=web_sample.get('text_tokens'),
            planning_tokens=None,
            metadata={'source': 'web_data', 'ki_enabled': True}
        )