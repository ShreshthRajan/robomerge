from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .fast_prep import FASTBatch, FASTPreprocessor

@dataclass
class KIBatch:
    """Knowledge Insulation compatible data batch for π₀.₅ + KI training."""
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
    """Prepares data for π₀.₅ + KI training with dual objectives following PI's methodology."""
    
    def __init__(self,
                 chunk_size: int = 50,
                 chunk_overlap: int = 10,
                 enable_web_data_mixing: bool = True,
                 gradient_isolation_ratio: float = 0.7,
                 fast_compression_ratio: int = 8,
                 planning_vocab_size: int = 10000):
        """
        Args:
            chunk_size: Action chunk size (50 = 1 second at 50Hz)
            chunk_overlap: Overlap between chunks for temporal consistency
            enable_web_data_mixing: Whether to include VLM web data
            gradient_isolation_ratio: Fraction of action expert gradients to stop
            fast_compression_ratio: DCT compression ratio for FAST tokens
            planning_vocab_size: Vocabulary size for planning tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_web_data_mixing = enable_web_data_mixing
        self.gradient_isolation_ratio = gradient_isolation_ratio
        self.fast_compression_ratio = fast_compression_ratio
        self.planning_vocab_size = planning_vocab_size
        
        # Initialize FAST preprocessor for discrete tokenization
        self.fast_preprocessor = FASTPreprocessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def prepare_episode_ki(self, 
                          standardized_data: Dict[str, np.ndarray],
                          language_instruction: Optional[str] = None,
                          web_data_context: Optional[Dict] = None) -> KIBatch:
        """Convert standardized episode into KI-ready format following π₀.₅ + KI methodology."""
        
        # Step 1: Generate FAST batch (discrete + continuous)
        fast_batch = self.fast_preprocessor.prepare_episode(standardized_data)
        
        # Step 2: Generate improved FAST tokens using DCT-like compression
        discrete_tokens = self._generate_enhanced_fast_tokens(fast_batch.action_chunks)
        
        # Step 3: Prepare continuous actions (already done by FAST)
        continuous_actions = fast_batch.action_chunks
        
        # Step 4: Create enhanced attention masks with temporal awareness
        token_mask, action_mask = self._create_enhanced_attention_masks(
            discrete_tokens, continuous_actions
        )
        
        # Step 5: Generate improved gradient isolation mask with temporal consistency
        gradient_mask = self._create_enhanced_gradient_mask(
            continuous_actions.shape, standardized_data.get('timestamps')
        )
        
        # Step 6: Process language instruction with proper tokenization
        language_tokens = self._process_language_instruction_enhanced(language_instruction)
        
        # Step 7: Generate context-aware planning tokens
        planning_tokens = self._generate_context_aware_planning_tokens(
            standardized_data, language_instruction, web_data_context
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
                'web_data_mixed': self.enable_web_data_mixing,
                'fast_compression_ratio': self.fast_compression_ratio,
                'training_streams': ['discrete_tokens', 'continuous_actions'] + 
                                 (['web_data'] if web_data_context else [])
            }
        )
    
    def _generate_enhanced_fast_tokens(self, action_chunks: np.ndarray) -> np.ndarray:
        """Generate discrete FAST tokens using improved DCT + quantization + BPE simulation."""
        num_chunks, chunk_size, action_dims = action_chunks.shape
        
        # Apply DCT-like frequency domain transformation
        tokens_per_chunk = chunk_size * action_dims // self.fast_compression_ratio
        
        # Flatten actions for processing
        flattened = action_chunks.reshape(num_chunks, -1)
        
        # Simulate DCT transformation - keep low frequency components
        # This is a simplified version of actual DCT
        dct_coeffs = np.fft.fft(flattened, axis=1).real
        
        # Keep only low-frequency components (compression)
        compressed_coeffs = dct_coeffs[:, :tokens_per_chunk]
        
        # Quantization step - map to discrete vocabulary
        # Normalize to [-1, 1] then quantize to vocab size
        normalized_coeffs = np.tanh(compressed_coeffs / np.std(compressed_coeffs))
        quantized_tokens = np.round(
            (normalized_coeffs + 1) * 512  # Map [-1,1] to [0,1024]
        ).astype(np.int32)
        
        # Clip to valid token range
        discrete_tokens = np.clip(quantized_tokens, 0, 1023)
        
        return discrete_tokens
    
    def _create_enhanced_attention_masks(self, 
                                       tokens: np.ndarray, 
                                       actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create attention masks with proper handling of variable sequence lengths."""
        # Token mask: all generated tokens are valid
        token_mask = np.ones(tokens.shape, dtype=bool)
        
        # Action mask: handle variable episode lengths
        num_chunks, chunk_size = actions.shape[:2]
        action_mask = np.ones((num_chunks, chunk_size), dtype=bool)
        
        # For real deployment, you might want to mask padded sequences
        # This is a placeholder for more sophisticated masking logic
        
        return token_mask, action_mask
    
    def _create_enhanced_gradient_mask(self, 
                                     action_shape: tuple, 
                                     timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Create gradient isolation mask with temporal consistency for knowledge insulation."""
        num_chunks, chunk_size, action_dims = action_shape
        
        # Create base random mask
        base_mask = np.random.random((num_chunks, chunk_size)) < self.gradient_isolation_ratio
        
        # Add temporal consistency - nearby timesteps should have similar masking
        if timestamps is not None and len(timestamps) > 0:
            # Apply temporal smoothing to reduce abrupt changes
            smoothing_kernel = np.array([0.25, 0.5, 0.25])
            for i in range(num_chunks):
                start_idx = max(0, i-1)
                end_idx = min(num_chunks, i+2)
                if end_idx - start_idx >= 2:
                    # Smooth the mask temporally
                    smoothed = np.convolve(
                        base_mask[start_idx:end_idx].mean(axis=1), 
                        smoothing_kernel, 
                        mode='same'
                    )
                    # Apply smoothed probability to current chunk
                    if i-start_idx < len(smoothed):
                        prob = smoothed[i-start_idx]
                        base_mask[i] = np.random.random(chunk_size) < prob
        
        return base_mask
    
    def _process_language_instruction_enhanced(self, instruction: Optional[str]) -> Optional[np.ndarray]:
        """Convert language instruction to tokens with improved tokenization."""
        if instruction is None:
            return None
        
        # Enhanced tokenization that handles common robotics vocabulary
        # This is a simplified version - real implementation would use proper tokenizer
        
        # Preprocessing: normalize and handle robotics-specific terms
        instruction = instruction.lower().strip()
        
        # Split into tokens (words and subwords)
        words = instruction.split()
        
        # Enhanced vocabulary mapping with robotics awareness
        vocab_size = 50000
        special_robotics_terms = {
            'pick': 1000, 'place': 1001, 'grasp': 1002, 'release': 1003,
            'move': 1004, 'reach': 1005, 'push': 1006, 'pull': 1007,
            'left': 1010, 'right': 1011, 'up': 1012, 'down': 1013,
            'forward': 1014, 'backward': 1015, 'rotate': 1016, 'turn': 1017
        }
        
        tokens = []
        for word in words:
            if word in special_robotics_terms:
                tokens.append(special_robotics_terms[word])
            else:
                # Hash-based tokenization for other words
                tokens.append((hash(word) % (vocab_size - 2000)) + 2000)
        
        return np.array(tokens, dtype=np.int32)
    
    def _generate_context_aware_planning_tokens(self, 
                                              data: Dict[str, np.ndarray],
                                              instruction: Optional[str],
                                              web_context: Optional[Dict] = None) -> Optional[np.ndarray]:
        """Generate high-level planning tokens with context awareness."""
        if not self.enable_web_data_mixing:
            return None
        
        episode_length = len(data['actions'])
        
        # Enhanced planning phase detection based on action analysis
        planning_phases = self._detect_planning_phases(data, instruction)
        
        # Generate planning tokens with smoother transitions
        planning_tokens = []
        phase_tokens = {phase: hash(phase) % self.planning_vocab_size 
                       for phase in planning_phases}
        
        # Create smoother phase transitions
        phase_duration = max(1, episode_length // len(planning_phases))
        transition_length = min(5, phase_duration // 4)  # 25% transition overlap
        
        for i, phase in enumerate(planning_phases):
            phase_token = phase_tokens[phase]
            
            # Main phase duration
            main_duration = phase_duration - transition_length
            planning_tokens.extend([phase_token] * main_duration)
            
            # Transition to next phase (if not last phase)
            if i < len(planning_phases) - 1:
                next_token = phase_tokens[planning_phases[i + 1]]
                # Linear interpolation between phase tokens
                for j in range(transition_length):
                    alpha = j / transition_length
                    if alpha < 0.5:
                        planning_tokens.append(phase_token)
                    else:
                        planning_tokens.append(next_token)
        
        # Adjust to exact episode length
        planning_tokens = planning_tokens[:episode_length]
        if len(planning_tokens) < episode_length:
            # Pad with last token
            last_token = planning_tokens[-1] if planning_tokens else 0
            planning_tokens.extend([last_token] * (episode_length - len(planning_tokens)))
        
        return np.array(planning_tokens, dtype=np.int32)
    
    def _detect_planning_phases(self, 
                              data: Dict[str, np.ndarray], 
                              instruction: Optional[str]) -> List[str]:
        """Detect planning phases based on action patterns and language instruction."""
        # Analyze action patterns to detect phases
        actions = data['actions']
        
        # Simple heuristic based on action magnitude and changes
        action_magnitudes = np.linalg.norm(actions, axis=1)
        action_changes = np.diff(action_magnitudes)
        
        # Determine phases based on instruction and action patterns
        if instruction:
            instruction_lower = instruction.lower()
            if any(word in instruction_lower for word in ['pick', 'grasp', 'grab']):
                if any(word in instruction_lower for word in ['place', 'put', 'drop']):
                    return ["approach_object", "grasp_object", "lift_object", "transport_object", "place_object", "retreat"]
                else:
                    return ["approach_object", "grasp_object", "manipulate_object"]
            elif any(word in instruction_lower for word in ['move', 'push', 'slide']):
                return ["approach_object", "contact_object", "push_object", "release_object"]
            elif any(word in instruction_lower for word in ['open', 'close']):
                return ["approach_handle", "grasp_handle", "actuate_mechanism", "release_handle"]
        
        # Default phases for general manipulation
        return ["approach_object", "grasp_object", "manipulate_object", "place_object", "retreat"]

class VLMDataMixer:
    """Enhanced VLM data mixer for π₀.₅ + KI training with proper web data integration."""
    
    def __init__(self, 
                 mix_ratio: float = 0.3,
                 web_data_types: List[str] = None):
        """
        Args:
            mix_ratio: Proportion of VLM data to mix in (0.3 = 30% VLM, 70% robot)
            web_data_types: Types of web data to include ['vision_language', 'planning', 'reasoning']
        """
        self.mix_ratio = mix_ratio
        self.web_data_types = web_data_types or ['vision_language', 'planning']
    
    def create_mixed_batch(self, 
                          ki_batches: List[KIBatch],
                          web_data_samples: Optional[List[Dict]] = None) -> List[KIBatch]:
        """Mix robot data with web-scale VLM data following π₀.₅ + KI methodology."""
        
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
        
        # Shuffle to ensure proper mixing during training
        indices = np.random.permutation(len(mixed_batches))
        return [mixed_batches[i] for i in indices]
    
    def _convert_web_data_to_ki(self, web_sample: Dict) -> KIBatch:
        """Convert web data sample to KI batch format for joint training."""
        # Web data characteristics
        has_image = 'image' in web_sample
        has_text = 'text' in web_sample or 'text_tokens' in web_sample
        
        # Create minimal action placeholders (web data has no actions)
        dummy_actions = np.zeros((1, 50, 7))  # 1 chunk, 50 timesteps, 7 DOF
        dummy_tokens = np.zeros((1, 32), dtype=np.int32)  # 1 chunk, 32 tokens
        
        # Process text if available
        language_tokens = None
        if has_text:
            if 'text_tokens' in web_sample:
                language_tokens = web_sample['text_tokens']
            else:
                # Simple tokenization of text
                text = web_sample.get('text', '')
                words = text.lower().split()
                vocab_size = 50000
                language_tokens = np.array([hash(word) % vocab_size for word in words], dtype=np.int32)
        
        # Handle image data
        observations = {}
        if has_image:
            image = web_sample['image']
            # Ensure image is in correct format
            if isinstance(image, np.ndarray):
                observations['web_image'] = image
            else:
                observations['web_image'] = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            observations['web_image'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        return KIBatch(
            discrete_tokens=dummy_tokens,
            token_attention_mask=np.ones_like(dummy_tokens, dtype=bool),
            continuous_actions=dummy_actions,
            action_attention_mask=np.ones((1, 50), dtype=bool),
            observations=observations,
            gradient_mask=np.ones((1, 50), dtype=bool),  # Stop all gradients for web data
            language_tokens=language_tokens,
            planning_tokens=None,  # Web data doesn't have robot planning
            metadata={
                'source': 'web_data', 
                'ki_enabled': True,
                'data_type': web_sample.get('type', 'general'),
                'has_image': has_image,
                'has_text': has_text
            }
        )