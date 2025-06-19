# robomerge/robomerge/main.py

from typing import Dict, Any, Optional, List, Union
import numpy as np
from datetime import datetime
import json

from .fast_prep import FASTBatch, FASTPreprocessor
from .ki_prep import KIBatch, KnowledgeInsulationPreprocessor, VLMDataMixer
from .rtc_prep import RTCBatch, RealTimeChunkingPreprocessor
from .ops_dashboard import OperationsDashboard
from .ingestion import DROIDIngestion
from .transform import DataStandardizer
from .validation import DataValidator, QualityMetrics

class RoboMerge:
    """Enhanced pipeline interface for robot data processing with full PI methodology support.
    
    Features:
    1. Data ingestion from DROID format
    2. Standardization and normalization  
    3. Quality validation with real-time monitoring
    4. FAST-compatible output formatting
    5. Knowledge Insulation (KI) preprocessing for π₀.₅ + KI training
    6. Real-Time Chunking (RTC) for deployment
    7. Operations dashboard for monitoring and management
    """
    
    def __init__(self, 
                 target_freq: float = 50.0,
                 min_completeness: float = 0.99,
                 max_action_jump: float = 0.5,
                 min_temporal_consistency: float = 0.95,
                 enable_ki: bool = False,
                 enable_rtc: bool = False,
                 enable_dashboard: bool = False,
                 ki_config: Optional[Dict[str, Any]] = None,
                 rtc_config: Optional[Dict[str, Any]] = None,
                 dashboard_config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced RoboMerge pipeline.
        
        Args:
            target_freq: Target control frequency (Hz)
            min_completeness: Minimum required data completeness (0-1)
            max_action_jump: Maximum allowed action discontinuity
            min_temporal_consistency: Minimum temporal consistency (0-1)
            enable_ki: Enable Knowledge Insulation preprocessing
            enable_rtc: Enable Real-Time Chunking preprocessing  
            enable_dashboard: Enable operations dashboard monitoring
            ki_config: Configuration for KI preprocessing
            rtc_config: Configuration for RTC preprocessing
            dashboard_config: Configuration for operations dashboard
        """
        self.enable_ki = enable_ki
        self.enable_rtc = enable_rtc
        self.enable_dashboard = enable_dashboard
        self.target_freq = target_freq
        
        # Initialize core components
        self.ingestion = DROIDIngestion(target_freq)
        self.standardizer = DataStandardizer(target_freq)
        self.validator = DataValidator(
            min_completeness=min_completeness,
            max_action_jump=max_action_jump,
            min_temporal_consistency=min_temporal_consistency
        )
        
        # Initialize KI components if enabled
        if self.enable_ki:
            ki_config = ki_config or {}
            self.ki_preprocessor = KnowledgeInsulationPreprocessor(
                chunk_size=ki_config.get('chunk_size', 50),
                chunk_overlap=ki_config.get('chunk_overlap', 10),
                enable_web_data_mixing=ki_config.get('enable_web_data_mixing', True),
                gradient_isolation_ratio=ki_config.get('gradient_isolation_ratio', 0.7),
                fast_compression_ratio=ki_config.get('fast_compression_ratio', 8),
                planning_vocab_size=ki_config.get('planning_vocab_size', 10000)
            )
            self.vlm_data_mixer = VLMDataMixer(
                mix_ratio=ki_config.get('mix_ratio', 0.3),
                web_data_types=ki_config.get('web_data_types', ['vision_language', 'planning'])
            )
        
        # Initialize RTC components if enabled
        if self.enable_rtc:
            rtc_config = rtc_config or {}
            self.rtc_preprocessor = RealTimeChunkingPreprocessor(
                chunk_size=rtc_config.get('chunk_size', 50),
                overlap_size=rtc_config.get('overlap_size', 10),
                max_latency_ms=rtc_config.get('max_latency_ms', 300.0),
                consistency_threshold=rtc_config.get('consistency_threshold', 0.1),
                partial_attention_decay=rtc_config.get('partial_attention_decay', 0.8)
            )
        
        # Initialize dashboard if enabled
        if self.enable_dashboard:
            dashboard_config = dashboard_config or {}
            self.dashboard = OperationsDashboard(
                max_history_hours=dashboard_config.get('max_history_hours', 24)
            )
            self.dashboard.start_monitoring()
    
    def process_episode(self, 
                       filepath: str, 
                       operator_id: Optional[str] = None,
                       raise_on_warning: bool = False,
                       language_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Process a single episode with optional operator tracking.
        
        Args:
            filepath: Path to DROID episode file
            operator_id: ID of operator who collected this data (for dashboard)
            raise_on_warning: Whether to raise error on quality warnings
            language_instruction: Optional language instruction for KI processing
        
        Returns:
            Dict containing standardized data and metadata
        """
        start_time = datetime.now()
        
        # Load and validate raw data
        episode = self.ingestion.load_episode(filepath)
        
        # Standardize to common format
        standardized = self.standardizer.standardize_episode(episode)
        
        # Validate data quality
        quality = self.validator.validate_episode(standardized)
        
        # Add KI-specific quality metrics if enabled
        ki_quality_metrics = {}
        if self.enable_ki:
            ki_quality_metrics = self._validate_ki_compatibility(standardized, language_instruction)
            quality_dict = quality.__dict__.copy()
            quality_dict.update(ki_quality_metrics)
        else:
            quality_dict = quality.__dict__
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add comprehensive metadata
        standardized['metadata'] = {
            'original_frequency': episode.frequency,
            'target_frequency': self.standardizer.target_freq,
            'action_dims': episode.actions.shape[1],
            'episode_length': len(episode.actions),
            'normalized': True,
            'ki_enabled': self.enable_ki,
            'rtc_enabled': self.enable_rtc,
            'dashboard_enabled': self.enable_dashboard,
            'processing_time_seconds': processing_time,
            'quality_metrics': {
                'completeness': quality.completeness,
                'temporal_consistency': quality.temporal_consistency,
                'action_smoothness': quality.action_smoothness,
                **ki_quality_metrics
            },
            'quality_warnings': quality.warnings,
            'processing_info': {
                'source_format': 'DROID',
                'pipeline_version': '0.3.0',
                'ki_version': '1.0.0' if self.enable_ki else None,
                'rtc_version': '1.0.0' if self.enable_rtc else None,
                'processed_at': start_time.isoformat(),
                'operator_id': operator_id
            }
        }
        
        # Update dashboard if enabled
        if self.enable_dashboard and operator_id:
            self._update_dashboard_metrics(operator_id, standardized, quality, processing_time)
        
        if raise_on_warning and quality.warnings:
            raise ValueError(f"Quality warnings: {quality.warnings}")
        
        return standardized
    
    def process_for_fast(self, 
                        filepath: str,
                        chunk_size: Optional[int] = None,
                        raise_on_warning: bool = False) -> FASTBatch:
        """Process episode directly to FAST-ready format (backward compatible)."""
        # Process through standard pipeline
        standardized = self.process_episode(filepath, raise_on_warning=raise_on_warning)
        
        # Initialize FAST preprocessor
        fast_prep = FASTPreprocessor(
            chunk_size=chunk_size if chunk_size else int(self.target_freq)
        )
        
        # Convert to FAST format
        return fast_prep.prepare_episode(standardized)
    
    def process_for_ki(self,
                      filepath: str,
                      language_instruction: Optional[str] = None,
                      web_data_context: Optional[Dict] = None,
                      operator_id: Optional[str] = None,
                      raise_on_warning: bool = False) -> KIBatch:
        """Process episode for π₀.₅ + KI training."""
        if not self.enable_ki:
            raise ValueError("KI processing not enabled. Initialize RoboMerge with enable_ki=True")
        
        # Process through standard pipeline
        standardized = self.process_episode(
            filepath, 
            operator_id=operator_id,
            raise_on_warning=raise_on_warning,
            language_instruction=language_instruction
        )
        
        # Convert to KI format
        return self.ki_preprocessor.prepare_episode_ki(
            standardized, 
            language_instruction=language_instruction,
            web_data_context=web_data_context
        )
    
    def process_for_rtc(self,
                       filepath: str,
                       language_instruction: Optional[str] = None,
                       simulated_latency_ms: Optional[float] = None,
                       operator_id: Optional[str] = None,
                       raise_on_warning: bool = False) -> RTCBatch:
        """Process episode for real-time deployment with RTC."""
        if not self.enable_rtc:
            raise ValueError("RTC processing not enabled. Initialize RoboMerge with enable_rtc=True")
        
        # First process with KI if available, otherwise use FAST
        if self.enable_ki:
            ki_batch = self.process_for_ki(
                filepath,
                language_instruction=language_instruction,
                operator_id=operator_id,
                raise_on_warning=raise_on_warning
            )
            return self.rtc_preprocessor.prepare_for_rtc(ki_batch, simulated_latency_ms)
        else:
            # Create KI-compatible batch from FAST processing
            fast_batch = self.process_for_fast(filepath, raise_on_warning=raise_on_warning)
            
            # Convert FAST batch to KI-compatible format for RTC
            pseudo_ki_batch = self._convert_fast_to_ki_format(fast_batch)
            return self.rtc_preprocessor.prepare_for_rtc(pseudo_ki_batch, simulated_latency_ms)
    
    def create_mixed_training_batch(self,
                               robot_episodes: List[str],
                               web_data_samples: Optional[List[Dict]] = None,
                               language_instructions: Optional[List[str]] = None,
                               operator_ids: Optional[List[str]] = None) -> List[KIBatch]:
        """Create mixed training batch for KI training with dashboard tracking."""
        if not self.enable_ki:
            raise ValueError("KI processing not enabled. Initialize RoboMerge with enable_ki=True")
        
        # Process robot episodes
        ki_batches = []
        for i, episode_path in enumerate(robot_episodes):
            instruction = language_instructions[i] if language_instructions else None
            operator_id = operator_ids[i] if operator_ids else None
            
            try:
                ki_batch = self.process_for_ki(
                    episode_path,
                    language_instruction=instruction,
                    operator_id=operator_id,
                    raise_on_warning=False
                )
                ki_batches.append(ki_batch)
                
                # Add to processing queue if dashboard enabled
                if self.enable_dashboard and operator_id:
                    self.dashboard.add_to_queue(
                        task_id=f"train_{i}_{int(datetime.now().timestamp())}",
                        operator_id=operator_id,
                        task_type="training_batch",
                        priority=2
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to process episode {episode_path}: {e}")
                if self.enable_dashboard:
                    self._log_processing_failure(episode_path, str(e), operator_id)
                continue
        
        # Mix with web data if provided
        if web_data_samples:
            mixed_batches = self.vlm_data_mixer.create_mixed_batch(ki_batches, web_data_samples)
            return mixed_batches
        else:
            return ki_batches
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive pipeline configuration and status."""
        base_info = {
            'target_frequency': self.standardizer.target_freq,
            'quality_thresholds': {
                'min_completeness': self.validator.min_completeness,
                'max_action_jump': self.validator.max_action_jump,
                'min_temporal_consistency': self.validator.min_temporal_consistency
            },
            'supported_formats': ['DROID'],
            'output_formats': ['FAST-compatible'],
            'features': {
                'ki_enabled': self.enable_ki,
                'rtc_enabled': self.enable_rtc,
                'dashboard_enabled': self.enable_dashboard
            },
            'pipeline_version': '0.3.0',
            'status': 'operational'
        }
        
        # Add KI-specific information
        if self.enable_ki:
            base_info.update({
                'ki_features': {
                    'gradient_isolation': True,
                    'fast_tokenization': True,
                    'web_data_mixing': self.ki_preprocessor.enable_web_data_mixing,
                    'planning_tokens': True,
                    'language_instruction_support': True
                },
                'ki_config': {
                    'chunk_size': self.ki_preprocessor.chunk_size,
                    'chunk_overlap': self.ki_preprocessor.chunk_overlap,
                    'gradient_isolation_ratio': self.ki_preprocessor.gradient_isolation_ratio,
                    'fast_compression_ratio': self.ki_preprocessor.fast_compression_ratio,
                    'planning_vocab_size': self.ki_preprocessor.planning_vocab_size,
                    'mix_ratio': self.vlm_data_mixer.mix_ratio
                },
                'output_formats': base_info['output_formats'] + ['KI-compatible']
            })
        
        # Add RTC-specific information
        if self.enable_rtc:
            base_info.update({
                'rtc_features': {
                    'real_time_chunking': True,
                    'inpainting_masks': True,
                    'partial_attention': True,
                    'latency_robustness': True,
                    'consistency_checking': True
                },
                'rtc_config': {
                    'chunk_size': self.rtc_preprocessor.chunk_size,
                    'overlap_size': self.rtc_preprocessor.overlap_size,
                    'max_latency_ms': self.rtc_preprocessor.max_latency_ms,
                    'consistency_threshold': self.rtc_preprocessor.consistency_threshold,
                    'partial_attention_decay': self.rtc_preprocessor.partial_attention_decay
                },
                'output_formats': base_info['output_formats'] + ['RTC-compatible']
            })
        
        # Add dashboard information
        if self.enable_dashboard:
            dashboard_summary = self.dashboard.get_dashboard_summary()
            base_info.update({
                'dashboard_status': {
                    'monitoring_active': self.dashboard.is_monitoring,
                    'operators_tracked': dashboard_summary['overview']['total_operators'],
                    'active_operators': dashboard_summary['overview']['active_operators'],
                    'processing_queue_size': dashboard_summary['queue']['total_items'],
                    'active_alerts': dashboard_summary['alerts']['total_active']
                }
            })
        
        return base_info
    
    def get_dashboard_summary(self) -> Optional[Dict[str, Any]]:
        """Get operations dashboard summary."""
        if not self.enable_dashboard:
            return None
        return self.dashboard.get_dashboard_summary()
    
    def test_rtc_robustness(self, 
                            rtc_batch: RTCBatch,
                            latency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """Test RTC robustness under different latency conditions."""
        if not self.enable_rtc:
            raise ValueError("RTC not enabled")
        
        return self.rtc_preprocessor.simulate_latency_robustness(rtc_batch, latency_range)
    
    def shutdown(self):
        """Gracefully shutdown the pipeline."""
        if self.enable_dashboard:
            self.dashboard.stop_monitoring()
    
    # Private helper methods
    def _validate_ki_compatibility(self, 
                                    standardized_data: Dict[str, np.ndarray],
                                    language_instruction: Optional[str] = None) -> Dict[str, float]:
        """Validate data compatibility with KI training requirements."""
        metrics = {}
        
        # Check action sequence length compatibility
        action_length = len(standardized_data['actions'])
        chunk_size = self.ki_preprocessor.chunk_size
        num_complete_chunks = action_length // chunk_size
        chunk_coverage = (num_complete_chunks * chunk_size) / action_length
        metrics['chunk_coverage'] = chunk_coverage
        
        # Check language instruction quality
        if language_instruction:
            instruction_words = len(language_instruction.split())
            metrics['instruction_length'] = min(instruction_words / 20.0, 1.0)
            
            robotics_keywords = ['pick', 'place', 'grasp', 'move', 'push', 'pull', 'rotate', 'turn', 'open', 'close']
            has_robotics_vocab = any(word in language_instruction.lower() for word in robotics_keywords)
            metrics['instruction_robotics_relevance'] = 1.0 if has_robotics_vocab else 0.5
        else:
            metrics['instruction_length'] = 0.0
            metrics['instruction_robotics_relevance'] = 0.0
        
        # Check action diversity for better tokenization
        actions = standardized_data['actions']
        action_std = np.std(actions, axis=0).mean()
        action_range = np.ptp(actions, axis=0).mean()
        metrics['action_diversity'] = min(action_range / 2.0, 1.0)
        
        # Check temporal consistency for gradient masking
        if 'timestamps' in standardized_data:
            timestamps = standardized_data['timestamps']
            time_diffs = np.diff(timestamps)
            temporal_regularity = 1.0 - (np.std(time_diffs) / np.mean(time_diffs))
            metrics['temporal_regularity'] = max(0.0, temporal_regularity)
        else:
            metrics['temporal_regularity'] = 1.0
        
        return metrics
    
    def _update_dashboard_metrics(self, 
                                    operator_id: str,
                                    standardized_data: Dict[str, Any],
                                    quality: QualityMetrics,
                                    processing_time: float):
        """Update dashboard with processing results."""
        result = {
            'success': len(quality.warnings) == 0,
            'quality_score': quality.completeness * quality.temporal_consistency * (1 - quality.action_smoothness/2),
            'completion_time': processing_time,
            'episode_length': standardized_data['metadata']['episode_length'],
            'action_dims': standardized_data['metadata']['action_dims']
        }
        
        self.dashboard.update_operator_metrics(operator_id, result)
    
    def _convert_fast_to_ki_format(self, fast_batch: FASTBatch) -> KIBatch:
        """Convert FAST batch to KI-compatible format for RTC processing."""
        # Create minimal KI batch structure
        return KIBatch(
            discrete_tokens=np.zeros((fast_batch.action_chunks.shape[0], 32), dtype=np.int32),
            token_attention_mask=np.ones((fast_batch.action_chunks.shape[0], 32), dtype=bool),
            continuous_actions=fast_batch.action_chunks,
            action_attention_mask=np.ones(fast_batch.action_chunks.shape[:2], dtype=bool),
            observations=fast_batch.observations,
            gradient_mask=np.ones(fast_batch.action_chunks.shape[:2], dtype=bool),
            language_tokens=None,
            planning_tokens=None,
            metadata={**fast_batch.metadata, 'converted_from_fast': True}
        )
    
    def _log_processing_failure(self, filepath: str, error: str, operator_id: Optional[str]):
        """Log processing failure to dashboard."""
        if operator_id:
            result = {
                'success': False,
                'quality_score': 0.0,
                'completion_time': 0.0,
                'error': error,
                'filepath': filepath
            }
            self.dashboard.update_operator_metrics(operator_id, result)