#!/usr/bin/env python3
"""
Comprehensive test suite for RoboMerge Phase 3
Tests all KI, RTC, and Operations Dashboard functionality
"""

import os
import sys
import time
import numpy as np
import h5py
from pathlib import Path
import tempfile
import json

# Add robomerge to path
sys.path.append(str(Path(__file__).parent.parent))

def create_test_data(output_path: str):
    """Create synthetic test data for validation."""
    print("Creating synthetic test data...")
    
    # Generate realistic robot trajectory
    timesteps = 400  # 8 seconds at 50Hz
    actions = np.zeros((timesteps, 7))
    states = np.zeros((timesteps, 7))
    timestamps = np.arange(timesteps) / 50.0
    
    # Generate smooth trajectories
    for joint in range(7):
        # Create smooth sinusoidal motion with noise
        base_freq = 0.5 + joint * 0.1
        actions[:, joint] = 0.3 * np.sin(timestamps * base_freq) + 0.1 * np.random.normal(0, 0.1, timesteps)
        states[:, joint] = np.cumsum(actions[:, joint]) * 0.02
    
    # Create dummy images
    wrist_img = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    ext_img = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    
    # Save as HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('actions', data=actions)
        f.create_dataset('states', data=states)
        f.create_dataset('timestamps', data=timestamps)
        images_group = f.create_group('images')
        images_group.create_dataset('wrist', data=wrist_img)
        images_group.create_dataset('external', data=ext_img)
    
    print(f"✅ Test data created: {output_path}")
    return output_path

def test_basic_pipeline():
    """Test basic FAST processing pipeline."""
    print("\n=== Testing Basic FAST Pipeline ===")
    
    from robomerge.ingestion import DROIDIngestion
    from robomerge.transform import DataStandardizer
    from robomerge.validation import DataValidator
    from robomerge.fast_prep import FASTPreprocessor
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        test_file = create_test_data(tmp.name)
    
    try:
        # Test ingestion
        ingestion = DROIDIngestion()
        episode = ingestion.load_episode(test_file)
        assert episode.actions.shape[1] == 7, "Action dimensions incorrect"
        assert episode.frequency > 0, "Frequency not calculated"
        print("✅ Data ingestion passed")
        
        # Test standardization
        standardizer = DataStandardizer()
        standardized = standardizer.standardize_episode(episode)
        assert 'actions' in standardized, "Actions missing from standardized data"
        assert np.all(standardized['actions'] >= -1) and np.all(standardized['actions'] <= 1), "Actions not normalized"
        print("✅ Data standardization passed")
        
        # Test validation
        validator = DataValidator()
        quality = validator.validate_episode(standardized)
        assert quality.completeness > 0.9, f"Poor data completeness: {quality.completeness}"
        assert quality.temporal_consistency > 0.9, f"Poor temporal consistency: {quality.temporal_consistency}"
        print("✅ Data validation passed")
        
        # Test FAST preprocessing
        preprocessor = FASTPreprocessor()
        fast_data = preprocessor.prepare_episode(standardized)
        assert fast_data.action_chunks.shape[1] == 50, "Chunk size incorrect"
        assert fast_data.action_chunks.shape[2] == 7, "Action dimensions incorrect"
        print("✅ FAST preprocessing passed")
        
        return True
        
    finally:
        os.unlink(test_file)

def test_ki_integration():
    """Test Knowledge Insulation functionality."""
    print("\n=== Testing KI Integration ===")
    
    from robomerge import RoboMerge
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        test_file = create_test_data(tmp.name)
    
    try:
        # Initialize KI pipeline
        ki_config = {
            'chunk_size': 50,
            'enable_web_data_mixing': True,
            'gradient_isolation_ratio': 0.7
        }
        
        pipeline = RoboMerge(enable_ki=True, ki_config=ki_config)
        
        # Test KI processing
        ki_batch = pipeline.process_for_ki(
            test_file,
            language_instruction="pick up the red object and place it carefully"
        )
        
        # Validate KI batch structure
        assert ki_batch.discrete_tokens is not None, "Discrete tokens not generated"
        assert ki_batch.continuous_actions is not None, "Continuous actions not generated"
        assert ki_batch.gradient_mask is not None, "Gradient mask not generated"
        assert ki_batch.language_tokens is not None, "Language tokens not generated"
        
        # Check shapes are consistent
        num_chunks = ki_batch.continuous_actions.shape[0]
        assert ki_batch.discrete_tokens.shape[0] == num_chunks, "Token/action shape mismatch"
        assert ki_batch.gradient_mask.shape[0] == num_chunks, "Gradient mask shape mismatch"
        
        # Validate gradient isolation ratio
        isolation_ratio = ki_batch.gradient_mask.mean()
        expected_ratio = ki_config['gradient_isolation_ratio']
        assert abs(isolation_ratio - expected_ratio) < 0.1, f"Gradient isolation ratio mismatch: {isolation_ratio} vs {expected_ratio}"
        
        print("✅ KI integration passed")
        return True
        
    finally:
        os.unlink(test_file)

def test_rtc_functionality():
    """Test Real-Time Chunking functionality."""
    print("\n=== Testing RTC Functionality ===")
    
    from robomerge import RoboMerge
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        test_file = create_test_data(tmp.name)
    
    try:
        # Initialize RTC pipeline
        rtc_config = {
            'max_latency_ms': 300.0,
            'consistency_threshold': 0.1
        }
        
        pipeline = RoboMerge(enable_ki=True, enable_rtc=True, rtc_config=rtc_config)
        
        # Test RTC processing
        rtc_batch = pipeline.process_for_rtc(
            test_file,
            language_instruction="pick up object",
            simulated_latency_ms=200.0
        )
        
        # Validate RTC batch structure
        assert rtc_batch.action_chunks is not None, "Action chunks not generated"
        assert rtc_batch.inpaint_masks is not None, "Inpainting masks not generated"
        assert rtc_batch.partial_attention_masks is not None, "Partial attention masks not generated"
        assert rtc_batch.latency_tolerance > 0, "Latency tolerance not calculated"
        
        # Test latency robustness
        latency_results = pipeline.test_rtc_robustness(rtc_batch, [0, 100, 200, 300])
        assert len(latency_results) == 4, "Latency robustness test failed"
        
        # Validate performance degrades gracefully with latency
        perf_0ms = latency_results['0ms']['expected_performance']
        perf_300ms = latency_results['300ms']['expected_performance']
        assert perf_0ms >= perf_300ms, "Performance should degrade with latency"
        
        print("✅ RTC functionality passed")
        return True
        
    finally:
        os.unlink(test_file)

def test_operations_dashboard():
    """Test Operations Dashboard functionality."""
    print("\n=== Testing Operations Dashboard ===")
    
    from robomerge import RoboMerge
    from robomerge.ops_dashboard import OperationsDashboard
    
    # Initialize dashboard
    dashboard = OperationsDashboard()
    dashboard.start_monitoring()
    
    try:
        # Add test operators
        operators = [("OP_001", "Test Operator 1"), ("OP_002", "Test Operator 2")]
        for op_id, name in operators:
            dashboard.add_operator(op_id, name)
        
        # Simulate operator activity
        for i in range(10):
            result = {
                'success': i % 3 != 0,  # 67% success rate
                'quality_score': 0.8 + 0.1 * np.random.random(),
                'completion_time': 300 + 100 * np.random.random()
            }
            dashboard.update_operator_metrics("OP_001", result)
        
        # Add queue items
        dashboard.add_to_queue("task_001", "OP_001", "pick_place", priority=2)
        dashboard.add_to_queue("task_002", "OP_002", "folding", priority=1)
        
        # Test dashboard queries
        summary = dashboard.get_dashboard_summary()
        assert summary['overview']['total_operators'] == 2, "Operator count incorrect"
        assert summary['queue']['total_items'] >= 2, "Queue items not added"
        
        leaderboard = dashboard.get_operator_leaderboard()
        assert len(leaderboard) == 2, "Leaderboard incorrect"
        assert leaderboard[0]['rank'] == 1, "Ranking incorrect"
        
        queue_status = dashboard.get_queue_status()
        assert queue_status['total_items'] >= 2, "Queue status incorrect"
        
        print("✅ Operations dashboard passed")
        return True
        
    finally:
        dashboard.stop_monitoring()

def test_end_to_end_integration():
    """Test complete end-to-end pipeline integration."""
    print("\n=== Testing End-to-End Integration ===")
    
    from robomerge import RoboMerge
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        test_file = create_test_data(tmp.name)
    
    try:
        # Initialize full pipeline
        pipeline = RoboMerge(
            enable_ki=True,
            enable_rtc=True,
            enable_dashboard=True
        )
        
        # Test mixed training batch creation
        episodes = [test_file] * 3
        instructions = ["pick object", "place object", "clean table"]
        operators = ["OP_001", "OP_002", "OP_003"]
        
        # Create mixed training batch
        training_batches = pipeline.create_mixed_training_batch(
            episodes, 
            language_instructions=instructions,
            operator_ids=operators
        )
        
        assert len(training_batches) == 3, "Training batch creation failed"
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        assert info['features']['ki_enabled'], "KI not enabled in pipeline info"
        assert info['features']['rtc_enabled'], "RTC not enabled in pipeline info"
        assert info['features']['dashboard_enabled'], "Dashboard not enabled in pipeline info"
        
        # Test graceful shutdown
        pipeline.shutdown()
        
        print("✅ End-to-end integration passed")
        return True
        
    finally:
        os.unlink(test_file)

def test_performance_benchmarks():
    """Test performance benchmarks across all methods."""
    print("\n=== Testing Performance Benchmarks ===")
    
    from robomerge import RoboMerge
    import time
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        test_file = create_test_data(tmp.name)
    
    try:
        pipeline = RoboMerge(enable_ki=True, enable_rtc=True)
        
        # Benchmark FAST processing
        start_time = time.time()
        fast_batch = pipeline.process_for_fast(test_file)
        fast_time = time.time() - start_time
        
        # Benchmark KI processing
        start_time = time.time()
        ki_batch = pipeline.process_for_ki(test_file, language_instruction="test")
        ki_time = time.time() - start_time
        
        # Benchmark RTC processing
        start_time = time.time()
        rtc_batch = pipeline.process_for_rtc(test_file, language_instruction="test")
        rtc_time = time.time() - start_time
        
        # Performance should be reasonable (under 10 seconds for test data)
        assert fast_time < 10.0, f"FAST processing too slow: {fast_time}s"
        assert ki_time < 10.0, f"KI processing too slow: {ki_time}s"
        assert rtc_time < 10.0, f"RTC processing too slow: {rtc_time}s"
        
        print(f"⚡ Performance benchmarks:")
        print(f"   FAST: {fast_time:.3f}s")
        print(f"   KI:   {ki_time:.3f}s")
        print(f"   RTC:  {rtc_time:.3f}s")
        
        print("✅ Performance benchmarks passed")
        return True
        
    finally:
        os.unlink(test_file)

def run_all_tests():
    """Run comprehensive test suite."""
    print("🧪 ROBOMERGE COMPREHENSIVE TEST SUITE")
    print("="*50)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("KI Integration", test_ki_integration),
        ("RTC Functionality", test_rtc_functionality),
        ("Operations Dashboard", test_operations_dashboard),
        ("End-to-End Integration", test_end_to_end_integration),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - FAILED: {e}")
    
    print("\n" + "="*50)
    print(f"🎯 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)