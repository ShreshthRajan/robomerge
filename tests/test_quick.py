#!/usr/bin/env python3
"""
Quick test script to verify RoboMerge installation and functionality.

Run this to verify everything is working:
    python tests/test_quick.py
"""

import sys
import os
from pathlib import Path

# Add robomerge to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import robomerge
        print("  ✅ Core RoboMerge imported")
        
        from robomerge import RoboMerge
        print("  ✅ Main RoboMerge class imported")
        
        from robomerge import DROIDIngestion, DataStandardizer, DataValidator, FASTPreprocessor
        print("  ✅ Core components imported")
        
        # Test optional components
        try:
            from robomerge import KnowledgeInsulationPreprocessor, VLMDataMixer
            print("  ✅ KI components imported")
        except ImportError:
            print("  ⚠️  KI components not available")
        
        try:
            from robomerge import RealTimeChunkingPreprocessor
            print("  ✅ RTC components imported")
        except ImportError:
            print("  ⚠️  RTC components not available")
        
        try:
            from robomerge import OperationsDashboard
            print("  ✅ Dashboard components imported")
        except ImportError:
            print("  ⚠️  Dashboard components not available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic RoboMerge functionality."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from robomerge import RoboMerge
        
        # Test initialization
        pipeline = RoboMerge()
        print("  ✅ Basic pipeline initialization")
        
        # Test with features
        pipeline_full = RoboMerge(
            enable_ki=True,
            enable_rtc=True,
            enable_dashboard=True
        )
        print("  ✅ Full pipeline initialization")
        
        # Test info
        info = pipeline_full.get_pipeline_info()
        print(f"  ✅ Pipeline info: v{info['pipeline_version']}")
        
        # Cleanup
        pipeline.shutdown()
        pipeline_full.shutdown()
        print("  ✅ Pipeline cleanup")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing dependencies...")
    
    required = ['numpy', 'h5py', 'matplotlib']
    optional = ['tensorflow']
    
    missing = []
    
    for dep in required:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (required)")
            missing.append(dep)
    
    for dep in optional:
        try:
            __import__(dep)
            print(f"  ✅ {dep} (optional)")
        except ImportError:
            print(f"  ⚠️  {dep} (optional - some features may not work)")
    
    return len(missing) == 0

def main():
    """Run all quick tests."""
    print("🚀 RoboMerge Quick Test Suite")
    print("="*40)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "="*40)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ RoboMerge is ready to use")
        print("\nNext steps:")
        print("  • Run: python demo_live_dashboard.py")
        print("  • Or check: demo.ipynb")
        return True
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Some issues detected")
        print("\nTo fix:")
        print("  • pip install -r requirements.txt")
        print("  • Check Python version (>=3.10)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)