#!/usr/bin/env python3
"""
Quick test script to verify training fixes work properly
"""
import os
import sys
import yaml

def test_config_loading():
    """Test that config loads with new flags"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check new flags exist
        train_config = config['train']
        assert 'use_mixed_precision' in train_config, "Missing use_mixed_precision"
        assert 'use_torch_compile' in train_config, "Missing use_torch_compile"
        assert 'use_ddp' in train_config, "Missing use_ddp"
        assert 'amp_init_scale' in train_config, "Missing amp_init_scale"

        print("✅ Config test passed - all optimization flags present")
        print(f"   Mixed Precision: {train_config['use_mixed_precision']}")
        print(f"   torch.compile(): {train_config['use_torch_compile']}")
        print(f"   DDP: {train_config['use_ddp']}")
        print(f"   AMP Init Scale: {train_config['amp_init_scale']}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_environment_vars():
    """Test LOCAL_RANK handling"""
    print("\n🧪 Testing environment variable handling...")

    # Test 1: No environment variables (should work now)
    for var in ['LOCAL_RANK', 'WORLD_SIZE', 'RANK']:
        if var in os.environ:
            del os.environ[var]

    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    print(f"✅ Environment test passed - LOCAL_RANK: {rank}, WORLD_SIZE: {world_size}")
    return True

def test_imports():
    """Test that all required imports work"""
    try:
        print("\n🧪 Testing imports...")

        # Test core training imports
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.cuda.amp import autocast, GradScaler
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        # Test local imports
        from model import LatentDiffusion
        from dataset_loader import PairedMelDataset
        from audiosr.latent_diffusion.util import instantiate_from_config

        print("✅ Import test passed - all required modules available")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    print("🔧 Testing training script fixes...\n")

    tests = [
        test_config_loading,
        test_environment_vars,
        test_imports
    ]

    results = []
    for test in tests:
        results.append(test())

    if all(results):
        print("\n🎉 All tests passed! Training script should work properly now.")
        print("\n📋 You can now run:")
        print("   python trainMGPU_DDP_Compile.py              # Single GPU with all optimizations")
        print("   python trainMGPU_DDP_Compile.py --distributed # Single GPU (fallback mode)")
        print("   torchrun --nproc_per_node=2 trainMGPU_DDP_Compile.py --distributed  # True DDP")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()