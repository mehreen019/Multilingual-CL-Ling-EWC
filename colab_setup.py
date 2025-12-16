"""
Setup script for Google Colab environment.
Run this first in your Colab notebook.
"""

def setup_colab():
    """Install dependencies and setup environment for Colab."""
    import subprocess
    import sys

    print("="*60)
    print("Setting up Linguistically-Aware EWC Experiment")
    print("="*60)

    # Install required packages
    print("\n📦 Installing dependencies...")
    packages = [
        'torch',
        'transformers',
        'datasets',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]

    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("\n✅ Installation complete!")

    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  No GPU detected. Training will be slower on CPU.")
        print("   Consider enabling GPU in Runtime > Change runtime type > GPU")

    print("\n" + "="*60)
    print("Setup complete! You can now run the experiment.")
    print("="*60)


def test_data_loading():
    """Test if data loading works correctly."""
    print("\n🧪 Testing data loading...")

    try:
        from datasets import load_dataset

        # Test loading IndicSentiment
        print("  Loading Bengali sample...")
        dataset_bn = load_dataset("ai4bharat/IndicSentiment", "bn", split="train", trust_remote_code=True)
        print(f"  ✓ Bengali: {len(dataset_bn)} samples")

        print("  Loading Hindi sample...")
        dataset_hi = load_dataset("ai4bharat/IndicSentiment", "hi", split="train", trust_remote_code=True)
        print(f"  ✓ Hindi: {len(dataset_hi)} samples")

        # Show sample
        sample = dataset_bn[0]
        print(f"\n  Sample Bengali review:")
        print(f"    Text: {sample.get('INDIC REVIEW', 'N/A')[:100]}...")
        print(f"    Label: {sample.get('LABEL', 'N/A')}")

        print("\n✅ Data loading test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        print("  This is okay - data will be loaded during training.")
        return False


if __name__ == "__main__":
    setup_colab()
    test_data_loading()
