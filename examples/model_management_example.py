#!/usr/bin/env python3
"""
Example script demonstrating SynthDoc's automatic model downloading feature.

This script shows how models are automatically downloaded when needed.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import synthdoc
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("🚀 SynthDoc Model Management Example")
    print("=" * 50)

    # Import SynthDoc - models will be checked during initialization
    print("\n1. Importing SynthDoc...")
    from synthdoc import SynthDoc, list_available_models, get_model_info

    # Show available models
    print("\n2. Available models:")
    models = list_available_models()
    for name, config in models.items():
        print(f"   📦 {name}: {config['description']} (~{config['size_mb']}MB)")

    # Show model status
    print("\n3. Checking model status...")
    for model_name in models:
        try:
            info = get_model_info(model_name)
            status = "✅ Downloaded" if info["downloaded"] else "❌ Not downloaded"
            print(f"   {model_name}: {status}")
        except Exception as e:
            print(f"   {model_name}: ❓ Error - {e}")

    # Initialize SynthDoc - this will show model status
    print("\n4. Initializing SynthDoc...")
    synth = SynthDoc(output_dir="./example_output")
    print(f"   ✅ SynthDoc initialized with output dir: {synth.output_dir}")

    # Example: Translate a document (this will auto-download the YOLO model if needed)
    print("\n5. Testing document translation (will auto-download model if needed)...")
    try:
        # Note: This would normally require actual image files
        # This is just to show the model download behavior
        print("   Note: In a real scenario, you would provide actual image files")
        print("   The YOLO model would be automatically downloaded when needed")

        # Show current model status after initialization
        print("\n6. Model status after initialization:")
        for model_name in models:
            try:
                info = get_model_info(model_name)
                status = "✅ Downloaded" if info["downloaded"] else "❌ Not downloaded"
                if info["downloaded"]:
                    status += f" ({info.get('local_size_mb', '?')}MB)"
                print(f"   {model_name}: {status}")
            except Exception as e:
                print(f"   {model_name}: ❓ Error - {e}")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n💡 Tips:")
    print("   • Models are automatically downloaded when first needed")
    print("   • Use 'synthdoc list-models' to see all available models")
    print("   • Use 'synthdoc download-models <model_name>' to download manually")
    print("   • Use 'synthdoc clean-models' to remove downloaded models")

    print("\n✅ Example completed!")


if __name__ == "__main__":
    main()
