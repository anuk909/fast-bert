import argparse

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fast_bert.config import (
    ONNX_DIR,
    ONNX_ORIGINAL_FILE,
    ONNX_QUANTIZED_FILE,
    PYTORCH_DIR,
    PYTORCH_QUANTIZED_FILE,
    PYTORCH_ORIGINAL_FILE,
    get_file_size,
    get_models_dir,
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create original and quantized PyTorch/ONNX models."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID",
    )
    return parser.parse_args()


def create_pytorch_models(model_id: str) -> None:
    """Create original and quantized PyTorch models."""
    print("=" * 60)
    print("Creating PyTorch Models (Original & Quantized)")
    print("=" * 60)

    models_dir = get_models_dir(model_id)
    pytorch_dir = models_dir / PYTORCH_DIR
    pytorch_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    print(f"Loading model: {model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Save original model and tokenizer to directory
    print(f"Saving original model to {pytorch_dir}")
    model.save_pretrained(pytorch_dir)
    tokenizer.save_pretrained(pytorch_dir)
    
    # Calculate size of saved model files
    total_size = get_file_size(pytorch_dir / PYTORCH_ORIGINAL_FILE)
    print(f"Saved original model, Size: {total_size:.2f} MB")

    # Load fresh model for quantization (quantize_ modifies in-place)
    print("Loading fresh model for quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Quantize and save
    print("Applying int8 dynamic quantization (torchao)...")
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    # torchao quantized models can't use save_pretrained (incompatible tensor storage)
    # Save state_dict with torch.save to separate file
    weights_path = pytorch_dir / PYTORCH_QUANTIZED_FILE
    print(f"Saving quantized model to {weights_path}")
    torch.save(model.state_dict(), weights_path)
    
    print(f"Saved quantized model, Size: {get_file_size(weights_path):.2f} MB")


def create_onnx_quantized(model_id: str) -> None:
    """Create original and quantized ONNX models."""
    print("\n" + "=" * 60)
    print("Creating ONNX Models (Original & Quantized)")
    print("=" * 60)

    models_dir = get_models_dir(model_id)
    onnx_dir = models_dir / ONNX_DIR
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print("Exporting model to ONNX format...")
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(onnx_dir)
    print(f"  Saved original model, Size: {get_file_size(onnx_dir / ONNX_ORIGINAL_FILE):.2f} MB")

    # Save tokenizer with ONNX model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(onnx_dir)

    # Quantize (saves to same directory with different filename)
    print("Quantizing ONNX model (AVX512 VNNI dynamic)...")
    quantizer = ORTQuantizer.from_pretrained(model)
    dq_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=onnx_dir, quantization_config=dq_config)
    print(f"  Saved quantized model, Size: {get_file_size(onnx_dir / ONNX_QUANTIZED_FILE):.2f} MB")


def main() -> None:
    """Run all quantization pipelines."""
    args = parse_args()
    create_pytorch_models(args.model_id)
    create_onnx_quantized(args.model_id)
    print("\n" + "=" * 60)
    print("All models created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
