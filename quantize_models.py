"""
Quantize DistilBERT models using PyTorch (torchao) and ONNX Runtime (Optimum).

Usage:
    uv run python quantize_models.py
"""

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from config import (
    MODEL_ID,
    MODELS_DIR,
    ONNX_DIR,
    ONNX_QUANTIZED_DIR,
    PYTORCH_ORIGINAL,
    PYTORCH_QUANTIZED,
)


def create_pytorch_models() -> None:
    """Create original and quantized PyTorch models."""
    print("=" * 60)
    print("Creating PyTorch Models (Original & Quantized)")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    # Load pre-trained model
    print(f"Loading model: {MODEL_ID}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID)

    # Save original
    original_path = MODELS_DIR / PYTORCH_ORIGINAL
    print(f"Saving original model to {original_path}")
    torch.save(model.state_dict(), original_path)
    print(f"  Size: {original_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Quantize and save
    print("Applying int8 dynamic quantization (torchao)...")
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    quantized_path = MODELS_DIR / PYTORCH_QUANTIZED
    print(f"Saving quantized model to {quantized_path}")
    torch.save(model.state_dict(), quantized_path)
    print(f"  Size: {quantized_path.stat().st_size / 1024 / 1024:.2f} MB")


def create_onnx_quantized() -> None:
    """Create quantized ONNX model using Optimum."""
    print("\n" + "=" * 60)
    print("Creating ONNX Quantized Model (Optimum)")
    print("=" * 60)

    onnx_path = MODELS_DIR / ONNX_DIR
    onnx_quantized_path = MODELS_DIR / ONNX_QUANTIZED_DIR

    # Export to ONNX
    print("Exporting model to ONNX format...")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(onnx_path)
    print(f"  Saved to {onnx_path}")

    # Quantize
    print("Quantizing ONNX model (AVX512 VNNI dynamic)...")
    quantizer = ORTQuantizer.from_pretrained(model)
    dq_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=onnx_quantized_path, quantization_config=dq_config)
    print(f"  Saved to {onnx_quantized_path}")

    # Save tokenizer with quantized model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(onnx_quantized_path)


def main() -> None:
    """Run all quantization pipelines."""
    create_pytorch_models()
    create_onnx_quantized()
    print("\n" + "=" * 60)
    print("All models created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
