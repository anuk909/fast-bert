"""
Shared configuration and utilities for fast-bert.
"""

import os
import warnings
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Model configuration
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODELS_DIR = Path("models")

# File names
PYTORCH_ORIGINAL = "model_pytorch_original.pth"
PYTORCH_QUANTIZED = "model_pytorch_quantized.pth"
ONNX_DIR = "model_onnx"
ONNX_QUANTIZED_DIR = "model_onnx_quantized"

# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.set_num_threads(1)

# Suppress TF32 deprecation warning from torchao (uses old API internally)
# See: https://github.com/pytorch/ao/issues - torchao needs to update to new TF32 API
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior",
    category=UserWarning,
)


def get_file_size(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def load_pytorch(is_quantized: bool) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer, float] | None:
    """Load original PyTorch model."""
    path = MODELS_DIR / (PYTORCH_QUANTIZED if is_quantized else PYTORCH_ORIGINAL)
    if not path.exists():
        return None
    print(f"Loading PyTorch model from {path} (Quantized: {is_quantized})")

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID)
    if is_quantized:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer, get_file_size(path)

def load_onnx(is_quantized: bool) -> tuple[ORTModelForSequenceClassification, DistilBertTokenizer, float] | None:
    """Load ONNX quantized model."""
    onnx_dir = MODELS_DIR / (ONNX_QUANTIZED_DIR if is_quantized else ONNX_DIR)
    onnx_file = onnx_dir / "model_quantized.onnx" if is_quantized else onnx_dir / "model.onnx"
    if not onnx_file.exists():
        return None
    print(f"Loading ONNX model from {onnx_file} (Quantized: {is_quantized})")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForSequenceClassification.from_pretrained(
        onnx_dir, file_name=onnx_file.name
    )

    return model, tokenizer, get_file_size(onnx_file)
