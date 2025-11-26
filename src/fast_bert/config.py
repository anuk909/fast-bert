import os
import warnings
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Directory and file names
MODELS_DIR = Path("models")
PYTORCH_DIR = "pytorch"
PYTORCH_ORIGINAL_FILE = "model.safetensors"
PYTORCH_QUANTIZED_FILE = "model_quantized.pth"

ONNX_DIR = "onnx"
ONNX_ORIGINAL_FILE = "model.onnx"
ONNX_QUANTIZED_FILE = "model_quantized.onnx"

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


def get_models_dir(model_id: str) -> Path:
    """Get the models directory based on model name extracted from HuggingFace model ID."""
    return Path(MODELS_DIR / model_id.split("/")[-1])


def get_file_size(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def load_pytorch(
    is_quantized: bool, model_id: str
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, float] | None:
    """Load PyTorch model (original or quantized) from disk."""
    models_dir = get_models_dir(model_id)
    model_dir = models_dir / PYTORCH_DIR
    if not model_dir.exists():
        return None
    
    weights_file = PYTORCH_QUANTIZED_FILE if is_quantized else PYTORCH_ORIGINAL_FILE
    weights_path = model_dir / weights_file
    if not weights_path.exists():
        return None
    print(f"Loading PyTorch model from {weights_path} (Quantized: {is_quantized})")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if is_quantized:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer, get_file_size(weights_path)

def load_onnx(
    is_quantized: bool, model_id: str
) -> tuple[ORTModelForSequenceClassification, AutoTokenizer, float] | None:
    """Load ONNX model (original or quantized) from disk."""
    models_dir = get_models_dir(model_id)
    onnx_dir = models_dir / ONNX_DIR
    onnx_file = ONNX_QUANTIZED_FILE if is_quantized else ONNX_ORIGINAL_FILE
    onnx_path = onnx_dir / onnx_file
    if not onnx_path.exists():
        return None
    print(f"Loading ONNX model from {onnx_path} (Quantized: {is_quantized})")

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    model = ORTModelForSequenceClassification.from_pretrained(
        onnx_dir, file_name=onnx_file
    )

    return model, tokenizer, get_file_size(onnx_path)
