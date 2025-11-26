# Fast BERT

Quantize, benchmark, and evaluate DistilBERT models for sentiment classification.

This project demonstrates how to optimize transformer models using:

- **PyTorch torchao** - Int8 dynamic quantization
- **ONNX Runtime + Optimum** - AVX512 VNNI quantization

## Features

- 🔧 **Model Quantization** - Create optimized model variants with reduced size
- ⚡ **Benchmarking** - Measure inference latency and throughput
- 📊 **Evaluation** - Validate accuracy on SST-2 sentiment classification dataset
- 📈 **Degradation Analysis** - Compare quantized models against the original

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv sync
```

## Usage

### 1. Quantize Models

Creates original and quantized models using PyTorch (torchao) and ONNX Runtime:

```bash
uv run python quantize_models.py
```

**Output files:**

| File                                 | Description                      |
| ------------------------------------ | -------------------------------- |
| `models/model_pytorch_original.pth`  | Original PyTorch weights         |
| `models/model_pytorch_quantized.pth` | Int8 quantized PyTorch weights   |
| `models/model_onnx/`                 | ONNX exported model              |
| `models/model_onnx_quantized/`       | AVX512 VNNI quantized ONNX model |

### 2. Benchmark Inference

Measures latency and throughput for all model variants:

```bash
uv run python benchmark_models.py
```

**Options:**

| Flag            | Default | Description                     |
| --------------- | ------- | ------------------------------- |
| `--num-samples` | 100     | Number of benchmark samples     |
| `--warmup-runs` | 10      | Warmup runs before benchmarking |
| `--seq-len`     | 32      | Input sequence length           |

**Example:**

```bash
uv run python benchmark_models.py --num-samples 200 --warmup-runs 20 --seq-len 64
```

### 3. Evaluate Accuracy

Evaluates models on SST-2 sentiment classification validation set:

```bash
uv run python evaluate_models.py
```

**Options:**

| Flag            | Default | Description                               |
| --------------- | ------- | ----------------------------------------- |
| `--batch-size`  | 32      | Batch size for evaluation                 |
| `--max-samples` | None    | Max samples to evaluate (None = full set) |

**Example:**

```bash
uv run python evaluate_models.py --batch-size 64 --max-samples 500
```

## Model Variants

| Variant                | Framework              | Description                                                               |
| ---------------------- | ---------------------- | ------------------------------------------------------------------------- |
| PyTorch Original       | PyTorch                | Base DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) |
| PyTorch Quantized      | PyTorch + torchao      | Int8 dynamic activation/weight quantization                               |
| ONNX Runtime           | ONNX Runtime           | Exported ONNX model                                                       |
| ONNX Runtime Quantized | ONNX Runtime + Optimum | AVX512 VNNI dynamic quantization                                          |

## Project Structure

```
fast-bert/
├── config.py              # Shared configuration and model loaders
├── quantize_models.py     # Model quantization pipeline
├── benchmark_models.py    # Inference benchmarking
├── evaluate_models.py     # Accuracy evaluation on SST-2
├── pyproject.toml         # Project dependencies
└── models/                # Generated model files
    ├── model_pytorch_original.pth
    ├── model_pytorch_quantized.pth
    ├── model_onnx/
    └── model_onnx_quantized/
```

## Dependencies

- `torch` - PyTorch deep learning framework
- `torchao` - PyTorch quantization library
- `transformers` - Hugging Face transformers
- `optimum[onnxruntime]` - ONNX Runtime optimization
- `datasets` - Hugging Face datasets
- `scikit-learn` - Evaluation metrics
