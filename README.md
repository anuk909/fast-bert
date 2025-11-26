# Fast BERT

<p align="center">
  <img src="assets/fast_bert_wide.jpg" alt="Fast BERT Logo">
</p>

Quantize, benchmark, and evaluate BERT models for sequence classification on CPU.

This project demonstrates how to optimize transformer models using:

- **PyTorch torchao** - Int8 dynamic quantization
- **ONNX Runtime + Optimum** - AVX512 VNNI quantization

## Features

- 🔧 **Model Quantization** - Create optimized model variants with reduced size
- ⚡ **Benchmarking** - Measure inference latency and throughput
- 📊 **Evaluation** - Validate accuracy on classification datasets
- 📈 **Degradation Analysis** - Compare quantized models against the original

## Setup

```bash
# Clone the repository
git clone git@github.com:anuk909/fast-bert.git
cd fast-bert

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and setup project
uv sync
```

## Usage

After installation, you can use the CLI commands or run modules directly.

### 1. Quantize Models

Creates original and quantized models using PyTorch (torchao) and ONNX Runtime:

```bash
uv run fast-bert-quantize --model-id <model-id>
```

**Options:**

| Flag         | Description          |
| ------------ | -------------------- |
| `--model-id` | HuggingFace model ID |

**Output files:**

| File                                        | Description                      |
| ------------------------------------------- | -------------------------------- |
| `models/<model-name>/pytorch/original.pth`  | Original PyTorch weights         |
| `models/<model-name>/pytorch/quantized.pth` | Int8 quantized PyTorch weights   |
| `models/<model-name>/onnx/`                 | ONNX exported model              |
| `models/<model-name>/onnx_quantized/`       | AVX512 VNNI quantized ONNX model |

### 2. Benchmark Inference

Measures latency and throughput for all model variants:

```bash
uv run fast-bert-benchmark --model-id <model-id>
```

**Options:**

| Flag            | Default | Description                     |
| --------------- | ------- | ------------------------------- |
| `--model-id`    |         | HuggingFace model ID            |
| `--num-samples` | 1000    | Number of benchmark samples     |
| `--warmup-runs` | 100     | Warmup runs before benchmarking |
| `--seq-len`     | 32      | Input sequence length           |

**Example:**

```bash
uv run fast-bert-benchmark --model-id distilbert-base-uncased-finetuned-sst-2-english --num-samples 200 --warmup-runs 20 --seq-len 64
```

### 3. Evaluate Accuracy

Evaluates models on a classification dataset:

```bash
uv run fast-bert-evaluate --model-id <model-id> --dataset <dataset> --dataset-split <split> --dataset-input-column <column>
```

**Options:**

| Flag                     | Default | Description                                                      |
| ------------------------ | ------- | ---------------------------------------------------------------- |
| `--model-id`             |         | HuggingFace model ID                                             |
| `--dataset`              |         | Dataset name to load from HuggingFace                            |
| `--dataset-config`       | None    | Dataset configuration/subset name                                |
| `--dataset-split`        |         | Dataset split name (e.g., validation, test)                      |
| `--dataset-input-column` |         | Input column name in the dataset                                 |
| `--max-samples`          | None    | Max samples to evaluate (None = full set)                        |
| `--normalize-labels`     | False   | Extract numeric prefix from labels (e.g., '0_not_relevant' -> 0) |

**Example:**

```bash
uv run fast-bert-evaluate --model-id distilbert-base-uncased-finetuned-sst-2-english --dataset glue --dataset-config sst2 --dataset-split validation --dataset-input-column sentence
```

## Example Commands

### DistilBERT (SST-2 Sentiment Classification)

```bash
# Quantize
uv run fast-bert-quantize --model-id distilbert-base-uncased-finetuned-sst-2-english

# Evaluate
uv run fast-bert-evaluate --model-id distilbert-base-uncased-finetuned-sst-2-english --dataset glue --dataset-config sst2 --dataset-split validation --dataset-input-column sentence

# Benchmark
uv run fast-bert-benchmark --model-id distilbert-base-uncased-finetuned-sst-2-english
```

### TinyBERT (Frugal AI Text Classification)

```bash
# Quantize
uv run fast-bert-quantize --model-id ParisNeo/TinyBert-frugal-ai-text-classification

# Evaluate
uv run fast-bert-evaluate --model-id ParisNeo/TinyBert-frugal-ai-text-classification --dataset QuotaClimat/frugalaichallenge-text-train --dataset-split test --dataset-input-column quote --normalize-labels

# Benchmark
uv run fast-bert-benchmark --model-id ParisNeo/TinyBert-frugal-ai-text-classification
```

## Model Variants

| Variant                | Framework              | Description                                 |
| ---------------------- | ---------------------- | ------------------------------------------- |
| PyTorch Original       | PyTorch                | Base Bert model                             |
| PyTorch Quantized      | PyTorch + torchao      | Int8 dynamic activation/weight quantization |
| ONNX Runtime           | ONNX Runtime           | Exported ONNX model                         |
| ONNX Runtime Quantized | ONNX Runtime + Optimum | AVX512 VNNI dynamic quantization            |

## Benchmark Results

Results from local machine:

- **CPU**: 11th Gen Intel Core i7-1165G7 @ 2.80GHz (4 cores, 8 threads)
- **RAM**: 16GB
- **OS**: Linux (x86_64)

### DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)

**Evaluation on SST-2 Validation Set (872 samples):**

| Model                  | Accuracy | Precision | Recall | F1 Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| PyTorch Original       | 0.9106   | 0.9110    | 0.9106 | 0.9105   |
| PyTorch Quantized      | 0.9083   | 0.9087    | 0.9083 | 0.9082   |
| ONNX Runtime           | 0.9106   | 0.9110    | 0.9106 | 0.9105   |
| ONNX Runtime Quantized | 0.9071   | 0.9075    | 0.9071 | 0.9071   |

**Benchmark (1000 samples, 100 warmup runs):**

| Framework              | Size (MB) | Latency (ms) | Std (ms) | IPS    |
| ---------------------- | --------- | ------------ | -------- | ------ |
| PyTorch Original       | 255.43    | 48.43        | 2.65     | 20.65  |
| PyTorch Quantized      | 132.44    | 28.09        | 1.86     | 35.60  |
| ONNX Runtime           | 255.52    | 14.96        | 3.58     | 66.86  |
| ONNX Runtime Quantized | 64.25     | 6.47         | 2.83     | 154.44 |

### TinyBERT (`ParisNeo/TinyBert-frugal-ai-text-classification`)

**Evaluation on Frugal AI Challenge Test Set (1219 samples, 8 classes):**

| Model                  | Accuracy | Precision | Recall | F1 Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| PyTorch Original       | 0.8031   | 0.8037    | 0.8031 | 0.8029   |
| PyTorch Quantized      | 0.7982   | 0.8000    | 0.7982 | 0.7974   |
| ONNX Runtime           | 0.8031   | 0.8037    | 0.8031 | 0.8029   |
| ONNX Runtime Quantized | 0.5176   | 0.6608    | 0.5176 | 0.4628   |

> ⚠️ **Note**: ONNX Runtime Quantized shows significant accuracy degradation for TinyBERT (-28.55% accuracy). This is likely due to the smaller model being more sensitive to quantization.

**Benchmark (1000 samples, 100 warmup runs):**

| Framework              | Size (MB) | Latency (ms) | Std (ms) | IPS    |
| ---------------------- | --------- | ------------ | -------- | ------ |
| PyTorch Original       | 54.76     | 5.70         | 0.68     | 175.59 |
| PyTorch Quantized      | 41.53     | 9.57         | 0.77     | 104.45 |
| ONNX Runtime           | 54.82     | 2.13         | 0.45     | 469.30 |
| ONNX Runtime Quantized | 13.88     | 1.37         | 0.17     | 731.84 |
