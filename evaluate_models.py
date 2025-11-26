import argparse
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from config import load_onnx, load_pytorch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Bert model variants on SST-2 sentiment classification dataset."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HuggingFace model ID (default: distilbert-base-uncased-finetuned-sst-2-english)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="glue",
        help="Dataset name to load from HuggingFace (default: glue)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="sst2",
        help="Dataset configuration/subset name (default: sst2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None = full validation set)",
    )
    return parser.parse_args()


def load_sst2_validation(
    dataset_name: str, dataset_config: str, max_samples: int | None = None
) -> tuple[list[str], list[int]]:
    """Load SST-2 validation dataset."""
    print(f"Loading {dataset_name}/{dataset_config} validation dataset...")
    dataset = load_dataset(dataset_name, dataset_config, split="validation")

    texts = list(dataset["sentence"])
    labels = list(dataset["label"])

    if max_samples is not None:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    print(f"Loaded {len(texts)} samples")
    return texts, labels


def predict(
    model: Any, name: str, tokenizer: Any, texts: list[str]
) -> list[int]:
    """Run predictions on texts one at a time."""
    predictions = []

    with torch.no_grad():
        for text in tqdm(texts, desc=f"Evaluating {name}"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
            )

            outputs = model(**inputs)
            logits = outputs.logits

            # Handle both torch tensors and numpy arrays
            logits_np = logits.numpy() if hasattr(logits, "numpy") else logits
            pred = np.argmax(logits_np, axis=1).item()
            predictions.append(pred)

    return predictions


def evaluate_model(
    name: str,
    loader_fn: Any,
    texts: list[str],
    labels: list[int],
) -> dict | None:
    """Evaluate a single model."""
    print(f"\n[{'=' * 10} {name.upper()} {'=' * 10}]")

    loaded = loader_fn()
    if not loaded:
        print("   Skipping: Model files not found.")
        return None

    model, tokenizer, _file_size = loaded

    # Get predictions
    predictions = predict(model, name, tokenizer, texts)

    return {
        "Model": name,
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, average="binary"),
        "Recall": recall_score(labels, predictions, average="binary"),
        "F1 Score": f1_score(labels, predictions, average="binary"),
    }


def print_results(results: list[dict]) -> None:
    """Print evaluation summary with degradation analysis."""
    if not results:
        return

    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY - SST-2 Validation Set")
    print("=" * 100)
    print(
        f"{'Model':<25} | {'Accuracy':<10} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['Model']:<25} | {r['Accuracy']:<10.4f} | "
            f"{r['Precision']:<10.4f} | {r['Recall']:<10.4f} | {r['F1 Score']:<10.4f}"
        )

    print("=" * 100)

    # Degradation analysis
    if len(results) > 1:
        baseline = results[0]
        print("\n📊 Degradation Analysis (vs. Original PyTorch):")
        print("-" * 60)

        for r in results[1:]:
            acc_diff = r["Accuracy"] - baseline["Accuracy"]
            f1_diff = r["F1 Score"] - baseline["F1 Score"]
            status = "✅" if f1_diff >= -0.01 else "⚠️"

            print(f"\n{status} {r['Model']}:")
            print(f"   Accuracy:  {acc_diff:+.4f}")
            print(f"   F1 Score:  {f1_diff:+.4f}")


def main() -> None:
    """Run evaluation for all models."""
    args = parse_args()

    texts, labels = load_sst2_validation(args.dataset, args.dataset_config, args.max_samples)

    tasks = {
        "PyTorch Original": lambda: load_pytorch(is_quantized=False),
        "PyTorch Quantized": lambda: load_pytorch(is_quantized=True),
        "ONNX Runtime": lambda: load_onnx(is_quantized=False),
        "ONNX Runtime Quantized": lambda: load_onnx(is_quantized=True),
    }

    results = [
        evaluate_model(name, loader, texts, labels)
        for name, loader in tasks.items()
    ]
    print_results([r for r in results if r])


if __name__ == "__main__":
    main()
