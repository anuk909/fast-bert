import argparse
from typing import Any

import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from fast_bert.config import load_onnx, load_pytorch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Bert model variants on relevant sequence classification dataset."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g., 'ParisNeo/TinyBert-frugal-ai-text-classification'). "
             "If provided, loads models from a directory based on the model name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to load from HuggingFace",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration/subset name",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        help="Dataset split name",
    )
    parser.add_argument(
        "--dataset-input-column",
        type=str,
        help="Input column name in the dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: None = full validation set)",
    )
    parser.add_argument(
        "--normalize-labels",
        action="store_true",
        help="Extract numeric prefix from labels (e.g., '0_not_relevant' -> 0)",
    )
    return parser.parse_args()


def load_samples_from_dataset(
    dataset_name: str,
    dataset_config: str | None,
    input_column: str,
    split: str,
    max_samples: int | None = None,
    normalize_labels: bool = False,
) -> tuple[list[str], list[int]]:
    """Load samples from a dataset.
    
    Returns:
        texts: List of input texts
        labels: List of integer labels
    """
    print(f"Loading {dataset_name}/{dataset_config} validation dataset...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    texts = list(dataset[input_column])
    labels = list(dataset["label"])

    if max_samples is not None:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    # Convert labels to integers
    if normalize_labels:
        # Extract numeric prefix from labels like "0_not_relevant" -> 0
        labels = [int(str(label).split("_")[0]) for label in labels]
        print(f"Normalized labels from prefix (e.g., '{labels[0]}' -> {labels[0]})")
    else:
        labels = [int(label) for label in labels]

    print(f"Loaded {len(texts)} samples with {len(set(labels))} classes")
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
                truncation=True,
                max_length=model.config.max_position_embeddings,
            )

            outputs = model(**inputs)
            predicted_class = outputs.logits.argmax(-1).item()
            predictions.append(predicted_class)

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
        "Precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(labels, predictions, average="weighted", zero_division=0),
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

    texts, labels = load_samples_from_dataset(
        args.dataset,
        args.dataset_config,
        args.dataset_input_column,
        args.dataset_split,
        args.max_samples,
        args.normalize_labels,
    )

    model_id = args.model_id

    tasks = {
        "PyTorch Original": lambda: load_pytorch(is_quantized=False, model_id=model_id),
        "PyTorch Quantized": lambda: load_pytorch(is_quantized=True, model_id=model_id),
        "ONNX Runtime": lambda: load_onnx(is_quantized=False, model_id=model_id),
        "ONNX Runtime Quantized": lambda: load_onnx(is_quantized=True, model_id=model_id),
    }

    results = [
        evaluate_model(name, loader, texts, labels)
        for name, loader in tasks.items()
    ]
    print_results([r for r in results if r])


if __name__ == "__main__":
    main()
