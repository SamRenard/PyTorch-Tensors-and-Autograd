"""
Day 51 - PyTorch Tensors & Autograd
Module 1: Dataset, DataLoader, Batching, and the Importance of Shuffling

Covers:
    - The Dataset abstraction: __len__ and __getitem__
    - Wrapping a Dataset with DataLoader for automatic batching
    - Why batch size matters (memory, gradient noise, throughput)
    - Why shuffle=True matters for training (avoiding order bias)
    - Iterating over epochs vs. batches

Run:
    python 01_dataset_dataloader_basics.py
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def dataset_interface_demo() -> None:
    section("1. The Dataset Interface: __len__ and __getitem__")

    # torch.utils.data.Dataset is an abstract class with exactly two
    # required methods. PyTorch relies on this minimal contract to plug
    # ANY data source into the same DataLoader machinery.
    features = torch.randn(10, 3)
    labels = torch.randint(0, 2, (10,))

    # TensorDataset is a ready-made Dataset for simple tensor pairs
    dataset = TensorDataset(features, labels)

    print("Dataset length (__len__)   :", len(dataset))
    print("First item     (__getitem__):", dataset[0])
    print("Item type:", type(dataset[0]))


def dataloader_batching_demo() -> None:
    section("2. DataLoader: Automatic Batching")

    features = torch.arange(20).float().reshape(10, 2)
    labels = torch.arange(10)
    dataset = TensorDataset(features, labels)

    # DataLoader wraps a Dataset and yields batches instead of single samples.
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"Dataset size: {len(dataset)}, batch_size=4 -> {len(loader)} batches\n")
    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {batch_idx}: x shape={tuple(x_batch.shape)}, y={y_batch.tolist()}")


def batch_size_tradeoffs() -> None:
    section("3. Why Batch Size Matters")

    print(
        "Small batch size (e.g. 8-32):\n"
        "  + Noisier gradient estimates can help escape sharp local minima\n"
        "  + Lower memory usage\n"
        "  - Slower wall-clock training (less parallelism per step)\n"
    )
    print(
        "Large batch size (e.g. 256+):\n"
        "  + Smoother, more stable gradient estimates\n"
        "  + Better hardware utilization (GPU parallelism)\n"
        "  - Higher memory usage\n"
        "  - May generalize slightly worse without learning-rate adjustments\n"
    )


def shuffle_importance_demo() -> None:
    section("4. Why shuffle=True Matters")

    # Simulate a dataset that is sorted by label (a very common real-world
    # scenario: e.g. all "cat" images first, then all "dog" images).
    features = torch.arange(12).float().reshape(12, 1)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    dataset = TensorDataset(features, labels)

    print("Without shuffling (shuffle=False):")
    loader_no_shuffle = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch_idx, (_, y_batch) in enumerate(loader_no_shuffle):
        print(f"  Batch {batch_idx} labels: {y_batch.tolist()}  <- all one class!")

    print("\nWith shuffling (shuffle=True):")
    torch.manual_seed(7)
    loader_shuffle = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_idx, (_, y_batch) in enumerate(loader_shuffle):
        print(f"  Batch {batch_idx} labels: {y_batch.tolist()}  <- mixed classes")

    print(
        "\nWithout shuffling, each batch only contains one class, so the model\n"
        "sees a biased gradient direction per step -- it can overfit to whatever\n"
        "class it saw most recently. Shuffling breaks this ordering bias.\n"
        "Rule of thumb: shuffle=True for training, shuffle=False for evaluation\n"
        "(evaluation order does not affect learning and reproducibility is nice)."
    )


def epoch_vs_batch_iteration() -> None:
    section("5. Epoch vs. Batch Iteration")

    dataset = TensorDataset(torch.randn(9, 2), torch.randint(0, 2, (9,)))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    n_epochs = 2
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            print(f"  batch {batch_idx}: {x_batch.shape[0]} samples")
        # Note: the last batch may be smaller if len(dataset) % batch_size != 0
    print(
        "\nAn epoch = one full pass over the dataset (all batches).\n"
        "DataLoader automatically reshuffles between epochs when shuffle=True."
    )


if __name__ == "__main__":
    dataset_interface_demo()
    dataloader_batching_demo()
    batch_size_tradeoffs()
    shuffle_importance_demo()
    epoch_vs_batch_iteration()
