"""
Day 51 - PyTorch Tensors & Autograd
Module 2: Custom Dataset Class + DataLoader Batch Iteration Practice

Task: write a custom Dataset class from scratch and use a DataLoader to
iterate over it in batches.

Scenario: a small synthetic "housing price" dataset loaded from an
in-memory list of records (standing in for a CSV/database source), with
on-the-fly feature normalization applied inside __getitem__.

Run:
    python 02_custom_dataset_batching.py
"""

import torch
from torch.utils.data import Dataset, DataLoader


class HousingDataset(Dataset):
    """
    Custom Dataset for a small housing price example.

    Each raw record is a dict: {"sqft": float, "bedrooms": int, "price": float}
    __getitem__ converts a record into normalized (features, target) tensors
    on the fly -- this is the standard pattern for custom preprocessing.
    """

    def __init__(self, records: list[dict]):
        self.records = records

        # Precompute normalization statistics from the full dataset.
        sqft_values = torch.tensor([r["sqft"] for r in records], dtype=torch.float32)
        bedroom_values = torch.tensor([r["bedrooms"] for r in records], dtype=torch.float32)

        self.sqft_mean, self.sqft_std = sqft_values.mean(), sqft_values.std()
        self.bedroom_mean, self.bedroom_std = bedroom_values.mean(), bedroom_values.std()

    def __len__(self) -> int:
        # Required by the Dataset contract: total number of samples.
        return len(self.records)

    def __getitem__(self, idx: int):
        # Required by the Dataset contract: return one (features, target) pair.
        record = self.records[idx]

        sqft_norm = (record["sqft"] - self.sqft_mean) / self.sqft_std
        bedrooms_norm = (record["bedrooms"] - self.bedroom_mean) / self.bedroom_std

        features = torch.tensor([sqft_norm, bedrooms_norm], dtype=torch.float32)
        target = torch.tensor([record["price"]], dtype=torch.float32)

        return features, target


def generate_raw_records(n: int = 24) -> list[dict]:
    """Simulate loading raw, unnormalized records from an external source."""
    torch.manual_seed(1)
    records = []
    for _ in range(n):
        sqft = float(torch.randint(600, 3000, (1,)).item())
        bedrooms = int(torch.randint(1, 6, (1,)).item())
        # A synthetic, noisy price relationship
        price = 50_000 + sqft * 120 + bedrooms * 8_000 + float(torch.randn(1).item()) * 5_000
        records.append({"sqft": sqft, "bedrooms": bedrooms, "price": price})
    return records


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("1. Building the Custom Dataset")

    raw_records = generate_raw_records(n=24)
    dataset = HousingDataset(raw_records)

    print("Total samples:", len(dataset))
    features_0, target_0 = dataset[0]
    print("Sample 0 -> features (normalized):", features_0, "target (price):", target_0)

    section("2. Wrapping It with a DataLoader")

    train_loader = DataLoader(dataset, batch_size=6, shuffle=True, drop_last=False)

    print(f"Total batches per epoch: {len(train_loader)}\n")
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: features shape={tuple(x_batch.shape)}, "
              f"targets shape={tuple(y_batch.shape)}")

    section("3. Full Iteration Across Two Epochs")

    for epoch in range(1, 3):
        print(f"\nEpoch {epoch}:")
        running_sum = 0.0
        for x_batch, y_batch in train_loader:
            running_sum += y_batch.sum().item()
        print(f"  Sum of all target prices seen this epoch: {running_sum:.2f}")

    section("4. Note on num_workers")

    print(
        "DataLoader(..., num_workers=N) can load batches in parallel using N\n"
        "worker subprocesses, which overlaps data loading with GPU computation.\n"
        "It is omitted here for simplicity and cross-platform script portability,\n"
        "but is recommended for real datasets with expensive I/O or preprocessing."
    )


if __name__ == "__main__":
    main()
