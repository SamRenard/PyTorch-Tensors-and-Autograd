# Day 51 — PyTorch Tensors & Autograd (Dataset, DataLoader, Batching)

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 2
("Mathematics & Deep Learning"), Block 4 of 5.

## Overview

This module covers PyTorch's data-loading pipeline: the `Dataset`
abstraction, wrapping it with `DataLoader` for automatic batching, why
batch size is a real design decision, and why shuffling training data is
critical to avoid biased gradient updates. It closes with a hands-on
custom `Dataset` class built from scratch and iterated over with a
`DataLoader`.

## Learning Objectives

By the end of this module you should be able to:

- Explain the two-method `Dataset` contract (`__len__`, `__getitem__`)
- Use `DataLoader` to automatically batch, shuffle, and iterate over data
- Reason about batch size trade-offs (memory, gradient noise, throughput)
- Explain why `shuffle=True` is essential for training but not for evaluation
- Distinguish an epoch (one full pass) from a batch (one loader step)
- Write a custom `Dataset` subclass that performs on-the-fly preprocessing
  (e.g. feature normalization) inside `__getitem__`

## File Structure

```
day51_pytorch/
├── 01_dataset_dataloader_basics.py  # Dataset/DataLoader fundamentals, batch size, shuffle
├── 02_custom_dataset_batching.py    # Custom Dataset class + DataLoader batch iteration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.9+
- PyTorch 2.x

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python 01_dataset_dataloader_basics.py
python 02_custom_dataset_batching.py
```

Each script prints labeled, self-explanatory output to the console.

## Key Concepts Reference

| Concept                    | API / Idea                                          |
|------------------------------|---------------------------------------------------------|
| Dataset contract               | `__len__(self)`, `__getitem__(self, idx)`                 |
| Ready-made tensor dataset          | `torch.utils.data.TensorDataset`                             |
| Automatic batching                    | `torch.utils.data.DataLoader(dataset, batch_size=...)`        |
| Shuffling training data                   | `DataLoader(..., shuffle=True)`                                |
| Parallel data loading                        | `DataLoader(..., num_workers=N)`                                 |
| Epoch                                            | One full pass over every batch in the DataLoader                  |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Study `01_dataset_dataloader_basics.py`. Pay close
   attention to the shuffle demo — understand exactly why an unshuffled,
   label-sorted dataset produces biased batches during training.
2. **Practical coding (2h):** Work through `02_custom_dataset_batching.py`.
   Try writing your own custom `Dataset` for a different toy scenario
   (e.g. text lengths, image file paths) using the same pattern.
3. **Consolidation:** Write a 5–10 sentence summary in your own words plus
   the key terms below (second-brain habit).
4. **Commit:** Push this folder to GitHub, wait 5 minutes, re-read your own
   code, and improve one thing before finalizing.

## Key Terms

`Dataset`, `DataLoader`, `__len__`, `__getitem__`, `batch_size`, `shuffle`,
`epoch`, `TensorDataset`, `num_workers`, `drop_last`

## Reference Sources

- PyTorch official tutorials — https://pytorch.org/tutorials/
- Andrej Karpathy — micrograd (educational autograd engine)
- "Neural Networks: Zero to Hero" playlist by Andrej Karpathy

## Progress

Day 51 / 150 — Streak: 1 🔥 — Overall completion: 33%
