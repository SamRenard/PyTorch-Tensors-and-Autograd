# Day 48 — PyTorch Tensors & Autograd

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 2
("Mathematics & Deep Learning"), Block 1 of 5.

## Overview

This module covers the foundation of PyTorch: tensors as the core data
structure, the operations you'll use constantly, the NumPy interoperability
layer, and PyTorch's automatic differentiation engine (Autograd), which
powers gradient-based training for every neural network.

## Learning Objectives

By the end of this module you should be able to:

- Create tensors from Python data, NumPy arrays, and built-in factory functions
- Inspect and reason about tensor shape, dtype, and device
- Perform arithmetic, matrix, reduction, and reshaping operations on tensors
- Understand the shared-memory relationship between NumPy arrays and CPU tensors
- Explain how `requires_grad` and `.backward()` build and traverse the
  computation graph
- Manage gradient accumulation correctly (`.zero_()`) between optimization steps
- Disable gradient tracking when appropriate (`torch.no_grad()`, `.detach()`)
- Write device-agnostic code that runs on CPU or GPU without modification

## File Structure

```
day48_pytorch/
├── 01_tensor_basics.py              # Tensor creation, attributes, ops, NumPy bridge
├── 02_autograd_basics.py            # Autograd fundamentals and a manual gradient step
├── 03_tensor_operations_practice.py # 20 tensor operations drill + device management
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.9+
- PyTorch 2.x
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run each script independently:

```bash
python 01_tensor_basics.py
python 02_autograd_basics.py
python 03_tensor_operations_practice.py
```

Each script prints labeled, self-explanatory output to the console so you
can follow along step by step.

## Key Concepts Reference

| Concept              | API                                          |
|-----------------------|----------------------------------------------|
| Tensor creation        | `torch.tensor()`, `torch.zeros()`, `torch.rand()` |
| NumPy bridge            | `.numpy()`, `torch.from_numpy()`              |
| Gradient tracking        | `requires_grad=True`                          |
| Backward pass              | `.backward()`                                 |
| Gradient reset               | `.grad.zero_()`                               |
| Disable tracking               | `torch.no_grad()`, `.detach()`                |
| Device transfer                  | `.to(device)`, `torch.cuda.is_available()`     |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Read the PyTorch Basics tutorial linked below, then
   run `01_tensor_basics.py` and `02_autograd_basics.py` while reading the
   inline comments.
2. **Practical coding (2h):** Work through `03_tensor_operations_practice.py`,
   then modify it — add your own tensors, break things, observe the errors.
3. **Consolidation:** Write a short summary in your own words (5–10
   sentences) covering the key terms below.
4. **Commit:** Push this folder to GitHub, wait 5 minutes, re-read your own
   code, and improve one thing before finalizing.

## Key Terms

`tensor`, `dtype`, `device`, `broadcasting`, `view` vs `clone`,
`requires_grad`, `computation graph`, `autograd`, `backward pass`,
`gradient accumulation`, `detach`, `CUDA`

## Reference Sources

- PyTorch official tutorials — https://pytorch.org/tutorials/
- Andrej Karpathy — micrograd (educational autograd engine)
- "Neural Networks: Zero to Hero" playlist by Andrej Karpathy

## Progress

Day 48 / 150 — Streak: 6 🔥 — Overall completion: 31%
