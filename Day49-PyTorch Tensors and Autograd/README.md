# Day 49 — PyTorch Tensors & Autograd (Autograd Mechanics)

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 2
("Mathematics & Deep Learning"), Block 2 of 5.

## Overview

This module goes one level deeper into PyTorch's Autograd engine: how the
computation graph is actually built and traversed, what `requires_grad` and
`grad_fn` mean under the hood, and how `.backward()` applies the chain rule
automatically. It also includes a hand-derived vs. autograd-verified
derivative exercise with a computation graph diagram.

## Learning Objectives

By the end of this module you should be able to:

- Distinguish leaf tensors from derived tensors in the computation graph
- Read a tensor's `grad_fn` to understand which operation produced it
- Explain, step by step, how `.backward()` walks the graph in reverse
- Manually derive the derivative of a composite function and verify it
  against PyTorch's autograd output
- Understand why the graph is freed after `.backward()` by default, and
  when `retain_graph=True` is needed

## File Structure

```
day49_pytorch/
├── 01_computation_graph.py               # Graph mechanics, grad_fn, chain rule
├── 02_manual_derivative_verification.py  # Hand-derived derivative + ASCII graph diagram
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
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
python 01_computation_graph.py
python 02_manual_derivative_verification.py
```

Each script prints labeled, self-explanatory output to the console.

## Key Concepts Reference

| Concept                      | API / Idea                                   |
|-------------------------------|------------------------------------------------|
| Leaf tensor                     | `tensor.is_leaf`, created directly by the user |
| Graph node reference               | `tensor.grad_fn`                                |
| Parent links                          | `grad_fn.next_functions`                        |
| Backward pass                            | `.backward()` — reverse-mode autodiff           |
| Repeated backward calls                     | `retain_graph=True`                             |
| Chain rule                                     | Applied automatically, edge by edge, in reverse |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Study `01_computation_graph.py` — read the code,
   run it, and make sure you can explain in your own words what a leaf
   tensor is and how `.backward()` traverses the graph.
2. **Practical coding (2h):** Work through `02_manual_derivative_verification.py`.
   Pick your own simple function, derive its derivative by hand on paper,
   sketch the computation graph, then verify your result with autograd.
3. **Consolidation:** Write a 5–10 sentence summary in your own words plus
   the key terms below (second-brain habit).
4. **Commit:** Push this folder to GitHub, wait 5 minutes, re-read your own
   code, and improve one thing before finalizing.

## Key Terms

`computation graph`, `leaf tensor`, `grad_fn`, `next_functions`,
`reverse-mode autodiff`, `chain rule`, `retain_graph`, `backward pass`

## Reference Sources

- PyTorch official tutorials — https://pytorch.org/tutorials/
- Andrej Karpathy — micrograd (educational autograd engine)
- "Neural Networks: Zero to Hero" playlist by Andrej Karpathy

## Progress

Day 49 / 150 — Streak: 6 🔥 — Overall completion: 31%
