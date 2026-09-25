# Day 50 — PyTorch Tensors & Autograd (nn.Module, Optimizers, Linear Regression)

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 2
("Mathematics & Deep Learning"), Block 3 of 5.

## Overview

This module moves from raw tensors and autograd to PyTorch's model-building
API: `nn.Module`. It covers how modules are structured internally, how
learnable parameters are tracked and exposed, how optimizers (SGD, Adam)
update those parameters, and closes with a complete, end-to-end linear
regression implementation using `nn.Linear`, `MSELoss`, and a proper
training loop.

## Learning Objectives

By the end of this module you should be able to:

- Explain the anatomy of an `nn.Module` subclass (`__init__`, `forward`,
  automatic parameter/submodule registration)
- Use `.parameters()` and `.named_parameters()` to inspect a model
- Describe the update rules behind SGD and Adam and when to prefer each
- Write the standard training loop cycle:
  `zero_grad() -> forward -> loss -> backward() -> step()`
- Build a linear regression model with `nn.Linear` and `nn.MSELoss`
- Split data into train/test sets and evaluate a model with `torch.no_grad()`
- Compare learned parameters against known ground truth to sanity-check training

## File Structure

```
day50_pytorch/
├── 01_nn_module_anatomy.py               # nn.Module structure, parameters(), SGD vs Adam
├── 02_linear_regression_training_loop.py # Full linear regression: nn.Linear + MSELoss + training loop
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
python 01_nn_module_anatomy.py
python 02_linear_regression_training_loop.py
```

Each script prints labeled, self-explanatory output to the console.

## Key Concepts Reference

| Concept                     | API / Idea                                         |
|-------------------------------|-------------------------------------------------------|
| Custom model                    | `class Net(nn.Module): def __init__/forward`            |
| Parameter access                   | `model.parameters()`, `model.named_parameters()`          |
| Loss function                         | `nn.MSELoss()`                                              |
| Optimizer (basic)                       | `torch.optim.SGD(model.parameters(), lr=...)`                 |
| Optimizer (adaptive)                       | `torch.optim.Adam(model.parameters(), lr=...)`                  |
| Training step cycle                           | `zero_grad() -> forward -> loss.backward() -> step()`              |
| Evaluation mode                                  | `model.eval()`, `torch.no_grad()`, `model.train()`                   |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Study `01_nn_module_anatomy.py`. Focus on why
   `super().__init__()` is required, how PyTorch auto-registers layers as
   parameters, and how SGD and Adam differ in their update rule.
2. **Practical coding (2h):** Work through `02_linear_regression_training_loop.py`.
   Try changing the learning rate, number of epochs, or noise level and
   observe how the training curve and recovered parameters change.
3. **Consolidation:** Write a 5–10 sentence summary in your own words plus
   the key terms below (second-brain habit).
4. **Commit:** Push this folder to GitHub, wait 5 minutes, re-read your own
   code, and improve one thing before finalizing.

## Key Terms

`nn.Module`, `nn.Linear`, `parameters()`, `named_parameters()`, `MSELoss`,
`SGD`, `Adam`, `zero_grad`, `training loop`, `model.eval()`, `torch.no_grad()`

## Reference Sources

- PyTorch official tutorials — https://pytorch.org/tutorials/
- Andrej Karpathy — micrograd (educational autograd engine)
- "Neural Networks: Zero to Hero" playlist by Andrej Karpathy

## Progress

Day 50 / 150 — Streak: 6 🔥 — Overall completion: 31%
