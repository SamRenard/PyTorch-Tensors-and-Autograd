# Day 55 - Softmax & Cross-Entropy from Scratch (and why not MSE)

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 2: Mathematics & Deep Learning).

Softmax, cross-entropy, mean squared error and binary cross-entropy implemented from scratch in NumPy,
with analytic gradients and an API deliberately mirroring PyTorch (`reduction`, `label_smoothing`,
`forward`/`backward`) so results can be compared one-to-one.

## Goals

- Understand the logic of softmax and why cross-entropy suits classification while MSE suits regression.
- Implement numerically stable softmax and cross-entropy, with gradients.
- Verify the results against independent references, including PyTorch.

## Features

- `softmax`, `log_softmax`, `logsumexp`, `sigmoid`, all stable for extreme inputs.
- `cross_entropy` with integer targets, `reduction` (`mean` / `sum` / `none`) and `label_smoothing`.
- `mse_loss` and `bce_with_logits`, plus an analytic gradient for every loss.
- Loss objects (`CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`) with `__call__` and `backward()`.
- Strict input validation: float targets, out-of-range labels and shape mismatches raise clear errors.
- Finite-difference gradient checking helpers.
- Continuous integration via GitHub Actions.

## Project structure

```
.
├── losses/
│   ├── __init__.py
│   ├── functional.py        # stateless losses, gradients, stable softmax
│   ├── modules.py           # CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
│   └── gradcheck.py         # numerical gradient and relative error
├── examples/
│   ├── softmax_ce_walkthrough.py     # step-by-step computation + stability demo
│   ├── mse_vs_cross_entropy.py       # why classification does not use MSE
│   ├── train_softmax_regression.py   # train a classifier on 8x8 digits
│   └── compare_with_pytorch.py       # prints max difference vs torch (needs torch)
├── tests/
│   └── test_losses.py
├── notes/
│   └── day55_summary.md     # short concept notes (Azerbaijani)
├── .github/workflows/ci.yml
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch                # optional: enables the PyTorch parity checks
```

## Quick start

```python
import numpy as np
from losses import CrossEntropyLoss, cross_entropy, softmax

logits = np.array([[2.0, 1.0, 0.1]])
target = np.array([0])                     # integer class index, NOT one-hot

print(softmax(logits))                     # [[0.659 0.24243 0.09857]]
print(cross_entropy(logits, target))       # 0.41703

ce = CrossEntropyLoss(label_smoothing=0.1)
loss = ce(logits, target)
dlogits = ce.backward()                    # (p - target_distribution) / N
```

Typical training step for a linear classifier:

```python
ce(X @ W + b, y)
d = ce.backward()
W -= lr * (X.T @ d)
b -= lr * d.sum(axis=0)
```

## The math

**Softmax** turns logits into probabilities:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

**Cross-entropy** is the negative log-probability of the true class. It simplifies to a log-sum-exp form
that never evaluates `log(0)`:

```
CE = -log p[y] = logsumexp(z) - z[y]
```

**Gradient.** Softmax and the log cancel, leaving the difference between prediction and target:

```
dCE/dz = p - one_hot(y)           (divide by N for the "mean" reduction)
```

With label smoothing `e`, the target distribution is `(1 - e) * one_hot + e / C`.

**MSE and BCE**

```
MSE = mean((pred - target)^2)               dMSE/dpred = 2 (pred - target) / size
BCE = max(x, 0) - x*y + log(1 + exp(-|x|))   dBCE/dx   = sigmoid(x) - y
```

**Numerical stability.** Subtracting the maximum logit before exponentiating leaves softmax unchanged but
prevents overflow:

```
naive  exp(z)/sum(exp(z)) for z = [1000, 1001, 1002]  ->  [nan nan nan]
stable softmax                                         ->  [0.09003 0.24473 0.66524]
```

## Why not MSE for classification?

With a sigmoid output `p`, the MSE gradient with respect to the logit carries an extra factor
`p (1 - p)` that vanishes exactly when the model is confidently wrong. Cross-entropy cancels that factor.
Gradient with respect to the logit when the true label is 1:

```
    z   p=sigmoid(z)     MSE grad     BCE grad
 -8.0       0.000335    -0.000670    -0.999665
 -6.0       0.002473    -0.004921    -0.997527
 -4.0       0.017986    -0.034690    -0.982014
 -2.0       0.119203    -0.184956    -0.880797
  0.0       0.500000    -0.250000    -0.500000
  2.0       0.880797    -0.025031    -0.119203
```

Starting one logistic neuron at `p = 0.0025`, gradient descent with learning rate 1.0 needs:

```
MSE  -> 249 steps to reach p >= 0.9
BCE  ->  17 steps
```

MSE remains the right choice for regression, where the target is a continuous quantity.

## Examples

```bash
python examples/softmax_ce_walkthrough.py
python examples/mse_vs_cross_entropy.py
python examples/train_softmax_regression.py
python examples/compare_with_pytorch.py        # requires torch
```

Softmax regression on the 8x8 digits dataset bundled with scikit-learn (64 features, 10 classes,
80/20 split, mini-batch 64, learning rate 0.5):

```
train 1437 | test 360 | initial loss 2.3026 (ln 10 = 2.3026)
epoch  1 | train loss 1.0304 | test acc 90.3%
epoch 10 | train loss 0.2582 | test acc 95.3%
epoch 30 | train loss 0.1506 | test acc 97.8%
Final: train accuracy 96.9% | test accuracy 97.8%
```

The initial loss equals `ln(10)`, exactly what a model that assigns equal probability to all 10 classes
should score.

## Verification

```bash
python -m pytest -v
```

Every function is checked against independent references:

| Check | Reference |
|-------|-----------|
| `softmax`, `log_softmax`, `logsumexp` | SciPy (`scipy.special`) |
| `cross_entropy`, `bce_with_logits` | scikit-learn `log_loss` |
| Label smoothing | Explicit target-distribution formula |
| All gradients (every reduction, with and without smoothing) | Central finite differences |
| Hand-computed case | `[2, 1, 0.1]`, target 0 -> `0.41703` |
| Extreme inputs | Logits around `+-1e4` stay finite |
| Input validation | Float targets, bad labels, shape mismatch, invalid reduction |
| Loss objects | Match the functional API, incl. per-sample upstream gradients |
| End to end | Softmax regression reaches over 95% accuracy |
| **PyTorch parity** | `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy_with_logits`: values and autograd gradients, all reductions (runs when `torch` is installed) |

`examples/compare_with_pytorch.py` prints the maximum absolute difference per loss and reduction;
values near `1e-15` mean agreement to floating-point precision.

## Common pitfalls

- **Double softmax.** `CrossEntropyLoss` expects raw logits; applying softmax first silently trains worse.
- **One-hot targets.** Targets are integer class indices here, as in PyTorch's default.
- **MSE `mean`** averages over every element, not just over the batch.
- **Broadcasting.** `mse_loss` rejects shape mismatches such as `(N, 1)` vs `(N,)` rather than broadcasting silently to `(N, N)`.

## Key takeaways

- Softmax + cross-entropy yields the gradient `p - y`: the error signal is simply prediction minus target.
- Cross-entropy punishes confident mistakes heavily; MSE saturates exactly where learning is needed most.
- Always compute `logsumexp` with the max-subtraction trick.
- Prove a from-scratch implementation correct against independent references, not by inspection.

Short concept notes (in Azerbaijani): [`notes/day55_summary.md`](notes/day55_summary.md).

## Limitations

- Class-index targets only; no soft-label (probability) targets, class weights or `ignore_index`.
- No focal loss, hinge loss or Huber loss.
- CPU / NumPy only.

## Roadmap

- [ ] Class weights and `ignore_index`
- [ ] Soft-label cross-entropy
- [ ] Huber / smooth-L1 and focal loss
- [ ] Plug these losses into the Day 54 `MLP`

## References

- 3Blue1Brown - [Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
- Andrej Karpathy - [Neural Networks: Zero to Hero (makemore)](https://karpathy.ai/zero-to-hero.html)
- [`torch.nn.functional.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [`torch.nn.functional.binary_cross_entropy_with_logits`](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)
- [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

## License

Released under the [MIT License](LICENSE).
