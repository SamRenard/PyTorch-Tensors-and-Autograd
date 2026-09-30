# Day 52 - PyTorch Tensors & Autograd (micrograd from scratch)

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 2: Mathematics & Deep Learning).

This project rebuilds the core of a deep learning framework from first principles: a scalar-valued
**reverse-mode automatic differentiation engine** and a small **neural network library** on top of it,
inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It is paired with a set
of annotated **PyTorch tensor and autograd** examples that show how the same ideas scale up to
n-dimensional arrays.

## Goals

- Understand backpropagation by implementing it, not just by calling `loss.backward()`.
- Verify the implementation rigorously (hand-derived values, finite differences, and PyTorch parity).
- Bridge the scalar engine to PyTorch's tensor-based autograd.

## Features

- `Value` class with `+`, `-`, `*`, `/`, `**`, `tanh`, `relu`, `sigmoid`, `exp`, `log` and reflected operators.
- Correct gradient accumulation for nodes that are reused in a graph.
- Iterative topological sort, so very deep graphs do not hit Python's recursion limit.
- Numerically stable `sigmoid`.
- `Neuron`, `Layer` and `MLP` with reproducible seeding and `zero_grad()`.
- Test suite covering analytic gradients, finite-difference checks and PyTorch comparison.
- Continuous integration via GitHub Actions.

## Project structure

```
.
├── micrograd/
│   ├── __init__.py
│   ├── engine.py            # Value class + backward()
│   └── nn.py                # Neuron, Layer, MLP
├── examples/
│   ├── train_xor.py         # Train an MLP on XOR with the custom engine
│   └── pytorch_basics.py    # Tensors, broadcasting, autograd in PyTorch
├── tests/
│   └── test_engine.py
├── notes/
│   └── day52_summary.md     # Concept notes, formulas, glossary
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
```

PyTorch is only required for `examples/pytorch_basics.py` and the parity test. The `micrograd`
package itself has no dependencies beyond the Python standard library.

## Quick start

```python
from micrograd import Value

x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)

out = (x1 * w1 + x2 * w2 + b).tanh()
out.backward()

print(out.data)   # 0.7071...
print(x1.grad)    # -1.5
print(w1.grad)    #  1.0
```

### Train a network

```python
from micrograd import MLP

model = MLP(2, [4, 4, 1], seed=0)

for _ in range(200):
    loss = sum((model(x) - y) ** 2 for x, y in zip(xs, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

Run the complete example:

```bash
python examples/train_xor.py
```

Output:

```
Parameters: 37
epoch   0 | loss 4.074818
epoch  25 | loss 2.132659
epoch  50 | loss 0.027832
epoch  75 | loss 0.000072
epoch 100 | loss 0.000000
...
Predictions after training:
  input=[0.0, 0.0] target=-1 prediction=-1.0000
  input=[0.0, 1.0] target=+1 prediction=+1.0000
  input=[1.0, 0.0] target=+1 prediction=+1.0000
  input=[1.0, 1.0] target=-1 prediction=-1.0000
```

### PyTorch counterpart

```bash
python examples/pytorch_basics.py
```

Covers tensor creation, shape manipulation, broadcasting, matrix multiplication, `requires_grad`,
gradient accumulation and zeroing, `no_grad`, `detach`, and a linear regression fitted with manual
gradient descent.

## How it works

1. **Forward pass.** Every operation creates a new `Value` that stores its result, its parent nodes
   and a closure (`_backward`) that knows the operation's *local* derivative.
2. **Topological ordering.** `backward()` orders the graph so every node is processed only after all
   nodes that depend on it.
3. **Backward pass.** The output gradient is seeded with `1.0`, then each node applies the chain rule:

   ```
   child.grad += local_derivative * node.grad
   ```

   `+=` (not `=`) matters: if a value is used twice, its gradients from both uses must be summed.

| Operation | Local derivative |
|-----------|------------------|
| `a + b` | `1`, `1` |
| `a * b` | `b`, `a` |
| `a ** n` | `n * a^(n-1)` |
| `tanh(a)` | `1 - tanh(a)^2` |
| `relu(a)` | `1` if `a > 0` else `0` |
| `sigmoid(a)` | `s * (1 - s)` |
| `exp(a)` | `exp(a)` |
| `log(a)` | `1 / a` |

## Testing

```bash
python -m pytest -v
```

| Test | What it verifies |
|------|------------------|
| Add / multiply | Hand-derived gradients |
| Reused node | Gradient accumulation (`a + a`) |
| Reflected operators | `2 + a`, `3 * a`, `8 / a`, ... |
| Unary ops | Finite-difference gradient checks for `tanh`, `relu`, `sigmoid`, `exp`, `log` |
| Sigmoid stability | No overflow for large positive/negative inputs |
| Reference expression | Matches the canonical lecture example |
| Deep graph | 5,000-node chain without recursion errors |
| MLP training | Loss decreases by more than 90% on XOR |
| PyTorch parity | Forward values and gradients match `torch` in float64 (skipped if torch is missing) |

## Key takeaways

- Backpropagation is the chain rule applied over a computation graph.
- Each node needs only its local derivative; everything else is bookkeeping.
- Gradients accumulate, so they must be zeroed between optimisation steps.
- PyTorch performs the same computation with tensors and optimised kernels instead of scalars.

More detail, formulas and a glossary are in [`notes/day52_summary.md`](notes/day52_summary.md).

## Limitations

- Scalar only, so it is educational and far too slow for real workloads.
- No batching, no optimisers beyond manual SGD, no GPU support.
- Exponents in `__pow__` must be plain `int` or `float`.

## Roadmap

- [ ] Computation graph visualisation with Graphviz
- [ ] Additional optimisers (momentum, Adam)
- [ ] Vectorised tensor version built on NumPy

## References

- Andrej Karpathy - [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Andrej Karpathy - [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- Andrej Karpathy - [micrograd repository](https://github.com/karpathy/micrograd)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

## License

Released under the [MIT License](LICENSE).
