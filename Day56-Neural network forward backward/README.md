# Day 56 - Character-Level Bigram Language Model on Azerbaijani Names

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 2: Mathematics & Deep Learning).

A re-implementation of Andrej Karpathy's *makemore* part 1 (the bigram language model, the seed of every LLM)
in pure NumPy, trained on **Azerbaijani given names**. Two equivalent models are built and compared:
a **counting** model and a **neural** model trained with gradient descent.

## Goals

- Understand a language model as `P(next character | context)`.
- Build the counting model, evaluate it with negative log-likelihood, and fix zero counts with smoothing.
- Build the neural version and show that it converges to the same solution as counting.
- Handle Azerbaijani text correctly (`ə ı ş ğ ö ü ç`, dotted vs dotless `i`).

## Features

- `az_lower`: Azerbaijani-aware lowercasing (Python's default `"İ".lower()` yields two code points).
- `load_names`: normalising, de-duplicating, validating loader; `split_names`: deterministic train/dev/test split.
- `CountingBigram`: count matrix, add-alpha smoothing, NLL, top bigrams, sampling.
- `NeuralBigram`: `W[x]` + softmax, analytic gradient with L2 regularisation, gradient descent.
- `generate`: name sampling with guaranteed termination (`max_len`) and `min_len`.
- 23 tests, including a finite-difference gradient check and a "neural converges to counting" test.
- Continuous integration via GitHub Actions.

## Project structure

```
.
├── bigram/
│   ├── __init__.py
│   ├── data.py             # az_lower, load_names, split_names, Vocab, bigrams
│   ├── counting.py         # CountingBigram
│   ├── neural.py           # NeuralBigram
│   ├── generation.py       # name sampling
│   └── functional.py       # stable softmax / log_softmax
├── data/
│   ├── names_az.txt        # 388 Azerbaijani names (see data/README.md)
│   └── README.md
├── examples/
│   ├── count_bigram.py         # smoothing sweep, evaluation, sampling
│   ├── neural_bigram.py        # training, regularisation vs smoothing
│   └── plot_bigram_matrix.py   # heatmap -> docs/bigram_matrix.png
├── docs/bigram_matrix.png
├── tests/test_bigram.py
├── notes/day56_summary.md      # short concept notes (Azerbaijani)
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

## Quick start

```python
import numpy as np
from bigram import CountingBigram, Vocab, load_names, split_names

names = load_names("data/names_az.txt")
train, dev, test = split_names(names, seed=42)
vocab = Vocab.from_names(names)

model = CountingBigram(vocab).fit(train)
print(model.nll(test, alpha=0.3))                       # 2.5329
print(model.sample(5, np.random.default_rng(0), alpha=0.3, min_len=2))
```

Use your own dataset (one name per line):

```bash
python examples/count_bigram.py path/to/my_names.txt
```

## The bigram bitmap

`counts[i, j]` is how often character `j` follows character `i`; `.` marks the start and end of a name.
Row `.` shows how names begin; column `.` shows how they end (many end in `ə`, `r`, `n`).

![Bigram counts](docs/bigram_matrix.png)

## The math

```
P(j | i)  = (N[i,j] + alpha) / (sum_k N[i,k] + alpha * V)        counting model
NLL       = -(1/n) * sum log P(next | current)                    perplexity = exp(NLL)
logits    = W[x]            (= one_hot(x) @ W)                    neural model
P         = softmax(logits)
loss      = NLL + lambda * mean(W^2)
dlogits   = (P - one_hot(y)) / n          dW = scatter-add of dlogits into the rows used
```

The neural model is just a lookup table of logits: with no regularisation, gradient descent drives
`softmax(W)` towards the normalised count matrix. L2 regularisation pulls `W` towards zero (a uniform
distribution), which does the same job as add-alpha smoothing.

## Results

Dataset: 388 names, vocabulary of 33 symbols (32 letters + `.`), split 310 / 38 / 40 (train / dev / test).
Metric: average negative log-likelihood per bigram (lower is better).

```
python examples/count_bigram.py
```

```
Uniform baseline NLL = ln(33) = 3.4965

   alpha     train       dev
     0.0    2.2929       inf
    0.01    2.2965    2.7442
     0.1    2.3253    2.6180
     0.3    2.3772    2.5998   <- lowest dev loss
     1.0    2.5047    2.6602
     2.0    2.6242    2.7480

Test NLL = 2.5329 | perplexity = 12.59 (uniform would be 33)
```

With `alpha = 0` the dev loss is infinite because some dev bigram never occurs in training, which is
exactly why smoothing is needed.

```
python examples/neural_bigram.py
```

```
after    1 steps | train loss 3.5165
after   50 steps | train loss 2.3895
after  200 steps | train loss 2.3157
after  800 steps | train loss 2.2979
counting optimum (alpha=0): train loss 2.2929

reg=0.0    train 2.2979  dev 2.7119
reg=0.03   train 2.3272  dev 2.6105
reg=0.1    train 2.3775  dev 2.5981   <- lowest dev loss
reg=0.3    train 2.4729  dev 2.6397
best counting model: alpha=0.3  dev 2.5998
best neural model:   reg=0.1    dev 2.5981

Neural test NLL = 2.5241 | counting test NLL = 2.5329
```

Findings:

- The neural model slowly approaches, and never beats, the counting optimum on training data
  (`2.2979` vs `2.2929`); counting is the maximum-likelihood solution.
- Tuned L2 regularisation and tuned smoothing reach the same quality (`2.598` vs `2.600` on dev).
- The dev and test sets are tiny (38 and 40 names), so differences in the third decimal are noise.

### Sampled names

```
counting:  gımimidərəhiq, bigkanan, şad, bə, löüm, evil, ovət, midə, kir, mel, fam, çəm ...
neural:    gülnan, janan, şad, bə, evil, midə, kir, kiham, ar, mel, fam, öğt ...
```

Honest assessment: some samples look name-like (`gülnan`, `janan`, `kiham`, `evil`, `midə`), but many are
not (`ehüayşıvətğmrasgüm`), and generated names are longer than real ones (mean 7.2 vs 5.6 characters).
Only 0.1% of 2,000 samples are real dataset names. That is the expected ceiling of a model that sees a
single previous character, and it motivates longer contexts (trigram, MLP, and later transformers).

## Testing

```bash
python -m pytest -v
```

| Test | What it verifies |
|------|------------------|
| `az_lower` | `İ -> i`, `I -> ı`, `Ə -> ə`; result length is preserved |
| Pitfall documentation | `"İ".lower()` really has two code points in Python |
| Loader | Normalises, de-duplicates, skips comments, rejects spaces/digits and empty files |
| Bundled dataset | Unique, only Azerbaijani letters, sensible lengths |
| Split | Deterministic, disjoint, covers every name |
| Vocabulary and bigrams | Round trip; `.` padding at both ends |
| Counting model | Hand-computed counts, probabilities and NLL (`ln 2 / 3`) |
| Smoothing | Rows sum to 1; unseen bigram gives `inf` at alpha 0, finite otherwise; huge alpha gives `ln V` |
| Neural model | `W[x]` equals `one_hot @ W`; gradient matches finite differences (relative error < 1e-7) |
| Convergence | Neural model matches the counting model's NLL within 0.02 |
| Optimality | Unregularised training loss never goes below the counting optimum |
| Regularisation | L2 shrinks the weights |
| Sampling | Reproducible, stays in vocabulary, terminates even if `.` is unreachable, honours `min_len` |

## Key takeaways

- A language model is a conditional probability table; training means estimating it.
- Counting and gradient descent find the same answer: the neural network is a learned lookup table.
- Evaluation needs held-out data; zero counts make an unsmoothed model's loss infinite.
- Regularisation and smoothing are two views of the same idea.
- Text handling is part of the model: a wrong lowercase rule silently corrupts the vocabulary.

Short concept notes (in Azerbaijani): [`notes/day56_summary.md`](notes/day56_summary.md).

## Limitations

- One character of context only, so names are often implausible.
- The bundled dataset is small and hand-compiled (see `data/README.md`); results will change with a larger one.
- Full-batch gradient descent, NumPy only.

## Roadmap

- [ ] Trigram model and a larger context window
- [ ] MLP language model (Bengio et al.) with character embeddings
- [ ] Larger, verified Azerbaijani names dataset
- [ ] Port the neural model to PyTorch and compare

## References

- Andrej Karpathy - [makemore](https://github.com/karpathy/makemore) and the lecture *The spelled-out intro to language modeling: building makemore*
- Andrej Karpathy - [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- 3Blue1Brown - [Neural Networks](https://www.3blue1brown.com/topics/neural-networks)

## License

Released under the [MIT License](LICENSE).
