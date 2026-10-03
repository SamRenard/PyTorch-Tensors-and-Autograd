# Day 58 — Batch Normalization & Weight Initialization

Part of **NIZAM AI · 150-Day AI Engineering Protocol** · Month 2 · Block Day 6/6

## 📌 Concept

Bad weight init makes signal explode or vanish across layers. **Xavier/He**
scale weights by layer size to keep signal stable. **BatchNorm** normalizes
each layer's input during training (mean 0, std 1), speeding up training and
reducing sensitivity to init.

## 📂 Files

| File | Description |
|---|---|
| `batchnorm_init.py` | Demo: signal std across 10 layers at 3 init scales; `ImprovedMLP` with He init + BatchNorm, trained to >97% test accuracy |
| `test_batchnorm.py` | 4 automated tests |

## ▶️ Usage

```bash
pip install torch scikit-learn
python3 batchnorm_init.py
python3 test_batchnorm.py
```

## ✅ Results

```
4/4 checks passed
Final test accuracy: 0.9749 (target: >97%)
```

(Same sklearn `digits` dataset as Day 57 — see that day's README for why.)

## 📚 Sources

- Karpathy — makemore series
- 3Blue1Brown — Neural Networks

---
*NIZAM AI Day 58/150*
