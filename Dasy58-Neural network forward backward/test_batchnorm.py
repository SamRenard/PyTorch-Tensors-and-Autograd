"""Day 58 — Tests for batchnorm_init.py"""
import torch
from batchnorm_init import signal_stats_across_layers, ImprovedMLP, train_improved


def test_small_init_vanishes():
    stds = signal_stats_across_layers(scale=0.01)
    assert stds[-1] < stds[0] * 0.1
    print("[PASS] too-small init causes signal to vanish across layers")


def test_xavier_init_more_stable_than_small():
    stds_small = signal_stats_across_layers(scale=0.01)
    stds_xavier = signal_stats_across_layers(scale=1 / (100 ** 0.5))
    assert stds_xavier[-1] > stds_small[-1]
    print("[PASS] Xavier-like scale preserves more signal than too-small scale")


def test_batchnorm_normalizes_activations():
    torch.manual_seed(0)
    model = ImprovedMLP()
    model.train()
    x = torch.randn(32, 64)
    out = model.net[0](x)   # Linear
    out = model.net[1](out)  # BatchNorm1d
    assert abs(out.mean().item()) < 0.1
    assert abs(out.std().item() - 1.0) < 0.2
    print("[PASS] BatchNorm1d output has ~zero mean, ~unit std")


def test_model_reaches_target_accuracy():
    _, test_acc = train_improved(epochs=300)
    assert test_acc > 0.97
    print(f"[PASS] model reaches target >97% test accuracy ({test_acc:.4f})")


if __name__ == "__main__":
    tests = [
        test_small_init_vanishes,
        test_xavier_init_more_stable_than_small,
        test_batchnorm_normalizes_activations,
        test_model_reaches_target_accuracy,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
