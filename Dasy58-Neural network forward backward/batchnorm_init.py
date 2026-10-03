"""
Day 58 — Batch Normalization & Weight Initialization
NIZAM AI · 150-Day AI Engineering Protocol

Goal: show why weight init matters (signal exploding/vanishing across
layers), then show batch norm fixing it -- then apply both to improve
yesterday's MNIST-digits MLP toward >97% test accuracy.
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Part 1: why initialization matters
# ---------------------------------------------------------------------
def signal_stats_across_layers(scale: float, n_layers: int = 10, width: int = 100):
    """Pass a unit-variance input through n_layers of Linear(no bias)+tanh,
    with weights ~ N(0, scale^2). Returns the std of the activations at
    each layer, to show whether the signal explodes or vanishes."""
    x = torch.randn(200, width)
    stds = [x.std().item()]
    for _ in range(n_layers):
        w = torch.randn(width, width) * scale
        x = torch.tanh(x @ w)
        stds.append(x.std().item())
    return stds


def demo_init_scales():
    print("=" * 60)
    print("Signal std across 10 layers, by weight init scale")
    print("=" * 60)
    for scale, label in [(1.0, "too large"), (0.01, "too small"), (1 / (100 ** 0.5), "Xavier-like (1/sqrt(fan_in))")]:
        stds = signal_stats_across_layers(scale)
        print(f"{label:28s} scale={scale:.4f}  stds={[f'{s:.3f}' for s in stds]}")
    print("-> too-large weights push pre-activations deep into tanh's saturated "
          "zone (output pinned near +-1, gradients near zero there); too-small "
          "weights shrink activations toward 0 layer after layer -- vanishing "
          "signal. Xavier scale (1/sqrt(fan_in)) keeps things closer to stable, "
          "though even it decays somewhat without normalization -- which is "
          "exactly the gap batch norm closes.")


# ---------------------------------------------------------------------
# Part 2: improved MLP -- Xavier init + BatchNorm, applied to Day 57's model
# ---------------------------------------------------------------------
class ImprovedMLP(nn.Module):
    def __init__(self, in_features=64, hidden=256, n_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # He init, pairs with ReLU
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def load_data(train_size=150):
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    t = torch.from_numpy
    return t(X_train), t(y_train), t(X_val), t(y_val), t(X_test), t(y_test)


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def train_improved(epochs=300, lr=1e-3, weight_decay=1e-4, train_size=1000):
    torch.manual_seed(0)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_size=train_size)
    model = ImprovedMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"val_acc={accuracy(model, X_val, y_val):.4f}")

    test_acc = accuracy(model, X_test, y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    return model, test_acc


if __name__ == "__main__":
    demo_init_scales()
    print("\n" + "=" * 60)
    print("Improved MLP: He init + BatchNorm, target >97% test accuracy")
    print("=" * 60)
    train_improved()
