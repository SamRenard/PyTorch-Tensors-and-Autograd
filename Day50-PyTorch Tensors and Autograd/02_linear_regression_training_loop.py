"""
Day 50 - PyTorch Tensors & Autograd
Module 2: Linear Regression with PyTorch (nn.Linear + MSELoss + Training Loop)

Covers:
    - Building a linear regression model with nn.Linear
    - Generating a synthetic dataset with a known ground-truth relationship
    - Writing a complete, idiomatic PyTorch training loop
    - Tracking loss over epochs and recovering the learned parameters
    - Evaluating the trained model on unseen data

Run:
    python 02_linear_regression_training_loop.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def generate_synthetic_data(n_samples: int = 200, n_features: int = 3, noise_std: float = 0.1):
    """
    Generate data from a known linear relationship:
        y = X @ true_weights + true_bias + noise

    Returns the data plus the ground-truth parameters, so we can later
    check how closely the trained model recovers them.
    """
    torch.manual_seed(42)

    true_weights = torch.tensor([2.5, -1.3, 0.7])
    true_bias = torch.tensor(4.0)

    X = torch.randn(n_samples, n_features)
    noise = torch.randn(n_samples) * noise_std
    y = X @ true_weights + true_bias + noise
    y = y.unsqueeze(1)  # shape (n_samples, 1) to match model output

    return X, y, true_weights, true_bias


def train_test_split(X: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def build_model(n_features: int) -> nn.Module:
    # A single nn.Linear layer IS linear regression: y = X @ W.T + b
    return nn.Linear(in_features=n_features, out_features=1)


def train(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
          epochs: int = 200, lr: float = 0.05) -> list:
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()             # 1. reset gradients from last step
        predictions = model(X_train)       # 2. forward pass
        loss = loss_fn(predictions, y_train)  # 3. compute MSE loss
        loss.backward()                          # 4. compute gradients
        optimizer.step()                          # 5. update weights and bias

        history.append(loss.item())
        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>4}/{epochs} | Loss: {loss.item():.6f}")

    return history


def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    loss_fn = nn.MSELoss()
    # Evaluation should not track gradients -- saves memory and computation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = loss_fn(predictions, y_test)
    model.train()
    return test_loss.item()


def compare_learned_vs_true(model: nn.Module, true_weights: torch.Tensor, true_bias: torch.Tensor) -> None:
    section("Learned Parameters vs. Ground Truth")

    learned_weights = model.weight.detach().squeeze()
    learned_bias = model.bias.detach().squeeze()

    print("True weights   :", true_weights.tolist())
    print("Learned weights:", [round(w, 3) for w in learned_weights.tolist()])
    print()
    print("True bias   :", true_bias.item())
    print("Learned bias:", round(learned_bias.item(), 3))


def main() -> None:
    section("1. Synthetic Dataset")
    X, y, true_weights, true_bias = generate_synthetic_data()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Feature dimensionality: {X_train.shape[1]}")

    section("2. Model Definition")
    model = build_model(n_features=X.shape[1])
    print(model)

    section("3. Training Loop")
    history = train(model, X_train, y_train, epochs=200, lr=0.05)
    print(f"\nLoss decreased from {history[0]:.6f} to {history[-1]:.6f}")

    section("4. Evaluation on Held-Out Test Set")
    test_loss = evaluate(model, X_test, y_test)
    print(f"Test MSE: {test_loss:.6f}")

    compare_learned_vs_true(model, true_weights, true_bias)


if __name__ == "__main__":
    main()
