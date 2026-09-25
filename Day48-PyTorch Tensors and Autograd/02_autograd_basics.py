"""
Day 48 - PyTorch Tensors & Autograd
Module 2: Autograd Fundamentals

Covers:
    - requires_grad and the computation graph
    - Automatic differentiation with .backward()
    - Gradient accumulation and zeroing
    - Disabling gradient tracking (torch.no_grad, detach)

Run:
    python 02_autograd_basics.py
"""

import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def simple_gradient() -> None:
    section("1. Simple Scalar Gradient")

    # y = x^2 + 3x + 1, dy/dx = 2x + 3
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x + 1

    y.backward()
    print(f"x = {x.item()}")
    print(f"y = {y.item()}")
    print(f"dy/dx (analytic 2x+3 = {2 * x.item() + 3}) -> autograd: {x.grad.item()}")


def vector_gradient() -> None:
    section("2. Vector Input with Gradient Argument")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y * y * 3  # z = 3 * (2x)^2 = 12x^2

    # backward() on a non-scalar tensor needs a matching-shape gradient
    grad_output = torch.tensor([1.0, 1.0, 1.0])
    z.backward(grad_output)

    print("x    :", x)
    print("z    :", z)
    print("dz/dx (analytic 24x):", 24 * x.detach())
    print("dz/dx (autograd)    :", x.grad)


def gradient_accumulation() -> None:
    section("3. Gradient Accumulation and Zeroing")

    x = torch.tensor(1.0, requires_grad=True)

    for step in range(3):
        y = x**2
        y.backward()
        print(f"step {step}: grad = {x.grad.item()} (accumulates without zero_())")

    # Gradients must be reset manually before the next optimization step
    x.grad.zero_()
    y = x**2
    y.backward()
    print(f"after zero_(): grad = {x.grad.item()}")


def disabling_gradient_tracking() -> None:
    section("4. Disabling Gradient Tracking")

    x = torch.tensor(2.0, requires_grad=True)

    # Option 1: torch.no_grad() context manager (used for inference)
    with torch.no_grad():
        y = x * 2
        print("Inside no_grad, y.requires_grad:", y.requires_grad)

    # Option 2: .detach() creates a new tensor sharing data but no graph
    z = x.detach()
    print("Detached tensor requires_grad:", z.requires_grad)

    # Option 3: tensor.requires_grad_(False) toggles tracking in place
    w = torch.tensor(3.0, requires_grad=True)
    w.requires_grad_(False)
    print("After requires_grad_(False):", w.requires_grad)


def mini_linear_regression_step() -> None:
    section("5. One Manual Gradient Descent Step (Mini Example)")

    # y = w * x + b, single sample, MSE loss
    x = torch.tensor(4.0)
    y_true = torch.tensor(9.0)

    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.01

    y_pred = w * x + b
    loss = (y_pred - y_true) ** 2

    loss.backward()
    print(f"Before update -> w={w.item():.4f}, b={b.item():.4f}, loss={loss.item():.4f}")

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()
    print(f"After update  -> w={w.item():.4f}, b={b.item():.4f}")


if __name__ == "__main__":
    simple_gradient()
    vector_gradient()
    gradient_accumulation()
    disabling_gradient_tracking()
    mini_linear_regression_step()
