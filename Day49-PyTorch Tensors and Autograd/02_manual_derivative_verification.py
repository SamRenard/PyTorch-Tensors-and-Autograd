"""
Day 49 - PyTorch Tensors & Autograd
Module 2: Manual Derivative Verification + Computation Graph Diagram

Task: take a simple function, compute its derivative analytically by hand,
verify the result with autograd, and visualize the computation graph
(as an ASCII diagram, standing in for the "draw it on paper" exercise).

Function used: f(x) = (2x + 3)^2 - 5x

Analytic derivative (by hand):
    f(x)  = (2x + 3)^2 - 5x
    Let u = 2x + 3   ->  du/dx = 2
    f(x)  = u^2 - 5x
    df/dx = 2u * du/dx - 5
          = 2*(2x + 3)*2 - 5
          = 4*(2x + 3) - 5
          = 8x + 12 - 5
          = 8x + 7

Run:
    python 02_manual_derivative_verification.py
"""

import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def analytic_derivative(x: float) -> float:
    """Hand-derived derivative of f(x) = (2x + 3)^2 - 5x."""
    return 8 * x + 7


def autograd_derivative(x_value: float) -> float:
    """Same derivative computed automatically by PyTorch's autograd."""
    x = torch.tensor(x_value, requires_grad=True)
    u = 2 * x + 3     # Node 1: linear term
    v = u**2           # Node 2: square
    f = v - 5 * x        # Node 3: final combination

    f.backward()
    return x.grad.item(), f.item()


def print_graph_diagram() -> None:
    section("Computation Graph (ASCII diagram)")

    diagram = r"""
    Forward pass  (left to right):

        x ──┬────────────► [ *5 ] ──────────────┐
            │                                    │
            └──► [ *2, +3 ] ──► u ──► [ ^2 ] ──► v
                                                  │
                                                  ▼
                                          v - 5x = f(x)

    Node breakdown:
        Node 1: u = 2x + 3         (MulBackward, AddBackward)
        Node 2: v = u^2             (PowBackward)
        Node 3: f = v - 5x            (SubBackward)

    Backward pass  (right to left, chain rule applied automatically):

        df/dv = 1
        df/du = df/dv * dv/du = 1 * 2u          = 2u
        df/dx (via u branch) = df/du * du/dx    = 2u * 2 = 4u
        df/dx (via -5x branch)                  = -5
        df/dx (total)        = 4u - 5 = 4*(2x+3) - 5 = 8x + 7

    This is exactly what x.grad returns after f.backward() —
    PyTorch never uses a symbolic formula; it just multiplies local
    derivatives along the recorded graph, edge by edge, in reverse.
    """
    print(diagram)


def verify_multiple_points() -> None:
    section("Verifying Across Multiple Points")

    test_points = [-2.0, -0.5, 0.0, 1.0, 3.5]
    print(f"{'x':>6} | {'f(x)':>10} | {'analytic df/dx':>16} | {'autograd df/dx':>16} | match")
    print("-" * 68)

    for x_val in test_points:
        grad_autograd, f_val = autograd_derivative(x_val)
        grad_analytic = analytic_derivative(x_val)
        match = abs(grad_autograd - grad_analytic) < 1e-6
        print(f"{x_val:>6} | {f_val:>10.4f} | {grad_analytic:>16.4f} | "
              f"{grad_autograd:>16.4f} | {'OK' if match else 'MISMATCH'}")


if __name__ == "__main__":
    print_graph_diagram()
    verify_multiple_points()
