"""
Day 49 - PyTorch Tensors & Autograd
Module 1: Autograd Mechanics — Computation Graph, requires_grad, backward()

Covers:
    - How a computation graph is built dynamically as operations run
    - The role of requires_grad and leaf tensors
    - Inspecting the graph via grad_fn
    - How .backward() traverses the graph (reverse-mode autodiff)
    - Chain rule applied automatically across multiple operations

Run:
    python 01_computation_graph.py
"""

import torch


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def leaf_tensors_and_requires_grad() -> None:
    section("1. Leaf Tensors and requires_grad")

    # A "leaf" tensor is one created directly by the user, not derived
    # from another tensor via an operation.
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)

    print("x.is_leaf:", x.is_leaf)
    print("y.is_leaf:", y.is_leaf)
    print("x.grad_fn:", x.grad_fn)  # None -> leaf tensors have no grad_fn

    # Any tensor produced by an operation on tensors that require grad
    # automatically requires grad too, and is NOT a leaf.
    z = x * y
    print("z.is_leaf:", z.is_leaf)
    print("z.requires_grad:", z.requires_grad)


def inspecting_the_graph() -> None:
    section("2. Inspecting the Computation Graph via grad_fn")

    x = torch.tensor(2.0, requires_grad=True)

    a = x * 3          # MulBackward
    b = a + 1           # AddBackward
    c = b**2             # PowBackward

    print("a.grad_fn:", a.grad_fn)
    print("b.grad_fn:", b.grad_fn)
    print("c.grad_fn:", c.grad_fn)

    # next_functions links each node to its parents in the graph,
    # forming the path backward() will walk.
    print("\nc.grad_fn.next_functions:", c.grad_fn.next_functions)
    print("b.grad_fn.next_functions:", b.grad_fn.next_functions)


def backward_mechanics() -> None:
    section("3. How backward() Walks the Graph")

    # f(x) = (3x + 1)^2
    # Chain rule: df/dx = 2*(3x+1) * 3 = 6*(3x+1)
    x = torch.tensor(2.0, requires_grad=True)

    a = 3 * x       # a = 6
    b = a + 1        # b = 7
    c = b**2          # c = 49

    c.backward()  # walks c -> b -> a -> x, multiplying local derivatives

    expected = 6 * (3 * x.item() + 1)
    print(f"x = {x.item()}, c = {c.item()}")
    print(f"Expected df/dx = {expected}")
    print(f"Autograd df/dx = {x.grad.item()}")


def multi_variable_graph() -> None:
    section("4. Multi-Variable Computation Graph")

    # f(x, y) = x^2 * y + y^3
    # df/dx = 2xy
    # df/dy = x^2 + 3y^2
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    f = x**2 * y + y**3
    f.backward()

    print(f"x = {x.item()}, y = {y.item()}, f = {f.item()}")
    print(f"df/dx expected = {2 * x.item() * y.item()}, autograd = {x.grad.item()}")
    print(f"df/dy expected = {x.item()**2 + 3 * y.item()**2}, autograd = {y.grad.item()}")


def graph_is_freed_after_backward() -> None:
    section("5. The Graph Is Freed After backward() by Default")

    x = torch.tensor(1.0, requires_grad=True)
    y = x**2
    y.backward()

    try:
        y.backward()  # calling backward() again fails: buffers were freed
    except RuntimeError as e:
        print("Second backward() call raised RuntimeError, as expected:")
        print(" ->", str(e)[:120] + "...")

    # If you need to call backward() multiple times, use retain_graph=True
    x2 = torch.tensor(1.0, requires_grad=True)
    y2 = x2**2
    y2.backward(retain_graph=True)
    y2.backward()  # works now, gradients accumulate
    print("\nWith retain_graph=True, second backward() succeeds.")
    print("Accumulated grad:", x2.grad.item())


if __name__ == "__main__":
    leaf_tensors_and_requires_grad()
    inspecting_the_graph()
    backward_mechanics()
    multi_variable_graph()
    graph_is_freed_after_backward()
