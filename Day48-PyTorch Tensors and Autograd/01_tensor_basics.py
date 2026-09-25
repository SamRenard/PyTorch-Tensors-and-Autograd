"""
Day 48 - PyTorch Tensors & Autograd
Module 1: Tensor Basics

Covers:
    - Tensor creation (from data, from NumPy, with factory functions)
    - Core tensor attributes (shape, dtype, device)
    - Basic arithmetic and mathematical operations
    - The NumPy <-> PyTorch bridge (shared memory semantics)

Run:
    python 01_tensor_basics.py
"""

import numpy as np
import torch


def section(title: str) -> None:
    """Print a formatted section header for readable console output."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def tensor_creation() -> None:
    section("1. Tensor Creation")

    # From raw Python data
    from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("From list:\n", from_list)

    # Factory functions
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)          # uniform [0, 1)
    randn = torch.randn(2, 3)        # standard normal
    arange = torch.arange(0, 10, 2)  # like Python's range
    eye = torch.eye(3)               # identity matrix

    print("Zeros:\n", zeros)
    print("Ones:\n", ones)
    print("Random uniform:\n", rand)
    print("Random normal:\n", randn)
    print("Arange:\n", arange)
    print("Identity:\n", eye)

    # Explicit dtype control
    float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    print("Explicit float32 dtype:", float_tensor.dtype)


def tensor_attributes() -> None:
    section("2. Tensor Attributes")

    t = torch.rand(3, 4)
    print("Shape :", t.shape)
    print("Size()", t.size())
    print("Dtype :", t.dtype)
    print("Device:", t.device)
    print("Ndim  :", t.ndim)
    print("Numel :", t.numel())


def tensor_operations() -> None:
    section("3. Tensor Operations")

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a + b        :", a + b)
    print("a - b        :", a - b)
    print("a * b (elem) :", a * b)
    print("a / b        :", a / b)
    print("torch.add    :", torch.add(a, b))

    # In-place operation (trailing underscore convention)
    c = a.clone()
    c.add_(b)
    print("In-place add_:", c)

    # Matrix multiplication
    m1 = torch.rand(2, 3)
    m2 = torch.rand(3, 4)
    print("Matmul shape :", (m1 @ m2).shape)
    print("Matmul (torch.matmul) shape:", torch.matmul(m1, m2).shape)

    # Reshaping
    x = torch.arange(12)
    print("Reshaped (3,4):\n", x.reshape(3, 4))
    print("Viewed (4,3):\n", x.view(4, 3))

    # Reduction operations
    print("Sum :", x.sum().item())
    print("Mean:", x.float().mean().item())
    print("Max :", x.max().item())
    print("Argmax:", x.argmax().item())


def numpy_bridge() -> None:
    section("4. NumPy Bridge")

    # PyTorch tensor -> NumPy array (shares memory on CPU!)
    t = torch.ones(5)
    n = t.numpy()
    print("Tensor:", t)
    print("NumPy :", n)

    t.add_(1)  # modifying the tensor also modifies the NumPy array
    print("After in-place add on tensor:")
    print("Tensor:", t)
    print("NumPy :", n)

    # NumPy array -> PyTorch tensor (also shares memory)
    n2 = np.ones(5)
    t2 = torch.from_numpy(n2)
    np.add(n2, 1, out=n2)
    print("After in-place add on NumPy array:")
    print("NumPy :", n2)
    print("Tensor:", t2)


if __name__ == "__main__":
    tensor_creation()
    tensor_attributes()
    tensor_operations()
    numpy_bridge()
