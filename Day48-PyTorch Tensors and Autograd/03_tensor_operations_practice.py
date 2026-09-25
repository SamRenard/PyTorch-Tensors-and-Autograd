"""
Day 48 - PyTorch Tensors & Autograd
Module 3: 20 Tensor Operations Drill + CPU/GPU Device Management

This script exercises 20 distinct, commonly-used tensor operations and
demonstrates proper device handling (CPU vs GPU) so the same code runs
correctly regardless of what hardware is available.

Run:
    python 03_tensor_operations_practice.py
"""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU if present, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available -> using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available -> using CPU")
    return device


def device_management_demo(device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("Device Management")
    print("=" * 60)

    # Create a tensor and move it to the selected device
    t = torch.rand(3, 3)
    print("Original device:", t.device)

    t_on_device = t.to(device)
    print("Moved to      :", t_on_device.device)

    # Operations between tensors require matching devices
    a = torch.rand(3, 3, device=device)
    b = torch.rand(3, 3, device=device)
    c = a + b
    print("Result device :", c.device)

    # Always move back to CPU before converting to NumPy or printing large data
    c_cpu = c.to("cpu")
    print("Back on CPU   :", c_cpu.device)


def twenty_operations(device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("20 Core Tensor Operations")
    print("=" * 60)

    x = torch.arange(1, 13, dtype=torch.float32, device=device).reshape(3, 4)
    y = torch.rand(3, 4, device=device)
    print("x:\n", x)
    print("y:\n", y)

    # 1. Addition
    print("\n1. Addition:\n", x + y)

    # 2. Subtraction
    print("\n2. Subtraction:\n", x - y)

    # 3. Element-wise multiplication
    print("\n3. Element-wise multiplication:\n", x * y)

    # 4. Element-wise division
    print("\n4. Element-wise division:\n", x / (y + 1e-6))

    # 5. Matrix multiplication
    print("\n5. Matrix multiplication:\n", x @ y.T)

    # 6. Transpose
    print("\n6. Transpose:\n", x.T)

    # 7. Reshape
    print("\n7. Reshape to (4, 3):\n", x.reshape(4, 3))

    # 8. Squeeze / Unsqueeze
    z = torch.rand(1, 4, device=device)
    print("\n8a. Unsqueeze:\n", z.unsqueeze(0).shape)
    print("8b. Squeeze:\n", z.squeeze(0).shape)

    # 9. Concatenation
    print("\n9. Concatenation (dim=0):\n", torch.cat([x, x], dim=0))

    # 10. Stacking
    print("\n10. Stack (new dim):\n", torch.stack([x, x]).shape)

    # 11. Indexing / Slicing
    print("\n11. Slice x[:, 1:3]:\n", x[:, 1:3])

    # 12. Boolean masking
    mask = x > 6
    print("\n12. Boolean mask (x > 6):\n", x[mask])

    # 13. Sum along a dimension
    print("\n13. Sum along dim=1:\n", x.sum(dim=1))

    # 14. Mean along a dimension
    print("\n14. Mean along dim=0:\n", x.mean(dim=0))

    # 15. Max / Min with indices
    values, indices = x.max(dim=1)
    print("\n15. Max along dim=1 -> values:", values, "indices:", indices)

    # 16. Sorting
    sorted_vals, sorted_idx = torch.sort(x, dim=1, descending=True)
    print("\n16. Sorted (descending) along dim=1:\n", sorted_vals)

    # 17. Clamping
    print("\n17. Clamp values to [3, 8]:\n", x.clamp(3, 8))

    # 18. Broadcasting
    row_vector = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    print("\n18. Broadcasting (x * row_vector):\n", x * row_vector)

    # 19. Type casting
    print("\n19. Cast to int64:\n", x.to(torch.int64))

    # 20. Cloning vs. viewing (memory semantics)
    view_of_x = x.view(-1)
    clone_of_x = x.clone()
    view_of_x[0] = 999.0
    print("\n20. After modifying a view, original x changes:\n", x)
    print("    But clone_of_x remains independent:\n", clone_of_x)


if __name__ == "__main__":
    device = get_device()
    device_management_demo(device)
    twenty_operations(device)
