"""
Day 50 - PyTorch Tensors & Autograd
Module 1: nn.Module Anatomy, parameters(), and Optimizers (SGD, Adam)

Covers:
    - How nn.Module works internally (subclassing, __init__, forward)
    - Registering learnable parameters and submodules automatically
    - Inspecting a model's parameters via .parameters() / .named_parameters()
    - Building optimizers (SGD, Adam) and understanding their update rule
    - The standard optimizer.zero_grad() / loss.backward() / optimizer.step() cycle

Run:
    python 01_nn_module_anatomy.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class TinyNet(nn.Module):
    """
    A minimal nn.Module subclass to demonstrate the anatomy of a model.

    Every custom module must:
      1. Call super().__init__() first.
      2. Register learnable layers/parameters as attributes in __init__.
      3. Define forward() to describe how input flows through the layers.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()  # mandatory: sets up internal parameter tracking

        # Layers assigned as attributes are automatically registered
        # as submodules -- nn.Module tracks them via __setattr__.
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def module_anatomy_demo() -> None:
    section("1. nn.Module Anatomy")

    model = TinyNet(in_features=4, hidden_features=8, out_features=1)
    print(model)  # nn.Module implements a readable __repr__ automatically

    # A forward pass -- calling model(x) invokes model.forward(x)
    x = torch.rand(2, 4)
    output = model(x)
    print("\nInput shape :", x.shape)
    print("Output shape:", output.shape)


def parameters_inspection() -> None:
    section("2. Inspecting Parameters")

    model = TinyNet(in_features=4, hidden_features=8, out_features=1)

    # .parameters() yields every learnable tensor (weights and biases)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total learnable parameters:", total_params)

    # .named_parameters() also gives the name, useful for debugging/logging
    for name, param in model.named_parameters():
        print(f"  {name:20s} shape={tuple(param.shape)} requires_grad={param.requires_grad}")

    # Every parameter is a leaf tensor with requires_grad=True by default
    first_param = next(model.parameters())
    print("\nFirst parameter is_leaf:", first_param.is_leaf)
    print("First parameter requires_grad:", first_param.requires_grad)


def optimizer_comparison() -> None:
    section("3. Optimizers: SGD vs. Adam")

    torch.manual_seed(0)
    model_sgd = TinyNet(4, 8, 1)
    model_adam = TinyNet(4, 8, 1)
    model_adam.load_state_dict(model_sgd.state_dict())  # identical starting weights

    x = torch.rand(16, 4)
    y = torch.rand(16, 1)
    loss_fn = nn.MSELoss()

    # SGD: simple update rule  ->  param = param - lr * grad
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.1)

    # Adam: adaptive learning rates per parameter using running estimates
    # of the first and second moments of the gradient (momentum + RMSProp-like scaling)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.1)

    print(f"{'step':>4} | {'SGD loss':>10} | {'Adam loss':>10}")
    print("-" * 32)
    for step in range(5):
        # --- SGD step ---
        optimizer_sgd.zero_grad()          # 1. clear old gradients
        pred_sgd = model_sgd(x)             # 2. forward pass
        loss_sgd = loss_fn(pred_sgd, y)      # 3. compute loss
        loss_sgd.backward()                   # 4. backpropagate
        optimizer_sgd.step()                   # 5. update parameters

        # --- Adam step ---
        optimizer_adam.zero_grad()
        pred_adam = model_adam(x)
        loss_adam = loss_fn(pred_adam, y)
        loss_adam.backward()
        optimizer_adam.step()

        print(f"{step:>4} | {loss_sgd.item():>10.4f} | {loss_adam.item():>10.4f}")


def optimizer_internals() -> None:
    section("4. What optimizer.step() Actually Does (Conceptually)")

    print(
        "SGD update rule (per parameter):\n"
        "    param = param - lr * param.grad\n"
    )
    print(
        "Adam update rule (per parameter), using running averages of the\n"
        "gradient (m) and squared gradient (v):\n"
        "    m = beta1 * m + (1 - beta1) * grad\n"
        "    v = beta2 * v + (1 - beta2) * grad**2\n"
        "    param = param - lr * m_hat / (sqrt(v_hat) + eps)\n"
        "This adapts the effective step size per parameter, which is why\n"
        "Adam often converges faster with less manual learning-rate tuning."
    )


if __name__ == "__main__":
    module_anatomy_demo()
    parameters_inspection()
    optimizer_comparison()
    optimizer_internals()
