import math
from typing import List

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

import cudaq
from cudaq import spin

# Try to select a GPU-backed target if available
try:
    for t in cudaq.get_targets():
        name = getattr(t, "name", "")
        if "gpu" in name.lower() or "nvidia" in name.lower():
            print("[CUDA-Q] Using target:", name)
            cudaq.set_target(name)
            break
except Exception as e:
    print("[CUDA-Q] Warning: could not set target automatically:", e)


@cudaq.kernel
def vqc_kernel(n_qubits: int, x: List[float], thetas: List[float]):
    """Variational circuit with feature encoding and layered Ry/Rz + ring CX."""
    q = cudaq.qvector(n_qubits)

    # Encode classical features
    for i in range(n_qubits):
        ry(x[i], q[i])

    # Variational layers
    n_layers = len(thetas) // (2 * n_qubits)
    idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            ry(thetas[idx], q[i])
            idx += 1
            rz(thetas[idx], q[i])
            idx += 1
        for i in range(n_qubits - 1):
            cx(q[i], q[i + 1])
        cx(q[n_qubits - 1], q[0])


class QuantumHeadFunction(Function):
    """Autograd wrapper for scalar expectation of H = sum_i Z_i."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, theta: torch.Tensor, n_qubits: int):
        device = x.device
        x_np = x.detach().cpu().numpy().astype(np.float64)
        theta_np = theta.detach().cpu().numpy().astype(np.float64)

        h = sum(spin.z(i) for i in range(n_qubits))

        batch_size = x_np.shape[0]
        exp_vals = np.zeros(batch_size, dtype=np.float64)

        for b in range(batch_size):
            result = cudaq.observe(
                vqc_kernel,
                h,
                n_qubits,
                x_np[b].tolist(),
                theta_np.tolist(),
            )
            exp_vals[b] = result.expectation()

        exp_torch = torch.from_numpy(exp_vals).to(device=device, dtype=torch.float32).view(-1, 1)

        ctx.n_qubits = n_qubits
        ctx.save_for_backward(x, theta, exp_torch)
        return exp_torch

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, theta, _ = ctx.saved_tensors
        device = x.device
        n_qubits = ctx.n_qubits

        x_np = x.detach().cpu().numpy().astype(np.float64)
        theta_np = theta.detach().cpu().numpy().astype(np.float64)
        grad_out_np = grad_output.detach().cpu().numpy().astype(np.float64)

        shift = math.pi / 2.0
        h = sum(spin.z(i) for i in range(n_qubits))

        n_params = theta_np.shape[0]
        grad_theta_np = np.zeros_like(theta_np)

        for j in range(n_params):
            theta_plus = theta_np.copy()
            theta_minus = theta_np.copy()
            theta_plus[j] += shift
            theta_minus[j] -= shift

            grad_j = 0.0
            for b in range(x_np.shape[0]):
                res_plus = cudaq.observe(
                    vqc_kernel,
                    h,
                    n_qubits,
                    x_np[b].tolist(),
                    theta_plus.tolist(),
                )
                res_minus = cudaq.observe(
                    vqc_kernel,
                    h,
                    n_qubits,
                    x_np[b].tolist(),
                    theta_minus.tolist(),
                )
                df_dtheta = (res_plus.expectation() - res_minus.expectation()) / (2.0 * shift)
                grad_j += df_dtheta * grad_out_np[b, 0]

            grad_theta_np[j] = grad_j

        grad_theta = torch.from_numpy(grad_theta_np).to(device=device, dtype=theta.dtype)

        grad_x = torch.zeros_like(x)
        return grad_x, grad_theta, None


class CudaQHead(nn.Module):
    """Quantum head mapping [batch, n_qubits] -> [batch, n_classes]."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, n_classes: int = 10):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        n_params = 2 * n_qubits * n_layers
        self.theta = nn.Parameter(0.01 * torch.randn(n_params))

        self.classifier = nn.Linear(1, n_classes)

    def forward(self, embeddings: torch.Tensor):
        # Normalize embeddings to [-pi, pi] for stable encoding
        angles = torch.tanh(embeddings) * math.pi
        q_exp = QuantumHeadFunction.apply(angles, self.theta, self.n_qubits)  # [batch, 1]
        q_exp = q_exp / float(self.n_qubits)
        logits = self.classifier(q_exp)
        return logits
