import math
from typing import List

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

import cudaq
from cudaq import spin

# Try to pick a GPU-backed target if available
try:
    for t in cudaq.get_targets():
        name = getattr(t, "name", "")
        if "gpu" in name.lower() or "nvidia" in name.lower():
            print("[CUDA-Q] (attention) Using target:", name)
            cudaq.set_target(name)
            break
except Exception as e:
    print("[CUDA-Q] Warning in quantum_attention: could not set target:", e)


@cudaq.kernel
def similarity_kernel(n_qubits: int, x: List[float], thetas: List[float]):
    """Variational similarity circuit for token-pair differences."""
    q = cudaq.qvector(n_qubits)

    for i in range(n_qubits):
        ry(x[i], q[i])

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


class QuantumSimilarityFunction(Function):
    """Autograd Function for quantum similarity scores across token pairs."""

    @staticmethod
    def forward(ctx, x_pairs: torch.Tensor, theta: torch.Tensor, n_qubits: int):
        device = x_pairs.device
        x_np = x_pairs.detach().cpu().numpy().astype(np.float64)
        theta_np = theta.detach().cpu().numpy().astype(np.float64)

        h = sum(spin.z(i) for i in range(n_qubits))

        n_pairs = x_np.shape[0]
        exp_vals = np.zeros(n_pairs, dtype=np.float64)

        for i in range(n_pairs):
            result = cudaq.observe(
                similarity_kernel,
                h,
                n_qubits,
                x_np[i].tolist(),
                theta_np.tolist(),
            )
            exp_vals[i] = result.expectation()

        sims = torch.from_numpy(exp_vals).to(device=device, dtype=torch.float32).view(-1, 1)

        ctx.n_qubits = n_qubits
        ctx.save_for_backward(x_pairs, theta, sims)
        return sims

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_pairs, theta, _ = ctx.saved_tensors
        device = x_pairs.device
        n_qubits = ctx.n_qubits

        x_np = x_pairs.detach().cpu().numpy().astype(np.float64)
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
            for i in range(x_np.shape[0]):
                res_plus = cudaq.observe(
                    similarity_kernel,
                    h,
                    n_qubits,
                    x_np[i].tolist(),
                    theta_plus.tolist(),
                )
                res_minus = cudaq.observe(
                    similarity_kernel,
                    h,
                    n_qubits,
                    x_np[i].tolist(),
                    theta_minus.tolist(),
                )
                df_dtheta = (res_plus.expectation() - res_minus.expectation()) / (2.0 * shift)
                grad_j += df_dtheta * grad_out_np[i, 0]

            grad_theta_np[j] = grad_j

        grad_theta = torch.from_numpy(grad_theta_np).to(device=device, dtype=theta.dtype)
        grad_x_pairs = torch.zeros_like(x_pairs)
        return grad_x_pairs, grad_theta, None


class CudaQSelfAttention(nn.Module):
    """Single-head quantum similarity-based self-attention.

    Input tokens: [batch, T, d_model]

    - Project tokens to n_qubits dimension.
    - Build pairwise differences q_i - q_j.
    - Evaluate quantum similarity via CUDA-Q.
    - Softmax over j for each i.
    - Weighted sum of values (original tokens).
    - Residual + linear projection.
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        n_layers: int = 1,
        max_tokens: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.max_tokens = max_tokens

        self.to_qubits = nn.Linear(d_model, n_qubits)

        n_params = 2 * n_qubits * n_layers
        self.theta = nn.Parameter(0.01 * torch.randn(n_params))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor):
        b, T, d = tokens.shape
        if T > self.max_tokens:
            tokens = tokens[:, : self.max_tokens, :]
            T = self.max_tokens

        device = tokens.device

        q_embed = self.to_qubits(tokens)               # [b, T, n_qubits]
        q_embed = torch.tanh(q_embed) * math.pi        # [-pi, pi]

        q_i = q_embed.unsqueeze(2).expand(-1, -1, T, -1)
        q_j = q_embed.unsqueeze(1).expand(-1, T, -1, -1)
        q_diff = q_i - q_j                             # [b, T, T, n_qubits]

        x_pairs = q_diff.reshape(b * T * T, self.n_qubits)

        sims = QuantumSimilarityFunction.apply(x_pairs, self.theta, self.n_qubits)
        sims = sims.view(b, T, T)

        sims = sims / math.sqrt(self.n_qubits)
        attn_weights = torch.softmax(sims, dim=-1)
        attn_weights = self.dropout(attn_weights)

        values = tokens
        attn_out = torch.matmul(attn_weights, values)  # [b, T, d_model]

        out = self.out_proj(attn_out)
        out = out + tokens
        return out
