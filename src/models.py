import torch
import torch.nn as nn

from quantum_head import CudaQHead
from quantum_attention import CudaQSelfAttention


class ConvBackbone(nn.Module):
    """Small CNN mapping 1x28x28 FashionMNIST images to an embedding vector."""

    def __init__(self, embedding_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTBackbone(nn.Module):
    """Tiny Vision Transformer backbone returning a CLS embedding."""

    def __init__(
        self,
        embedding_dim: int = 4,
        patch_size: int = 7,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) * (28 // patch_size)
        patch_dim = patch_size * patch_size

        self.to_patches = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.proj = nn.Linear(dim, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        patches = self.to_patches(x)
        patches = patches.transpose(1, 2)
        tokens = self.patch_embed(patches)
        cls_tok = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        enc = self.encoder(tokens)
        cls_out = enc[:, 0, :]
        emb = self.proj(cls_out)
        emb = self.act(emb)
        return emb


class ViTBackboneTokens(nn.Module):
    """ViT-style backbone returning full token sequence (CLS + patches)."""

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 7,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) * (28 // patch_size)
        patch_dim = patch_size * patch_size

        self.to_patches = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        patches = self.to_patches(x)
        patches = patches.transpose(1, 2)
        tokens = self.patch_embed(patches)
        cls_tok = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        enc = self.encoder(tokens)
        return enc


class ClassicalHead(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalModelCNN(nn.Module):
    def __init__(self, embedding_dim: int = 4, n_classes: int = 10):
        super().__init__()
        self.backbone = ConvBackbone(embedding_dim=embedding_dim)
        self.head = ClassicalHead(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        return logits


class ClassicalModelViT(nn.Module):
    def __init__(self, embedding_dim: int = 4, n_classes: int = 10):
        super().__init__()
        self.backbone = ViTBackbone(embedding_dim=embedding_dim)
        self.head = ClassicalHead(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        return logits


class HybridModelCNN(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, n_classes: int = 10):
        super().__init__()
        self.backbone = ConvBackbone(embedding_dim=n_qubits)
        self.head = CudaQHead(n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        return logits


class HybridModelViT(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, n_classes: int = 10):
        super().__init__()
        self.backbone = ViTBackbone(embedding_dim=n_qubits)
        self.head = CudaQHead(n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.head(emb)
        return logits


class HybridViTQuantumAttention(nn.Module):
    """ViT backbone + CUDA-Q self-attention + CLS classifier."""

    def __init__(
        self,
        dim: int = 64,
        patch_size: int = 7,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 128,
        n_qubits: int = 4,
        n_layers: int = 1,
        max_tokens: int = 4,
        n_classes: int = 10,
    ):
        super().__init__()
        self.backbone_tokens = ViTBackboneTokens(
            dim=dim,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )
        self.q_self_attn = CudaQSelfAttention(
            d_model=dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            max_tokens=max_tokens,
            dropout=0.0,
        )
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone_tokens(x)
        tokens = self.q_self_attn(tokens)
        cls = tokens[:, 0, :]
        logits = self.classifier(cls)
        return logits
