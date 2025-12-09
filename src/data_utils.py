import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_mnist_loaders(batch_size: int = 64, num_workers: int = 2):
    """Return train and test DataLoaders for FashionMNIST."""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
