"""Generative AI, Autoencoders, Diffusion Models, and GANs for Chokkhu."""

from __future__ import annotations
from .autoencoders import (
    VAE,
    VQVAE,
    VectorQuantizer,
)

from .diffusion import (
    DDPM,
    SinusoidalTimeEmbedding,
    DenoisingMLP,
)

from .gan import (
    Generator,
    Discriminator,
    DCGAN,
    WGANGP,
)

__all__ = [
    # Autoencoders
    "VAE",
    "VQVAE",
    "VectorQuantizer",
    # Diffusion
    "DDPM",
    "SinusoidalTimeEmbedding",
    "DenoisingMLP",
    # GANs
    "Generator",
    "Discriminator",
    "DCGAN",
    "WGANGP",
]
