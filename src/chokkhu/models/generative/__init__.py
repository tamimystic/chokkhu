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

from .flow_matching import (
    VelocityMLP,
    FlowMatching,
    RectifiedFlow,
)

from .normalizing_flows import (
    AffineCouplingLayer,
    RealNVP,
)

from .lora import (
    LoRALinear,
    LoRAAdapter,
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
    # Flow Matching
    "VelocityMLP",
    "FlowMatching",
    "RectifiedFlow",
    # Normalizing Flows
    "AffineCouplingLayer",
    "RealNVP",
    # LoRA
    "LoRALinear",
    "LoRAAdapter",
]
