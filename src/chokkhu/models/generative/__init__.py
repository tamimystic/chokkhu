"""Generative AI, Autoencoders, Diffusion Models, and GANs for Chokkhu."""

from __future__ import annotations

from .autoencoders import (
    VAE,
    VQVAE,
    VectorQuantizer,
)
from .diffusion import (
    DDPM,
    DenoisingMLP,
    SinusoidalTimeEmbedding,
)
from .flow_matching import (
    FlowMatching,
    RectifiedFlow,
    VelocityMLP,
)
from .gan import (
    DCGAN,
    WGANGP,
    Discriminator,
    Generator,
)
from .latent_diffusion import (
    LatentCrossAttentionBlock,
    LatentDiffusionModel,
)
from .lora import (
    LoRAAdapter,
    LoRALinear,
)
from .normalizing_flows import (
    AffineCouplingLayer,
    RealNVP,
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
    "LatentDiffusionModel",
    "LatentCrossAttentionBlock",
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
