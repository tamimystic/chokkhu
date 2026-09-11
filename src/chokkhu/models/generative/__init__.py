"""Generative AI, Autoencoders, Diffusion Models, and GANs for Chokkhu."""

from __future__ import annotations

from .autoencoders import (
    VAE,
    VQVAE,
    VectorQuantizer,
)
from .controlnet import (
    ControlNet,
    ControlNetBlock,
    ZeroConv2D,
)
from .cyclegan import (
    CycleGAN,
    CycleGANGenerator,
    PatchGANDiscriminator,
)
from .dit import (
    DiffusionTransformer,
    DiTBlock,
    AdaLNZero,
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
from .discrete_diffusion import (
    DiscreteTextDiffusion,
)
from .guidance import (
    ClassifierFreeGuidance,
    DiffusionInpainter,
)

__all__ = [
    # Autoencoders
    "VAE",
    "VQVAE",
    "VectorQuantizer",
    # Diffusion & Control
    "DDPM",
    "SinusoidalTimeEmbedding",
    "DenoisingMLP",
    "LatentDiffusionModel",
    "LatentCrossAttentionBlock",
    "ControlNet",
    "ControlNetBlock",
    "ZeroConv2D",
    # GANs & Translation
    "Generator",
    "Discriminator",
    "DCGAN",
    "WGANGP",
    "CycleGAN",
    "CycleGANGenerator",
    "PatchGANDiscriminator",
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
    # Discrete Diffusion
    "DiscreteTextDiffusion",
    # Guidance & Inpainting
    "ClassifierFreeGuidance",
    "DiffusionInpainter",
    "DiffusionTransformer",
    "DiTBlock",
    "AdaLNZero",
]
