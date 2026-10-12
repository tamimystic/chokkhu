"""Generative AI, Autoencoders, Diffusion Models, and GANs for Chokkhu."""

from __future__ import annotations

from .autoencoders import (
    VAE,
    VQVAE,
    VectorQuantizer,
)
from .consistency import (
    ConsistencyMLP,
    ConsistencyModel,
    OptimalTransportFlowMatching,
    ReflowMatching,
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
from .diffusion import (
    DDPM,
    DenoisingMLP,
    SinusoidalTimeEmbedding,
)
from .discrete_diffusion import (
    DiscreteTextDiffusion,
)
from .dit import (
    AdaLNZero,
    DiffusionTransformer,
    DiTBlock,
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
from .gflownet import TrajectoryBalanceGFlowNet
from .guidance import (
    ClassifierFreeGuidance,
    DiffusionInpainter,
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
    # Flow Matching & Consistency Models
    "VelocityMLP",
    "FlowMatching",
    "RectifiedFlow",
    "ConsistencyModel",
    "ConsistencyMLP",
    "OptimalTransportFlowMatching",
    "ReflowMatching",
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
    "TrajectoryBalanceGFlowNet",
]
