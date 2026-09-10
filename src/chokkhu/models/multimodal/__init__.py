"""Multi-Modal Vision-Language and Cross-Modal Alignment Package.

Pure NumPy implementations of:
- CLIP: Dual-encoder contrastive language-image pre-training
- SigLIP: Sigmoid loss language-image pre-training
- LLaVAProjector: Linear, MLP, and Perceiver Resampler multimodal alignment adapters
"""

from .clip import CLIP, CLIPVisionEncoder, CLIPTextEncoder
from .siglip import SigLIP
from .llava_projector import LLaVALinearProjector, LLaVAMLPProjector, PerceiverResampler

__all__ = [
    "CLIP",
    "CLIPVisionEncoder",
    "CLIPTextEncoder",
    "SigLIP",
    "LLaVALinearProjector",
    "LLaVAMLPProjector",
    "PerceiverResampler",
]
