"""Multi-Modal Vision-Language and Cross-Modal Alignment Package.

Pure NumPy implementations of:
- CLIP: Dual-encoder contrastive language-image pre-training
- SigLIP: Sigmoid loss language-image pre-training
- CLAP: Contrastive language-audio pre-training
- LLaVAProjector: Linear, MLP, and Perceiver Resampler multimodal alignment adapters
"""

from .clap import CLAP, CLAPAudioEncoder, CLAPTextEncoder
from .clip import CLIP, CLIPTextEncoder, CLIPVisionEncoder
from .llava_projector import LLaVALinearProjector, LLaVAMLPProjector, PerceiverResampler
from .siglip import SigLIP

__all__ = [
    "CLIP",
    "CLIPVisionEncoder",
    "CLIPTextEncoder",
    "SigLIP",
    "CLAP",
    "CLAPAudioEncoder",
    "CLAPTextEncoder",
    "LLaVALinearProjector",
    "LLaVAMLPProjector",
    "PerceiverResampler",
]
