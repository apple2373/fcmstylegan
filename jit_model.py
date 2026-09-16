"""Profile-conditioned adapter around the local JiT implementation."""
from pathlib import Path
import sys
import torch.nn as nn

_JIT_ROOT = Path(__file__).resolve().parent / "JiT"
if str(_JIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_JIT_ROOT))
from model_jit import JiT_models  # noqa: E402

class ProfileMLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 128, out_dim), nn.SiLU())

    def forward(self, profile):
        if profile.ndim != 3 or tuple(profile.shape[1:]) != (3, 128):
            raise ValueError(f"profile must have shape (batch, 3, 128), got {tuple(profile.shape)}")
        return self.net(profile)

class ConditionalJiT(nn.Module):
    def __init__(self, model_name="JiT-B/16", img_size=128, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.net = JiT_models[model_name](input_size=img_size, in_channels=1, num_classes=1,
                                          attn_drop=attn_dropout, proj_drop=proj_dropout)
        # Reuse JiT's AdaLN and in-context conditioning path with profile embeddings.
        self.net.y_embedder = ProfileMLP(self.net.hidden_size)

    def forward(self, image, timestep, profile):
        return self.net(image, timestep, profile)
