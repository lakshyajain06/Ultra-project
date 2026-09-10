"""A compact, dependency-light ACT policy for Ultra demonstrations."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ACTConfig:
    state_dim: int = 44
    action_dim: int = 22
    chunk_size: int = 25
    camera_names: tuple[str, ...] = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    hidden_dim: int = 256
    latent_dim: int = 32
    feedforward_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    def to_dict(self):
        value = asdict(self)
        value["camera_names"] = list(self.camera_names)
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["camera_names"] = tuple(value.get("camera_names", ()))
        return cls(**value)


class ImageEncoder(nn.Module):
    """Small CNN suited to the low-resolution dataset cameras."""

    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(512, output_dim),
        )

    def forward(self, image):
        return self.net(image)


class ACTPolicy(nn.Module):
    """Conditional-VAE transformer predicting absolute joint-target chunks.

    Images must be float BCHW tensors in [0, 1]. During training, passing an
    action chunk enables the posterior; inference uses the zero latent prior.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        if config.chunk_size <= 0 or config.hidden_dim % config.num_heads:
            raise ValueError("chunk_size must be positive and hidden_dim divisible by num_heads")
        self.config = config
        h = config.hidden_dim
        self.state_encoder = nn.Sequential(nn.Linear(config.state_dim, h), nn.LayerNorm(h))
        self.image_encoders = nn.ModuleDict({name: ImageEncoder(h) for name in config.camera_names})
        self.camera_embeddings = nn.Parameter(torch.empty(len(config.camera_names), h))
        self.latent_encoder = nn.Sequential(
            nn.Linear(config.state_dim + config.chunk_size * config.action_dim + config.chunk_size, h), nn.ReLU(),
            nn.Linear(h, config.latent_dim * 2),
        )
        self.latent_project = nn.Linear(config.latent_dim, h)
        layer = nn.TransformerDecoderLayer(
            d_model=h, nhead=config.num_heads, dim_feedforward=config.feedforward_dim,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.action_queries = nn.Parameter(torch.empty(config.chunk_size, h))
        self.action_head = nn.Linear(h, config.action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.action_queries, std=0.02)
        if self.camera_embeddings.numel():
            nn.init.normal_(self.camera_embeddings, std=0.02)

    def forward(self, state, images=None, actions=None, is_pad=None):
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected state [B,{self.config.state_dim}], got {tuple(state.shape)}")
        images = images or {}
        missing = set(self.config.camera_names).difference(images)
        if missing:
            raise KeyError(f"Missing configured cameras: {sorted(missing)}")
        memory = [self.state_encoder(state)]
        for index, name in enumerate(self.config.camera_names):
            memory.append(self.image_encoders[name](images[name]) + self.camera_embeddings[index])
        mu = logvar = None
        if actions is not None:
            expected = (state.shape[0], self.config.chunk_size, self.config.action_dim)
            if tuple(actions.shape) != expected:
                raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
            if is_pad is None or tuple(is_pad.shape) != expected[:2]:
                raise ValueError(f"Expected is_pad {expected[:2]} with training actions")
            posterior = torch.cat((state, actions.flatten(1), is_pad.to(state.dtype)), dim=-1)
            mu, logvar = self.latent_encoder(posterior).chunk(2, dim=-1)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            latent = state.new_zeros((state.shape[0], self.config.latent_dim))
        memory.append(self.latent_project(latent))
        memory = torch.stack(memory, dim=1)
        queries = self.action_queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        return self.action_head(self.decoder(queries, memory)), mu, logvar

    @torch.no_grad()
    def predict(self, state, images=None):
        was_training = self.training
        self.eval()
        actions = self(state, images)[0]
        self.train(was_training)
        return actions
