"""Action Chunking with Transformers policy."""

import torch
from hydra.utils import instantiate
from torch import nn


class ACTPolicy(nn.Module):
    """Conditional-VAE transformer predicting absolute joint-target chunks.

    Images must be float BCHW tensors in [0, 1]. During training, passing an
    action chunk enables the posterior; inference uses the zero latent prior.
    """

    def __init__(self, config):
        super().__init__()
        config = dict(config)
        # Preserve inference compatibility with checkpoints created before the
        # encoder choice was stored explicitly.
        config.setdefault("image_encoder", {
            "_target_": "learning.models.encoders.CNNImageEncoder",
        })
        required = {
            "state_dim", "action_dim", "chunk_size", "camera_names", "hidden_dim", "latent_dim",
            "feedforward_dim", "num_heads", "num_layers", "dropout", "image_encoder",
        }
        missing = required.difference(config)
        unexpected = set(config).difference(required)
        if missing or unexpected:
            raise ValueError(f"Invalid ACT config; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        self.config = config
        if self.config["chunk_size"] <= 0 or self.config["hidden_dim"] % self.config["num_heads"]:
            raise ValueError("chunk_size must be positive and hidden_dim divisible by num_heads")
        if self.config["state_dim"] < 0 or (not self.config["state_dim"] and not self.config["camera_names"]):
            raise ValueError("state_dim must be nonnegative and at least one policy input is required")
        h = self.config["hidden_dim"]
        self.state_encoder = (
            nn.Sequential(nn.Linear(self.config["state_dim"], h), nn.LayerNorm(h))
            if self.config["state_dim"] else None
        )
        self.image_encoders = nn.ModuleDict({
            name: instantiate(
                self.config["image_encoder"],
                output_dim=h,
                _execution_whitelist_="learning.models.encoders.*",
            )
            for name in self.config["camera_names"]
        })
        self.camera_embeddings = nn.Parameter(torch.empty(len(self.config["camera_names"]), h))
        self.latent_encoder = nn.Sequential(
            nn.Linear(
                self.config["state_dim"] + self.config["chunk_size"] * self.config["action_dim"]
                + self.config["chunk_size"], h,
            ), nn.ReLU(),
            nn.Linear(h, self.config["latent_dim"] * 2),
        )
        self.latent_project = nn.Linear(self.config["latent_dim"], h)
        layer = nn.TransformerDecoderLayer(
            d_model=h, nhead=self.config["num_heads"], dim_feedforward=self.config["feedforward_dim"],
            dropout=self.config["dropout"], batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.config["num_layers"])
        self.action_queries = nn.Parameter(torch.empty(self.config["chunk_size"], h))
        self.action_head = nn.Linear(h, self.config["action_dim"])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.action_queries, std=0.02)
        if self.camera_embeddings.numel():
            nn.init.normal_(self.camera_embeddings, std=0.02)

    def forward(self, state, images=None, actions=None, is_pad=None):
        if state.ndim != 2 or state.shape[-1] != self.config["state_dim"]:
            raise ValueError(f"Expected state [B,{self.config['state_dim']}], got {tuple(state.shape)}")
        images = images or {}
        missing = set(self.config["camera_names"]).difference(images)
        if missing:
            raise KeyError(f"Missing configured cameras: {sorted(missing)}")
        memory = self._encode_observations(state, images)
        latent, mu, logvar = self._encode_latent(state, actions, is_pad)
        memory.append(self.latent_project(latent))
        memory = torch.stack(memory, dim=1)
        queries = self.action_queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        return self.action_head(self.decoder(queries, memory)), mu, logvar

    def _encode_observations(self, state, images):
        """Build the state and camera memory tokens consumed by ACT."""
        memory = [self.state_encoder(state)] if self.state_encoder is not None else []
        for index, name in enumerate(self.config["camera_names"]):
            image_token = self.image_encoders[name](images[name])
            memory.append(image_token + self.camera_embeddings[index])
        return memory

    def _encode_latent(self, state, actions, is_pad):
        """Sample the training posterior or return the inference-time prior."""
        if actions is None:
            latent = state.new_zeros((state.shape[0], self.config["latent_dim"]))
            return latent, None, None

        expected = (state.shape[0], self.config["chunk_size"], self.config["action_dim"])
        if tuple(actions.shape) != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        if is_pad is None or tuple(is_pad.shape) != expected[:2]:
            raise ValueError(f"Expected is_pad {expected[:2]} with training actions")
        posterior = torch.cat((state, actions.flatten(1), is_pad.to(state.dtype)), dim=-1)
        mu, logvar = self.latent_encoder(posterior).chunk(2, dim=-1)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return latent, mu, logvar

    @torch.no_grad()
    def predict(self, state, images=None):
        was_training = self.training
        self.eval()
        actions = self(state, images)[0]
        self.train(was_training)
        return actions
