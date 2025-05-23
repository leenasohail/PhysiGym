import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0).sub(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, simnorm_dim: int = 8):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class FeatureExtractor(nn.Module):
    """Handles both image-based and vector-based state inputs dynamically."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg

        obs_shape = env.observation_space.shape
        self.is_image = len(obs_shape) == 3  # Check if input is an image (C, H, W)

        if self.is_image:
            # CNN feature extractor
            num_channels = 8
            layers = [
                PixelPreprocess(),
                nn.Conv2d(obs_shape[0], num_channels, 7 * 8, stride=5),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 8, stride=5),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=3),
                nn.Mish(inplace=False),
                nn.Conv2d(num_channels, num_channels, 3, stride=1),
                nn.Flatten(),
            ]
            self.feature_extractor = nn.Sequential(*layers)
            self.feature_size = self._get_feature_size(obs_shape)
        else:
            # Directly flatten vector input
            self.feature_extractor = nn.Identity()
            self.feature_size = np.prod(obs_shape)

    def _get_feature_size(self, obs_shape):
        """Pass a dummy tensor through CNN to compute feature size dynamically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            out = self.feature_extractor(dummy_input)
            return int(np.prod(out.shape[1:]))

    def forward(self, x):
        if self.is_image:
            x = self.feature_extractor(x)  # Apply CNN
            x = x.view(x.size(0), -1)  # Flatten
        return x


class QNetwork(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.LazyLinear(256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.LazyLinear(64)
        self.fc4 = nn.LazyLinear(out_features=1)
        self.mish = nn.Mish()
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = self.feature_extractor(x)  # Extract features
        x = torch.cat([x, a], dim=1)  # Concatenate state and action

        x = self.mish(self.ln1(self.fc1(x)))
        x = self.mish(self.ln2(self.fc2(x)))
        x = self.mish(self.fc3(x))
        x = self.relu(
            self.fc4(x)
        )  # value Q function superior or equal to zero because the reward is also superior to zero and one
        return x

class ActorContinuous(nn.Module):
    """Policy network (ActorContinuous)"""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(env, cfg["cfg_FeatureExtractor"])
        action_dim = np.prod(env.action_space.shape)
        self.log_std_max = 2
        self.log_std_min = -5

        # Fully connected layers
        self.fc1 = nn.LazyLinear(256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.LazyLinear(256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.LazyLinear(256)
        self.fc_mean = nn.LazyLinear(action_dim)
        self.fc_logstd = nn.LazyLinear(action_dim)
        self.relu = nn.ReLU()
        self.mish = nn.Mish()
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features

        x = self.mish(self.ln1(self.fc1(x)))
        x = self.mish(self.ln2(self.fc2(x)))
        x = self.relu(self.fc3(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # Stable variance scaling

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class CellTransformerEncoder(nn.Module):
    def __init__(self, type_vocab_size, embed_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()

        # Embedding layers for input features
        self.type_embedding = nn.Embedding(type_vocab_size, embed_dim)
        self.dead_embedding = nn.Embedding(2, embed_dim)  # assuming dead is 0 or 1
        self.pos_linear = nn.Linear(2, embed_dim)  # (x, y) -> embed_dim

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # Set to True for (B, T, D) input shape
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection (optional, can be adjusted to your use case)
        self.output_head = nn.Linear(embed_dim, embed_dim)  # for example, to project to latent features

    def forward(self, state):
        # state: TensorDict with keys "type", "dead", "pos", "mask"
        type_embed = self.type_embedding(state["type"])           # (B, T, D)
        dead_embed = self.dead_embedding(state["dead"])           # (B, T, D)
        pos_embed = self.pos_linear(state["pos"])                 # (B, T, D)

        # Combine embeddings
        x = type_embed + dead_embed + pos_embed                    # (B, T, D)

        # Build attention mask
        attn_mask = ~state["mask"].bool()                          # (B, T), True where we want to ignore

        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (B, T, D)

        # Project to output (if needed)
        x = self.output_head(x)
        return x
