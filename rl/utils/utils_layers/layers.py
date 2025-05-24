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
    def __init__(self, mode="cls_cnn", dropout=0.1, embed_dim=64, output_dim=10):
        """
        mode options:
          - 'basic': simple sum embeddings + transformer + linear output
          - 'cnn': like basic but adds 1D CNN pooling after transformer output
          - 'cls_cnn': adds CLS token, transformer, CNN summary of tokens + CLS concat + final head
        """
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Embeddings
        self.type_embedding = nn.Linear(1, embed_dim)
        self.dead_embedding = nn.Linear(1, embed_dim)
        self.pos_linear = nn.Linear(2, embed_dim)

        # CLS token only used in 'cls_cnn' mode
        if self.mode == "cls_cnn":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4 if embed_dim >= 32 else 1,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CNN layers only used in 'cnn' and 'cls_cnn' modes
        if self.mode == "cnn":
            self.conv1 = nn.Conv1d(
                in_channels=output_dim,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.final = nn.Linear(32, output_dim)
        elif self.mode == "cls_cnn":
            self.cnn_summary = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.output_head = nn.Linear(embed_dim * 2, output_dim)
        else:
            # basic mode just projects transformer output tokens to output_dim
            self.output_head = nn.Linear(embed_dim, output_dim)

    def forward(self, state):
        B, T = state["type"].shape[:2]

        # Embed inputs
        type_embed = self.type_embedding(state["type"])
        dead_embed = self.dead_embedding(state["dead"])
        pos_embed = self.pos_linear(state["pos"])

        x = type_embed + dead_embed + pos_embed  # (B, T, D)

        # Build padding mask (True where to ignore)
        attn_mask = ~state["mask"].bool()
        if attn_mask.dim() == 3 and attn_mask.size(-1) == 1:
            attn_mask = attn_mask.squeeze(-1)

        if self.mode == "cls_cnn":
            # Add CLS token
            cls_token = self.cls_token.expand(B, 1, self.embed_dim)
            x = torch.cat([cls_token, x], dim=1)  # (B, T+1, D)

            # Update mask for CLS token: CLS token is never masked
            cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=attn_mask.device)
            attn_mask = torch.cat([cls_pad, attn_mask], dim=1)  # (B, T+1)

            x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
            cls_out = x[:, 0]  # CLS output (B, D)
            tokens = x[:, 1:]  # token outputs (B, T, D)

            tokens_cnn = self.cnn_summary(tokens.transpose(1, 2)).squeeze(-1)  # (B, D)
            combined = torch.cat([cls_out, tokens_cnn], dim=-1)  # (B, 2D)
            out = self.output_head(combined)  # (B, output_dim)
            return out

        else:
            # Basic and cnn modes: no CLS token
            x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (B, T, D)
            x = self.output_head(x)  # (B, T, output_dim)

            if self.mode == "cnn":
                mask = state["mask"]
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)  # (B, T, 1)
                x = x * mask  # mask out padded tokens

                x = x.permute(0, 2, 1)  # (B, output_dim, T)
                x = self.conv1(x)  # (B, 32, T')
                x = self.pool(x)  # (B, 32, 1)
                x = x.squeeze(-1)  # (B, 32)
                x = self.final(x)  # (B, output_dim)
                return x

            # Basic mode returns all token outputs (B, T, output_dim)
            return x  # (B, output_dim)
