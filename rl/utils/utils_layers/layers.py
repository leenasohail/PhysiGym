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
    def __init__(self, dropout=0.1):
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        super().__init__()

        # Embedding layers for input features
        self.type_embedding = nn.Linear(1, 2)
        self.dead_embedding = nn.Linear(1, 2)  # assuming dead is 0 or 1
        self.pos_linear = nn.Linear(2, 2)  # (x, y) -> embed_dim

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2,
            nhead=1,
            dim_feedforward=8,
            dropout=dropout,
            batch_first=True,  # Set to True for (B, T, D) input shape
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Output projection (optional, can be adjusted to your use case)
        self.output_head = nn.Linear(
            2, 10
        )  # for example, to project to latent features

    def forward(self, state):
        # state: TensorDict with keys "type", "dead", "pos", "mask"
        type_embed = self.type_embedding(state["type"])
        dead_embed = self.dead_embedding(state["dead"])  # (B, T, D)
        pos_embed = self.pos_linear(state["pos"])  # (B, T, D)

        # Combine embeddings
        x = type_embed + dead_embed + pos_embed  # (B, T, D)

        # Build attention mask
        attn_mask = ~state["mask"].bool()  # (B, T), True where we want to ignore

        # Pass through transformer
        x = self.transformer_encoder(
            x, src_key_padding_mask=attn_mask.squeeze(-1)
        )  # (B, T, D)

        # Project to output (if needed)
        x = self.output_head(x)
        return x


class CellTransformerEncoderWithCNN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.type_embedding = nn.Linear(1, 2)
        self.dead_embedding = nn.Linear(1, 2)
        self.pos_linear = nn.Linear(2, 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2, nhead=1, dim_feedforward=8, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_head = nn.Linear(2, 10)

        self.conv1 = nn.Conv1d(
            in_channels=10, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final = nn.Linear(32, 100)

    def forward(self, state):
        type_embed = self.type_embedding(state["type"])
        dead_embed = self.dead_embedding(state["dead"])
        pos_embed = self.pos_linear(state["pos"])

        x = type_embed + dead_embed + pos_embed
        attn_mask = ~state["mask"].bool()
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask.squeeze(-1))
        x = self.output_head(x)

        # Apply mask before Conv1D
        mask = state["mask"]
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # (B, T, 1)
        x = x * mask

        # Conv1D + Pooling
        x = x.permute(0, 2, 1)  # (B, 10, T)
        x = self.conv1(x)  # (B, 32, T')
        x = self.pool(x)  # (B, 32, 1)
        x = x.squeeze(-1)  # (B, 32)
        x = self.final(x)  # (B, 16)

        return x


class CellTransformerEncoderWithCLSCNN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.embed_dim = 64
        self.type_embedding = nn.Linear(1, self.embed_dim)
        self.dead_embedding = nn.Linear(1, self.embed_dim)
        self.pos_linear = nn.Linear(2, self.embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CNN for summarizing the token sequence (excluding CLS)
        self.cnn_summary = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, D, 1)
        )

        # Final output head
        self.output_head = nn.Linear(self.embed_dim * 2, 10)

    def forward(self, state):
        B, T, _ = state["type"].shape

        # Embedding
        type_embed = self.type_embedding(state["type"])
        dead_embed = self.dead_embedding(state["dead"])
        pos_embed = self.pos_linear(state["pos"])
        x = type_embed + dead_embed + pos_embed  # (B, T, D)

        # CLS token
        cls_token = self.cls_token.expand(B, 1, self.embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, T+1, D)

        attn_mask = ~state["mask"].bool()  # might be (B, T, 1)
        if attn_mask.dim() == 3 and attn_mask.size(-1) == 1:
            attn_mask = attn_mask.squeeze(-1)  # (B, T)

        cls_pad = torch.zeros(
            (B, 1), dtype=torch.bool, device=attn_mask.device
        )  # (B, 1)
        attn_mask = torch.cat([cls_pad, attn_mask], dim=1)  # (B, T+1)

        # Transformer
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (B, T+1, D)

        # Split CLS and tokens
        cls_out = x[:, 0]  # (B, D)
        tokens = x[:, 1:]  # (B, T, D)

        # CNN summary
        tokens_cnn = self.cnn_summary(tokens.transpose(1, 2)).squeeze(-1)  # (B, D)

        # Combine CLS + CNN
        combined = torch.cat([cls_out, tokens_cnn], dim=-1)  # (B, 2D)
        return self.output_head(combined)  # (B, output_dim)
