"""
model.py
========
GN2 Transformer architecture for jet flavour tagging.

Architecture:

    Input:
        - jet features   (n_jet_vars,)
        - track features (n_tracks, n_track_vars)
        - padding mask   (n_tracks,): True = real track, False = padded

    Per-track initialiser:
        - Linear(n_jet_vars + n_track_vars -> 256) + ReLU   (1 hidden layer)
        - Linear(256 -> 256)                                (output layer)

    Transformer Encoder:
        - 4 layers, 8 attention heads
        - embedding dim = 256, feed-forward dim = 512
        - Pre-LayerNorm

    Pooling:
        - Attention pooling: global jet representation (dim 128)
        - Per-track projection: track embeddings       (dim 128)

    Task-specific heads  (3 hidden layers: 128 -> 64 -> 32):
        1. Jet classification (primary): 4 classes  (b, c, light, tau)
"""

import logging
from pathlib import Path

import torch
from torch import nn

from ._constants import FLAVOUR_LABELS_INV
from .utils import get_activation, parse_label_map

logger = logging.getLogger("GN2.model     ")


class TransformerLayer(nn.Module):
    """
    Transformer encoder layer with ``Pre-LayerNorm`` (before attention & FFN).

    Attributes:
        d_model (int) : embedding dimension
        n_heads (int) : number of attention heads
        d_ff (int) : feed-forward inner dimension
        dropout (float) : dropout rate
    """
    def __init__(
        self,
        dim_emb: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """
        Initialize the transformer layer.

        Args:
            d_model (int): embedding dimension (default ``256``)
            n_heads (int): number of attention heads (default ``8``)
            d_ff (int): feed-forward inner dimension (default ``512``)
            dropout (float): dropout rate (default ``0.0``)
            activation (str): activation function to use in feed-forward network
                (default ``"relu"``). Choose from: ``"relu"``, ``"leakyrelu"``,
                ``"sigmoid"``, ``"tanh"``, ``"softplus"``.
        """
        super().__init__()
        self.dim_emb = dim_emb
        self.n_heads = n_heads
        self.norm1   = nn.LayerNorm(dim_emb)      # normalization before attention
        self.attn    = nn.MultiheadAttention(
            dim_emb,
            n_heads,
            dropout=dropout,
            batch_first=True,                     # note: batch_first=True for (B, T, d_model) input
        )
        self.norm2   = nn.LayerNorm(dim_emb)      # normalization before feed-forward
        self.ff      = nn.Sequential(
            nn.Linear(dim_emb, dim_ff),
            get_activation(activation),
            nn.Linear(dim_ff, dim_emb),
        )
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer layer.

        Args:
            inputs (torch.Tensor): shape ``(B, T, d_model)``.
            key_padding_mask (torch.Tensor, optional): shape ``(B, T)``,
                ``False`` = position to IGNORE.

        Returns:
            x (torch.Tensor): shape ``(B, T, d_model)`` transformed output
        """
        # self-attention with pre-norm
        residual = inputs           # save inputs for skip connection
        x = self.norm1(inputs)
        # avoid all False mask which causes attn to return nan; if all positions are masked
        safe_mask = key_padding_mask.clone()
        all_empty = ~key_padding_mask.any(dim=-1)   # (B,) True if all positions are masked
        # unmask the first position for batches where all are masked
        safe_mask[all_empty, 0] = True
        x, _ = self.attn(x, x, x, key_padding_mask=~safe_mask)
        x = self.drop(x) + residual

        # feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + residual

        return x


class AttentionPooling(nn.Module):
    """
    Attention pooling to produce a single global representation from
    the set of track embeddings.

    Attributes:
        d_in (int): input embedding dimension (from transformer)
    """
    def __init__(
        self,
        dim_in: int
    ):
        """
        Initialize the attention pooling layer.

        Args:
            d_in (int): input embedding dimension (from transformer)
        """
        super().__init__()
        self.query = nn.Linear(dim_in, 1)        # "score" for each track embedding

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the attention pooling layer.

        Args:
            x (torch.Tensor): shape ``(B, T, d_in)``, track embeddings from transformer
            padding_mask (torch.Tensor, optional): shape ``(B, T)``,
                ``True`` = real track, ``False`` = padding
        
        Returns:
            (torch.Tensor): shape ``(B, d_in)``, pooled jet representation
        """
        scores = self.query(x).squeeze(-1)          # (B, T), attention scores for each track
        if padding_mask is not None:
            # mask padded positions to -inf so that softmax gives zero weight to ignored tracks
            scores = scores.masked_fill(~padding_mask, float('-inf'))
        # handle case where all tracks are masked (e.g. no tracks) to avoid nan in softmax
        # (if all masked, set scores to zero)
        all_masked = (~padding_mask).all(dim=-1, keepdim=True)          # (B, 1)
        scores     = scores.masked_fill(all_masked.expand_as(scores), 0.0)  # uniform fallback
        weights    = torch.softmax(scores, dim=-1)
        pooled     = (weights.unsqueeze(-1) * x).sum(dim=1)                # (B, d_in)
        return pooled


def _mlp(
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: str = "relu",
) -> nn.Sequential:
    """
    Build an MLP with ReLU activations.
    
    Args:
        in_dim (int): input dimension
        hidden_dims (list[int]): list of hidden layer dimensions
        out_dim (int): output dimension
        activation (str): activation function to use in hidden layers (default ``"relu"``).
            Choose from: ``"relu"``, ``"leakyrelu"``, ``"sigmoid"``, ``"tanh"``, ``"softplus"``.

    Returns:
        nn.Sequential: the MLP model
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), get_activation(activation)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# GN2 model
# ---------------------------------------------------------------------------
class GN2(nn.Module):
    """
    GN2 transformer-based jet flavour tagger.

    Attributes:
        n_jet_vars (int): number of jet-level input features.
        n_track_vars (int): number of track-level input features.
        n_classes (int): number of jet flavour classes (default ``4``: b, c, light, tau).
        init_hidden_dim (int): hidden dimension of the per-track initialiser (default ``256``).
        init_output_dim (int): output dimension of the per-track initialiser (default ``256``).
        embed_dim (int): transformer embedding dimension (default ``256``).
        n_heads (int): number of attention heads (default ``8``).
        n_layers (int): number of transformer encoder layers (default ``4``).
        ff_dim (int): feed-forward inner dimension (default ``512``).
        pool_dim (int): output dimension of attention pooling (default ``128``).
        dropout (float): dropout rate (default ``0.0``).
        head_hidden_dims (list[int]): list of hidden layer dimensions for the task heads MLP
            (default ``[128, 64, 32]``).
        activation (str): activation function to use in MLPs and transformer (default ``"relu"``).
            Choose from: ``"relu"``, ``"leakyrelu"``, ``"sigmoid"``, ``"tanh"``, ``"softplus"``.
    """

    def __init__(
        self,
        n_jet_vars: int,
        n_track_vars: int,
        n_classes: int = 4,
        init_hidden_dim: int = 256,
        init_output_dim: int = 256,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        pool_dim: int = 128,
        dropout: float = 0.0,
        head_hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ):
        """
        Initialize the GN2 model.

        Args:
            n_jet_vars (int): number of jet-level input features.
            n_track_vars (int): number of track-level input features.
            n_classes (int, optional): number of jet flavour classes.
                (default ``4``: b, c, light, tau)
            init_hidden_dim (int, optional): hidden dimension of the per-track initialiser.
                (default ``256``)
            init_output_dim (int, optional): output dimension of the per-track initialiser.
                (default ``256``)
            embed_dim (int, optional): transformer embedding dimension. (default ``256``)
            n_heads (int, optional): number of attention heads. (default ``8``)
            n_layers (int, optional): number of transformer encoder layers. (default ``4``)
            ff_dim (int, optional): feed-forward inner dimension. (default ``512``)
            pool_dim (int, optional): output dimension of attention pooling. (default ``128``)
            dropout (float, optional): dropout rate. (default ``0.0``)
            head_hidden_dims (list[int], optional): hidden layer dimensions for the task heads MLP
                (default ``[128, 64, 32]``)
            activation (str, optional): activation function to use in MLPs and transformer.
                Choose from: ``"relu"``, ``"leakyrelu"``, ``"sigmoid"``, ``"tanh"``, ``"softplus"``.
                (default ``"relu"``)
        """
        super().__init__()

        self.n_jet_vars       = n_jet_vars
        self.n_track_vars     = n_track_vars
        self.n_classes        = n_classes
        self.embed_dim        = embed_dim
        self.pool_dim         = pool_dim
        self.head_hidden_dims = (
            list(head_hidden_dims) if head_hidden_dims is not None else [128, 64, 32]
        )

        in_dim = n_jet_vars + n_track_vars   # combined per-track input

        self.activation = activation

        # 1. Per-track initialiser (1 hidden layer + output, size `init_hidden_dim` and `init_output_dim`)
        self.track_init = nn.Sequential(
            nn.Linear(in_dim, init_hidden_dim),
            get_activation(activation),
            nn.Linear(init_hidden_dim, init_output_dim),
        )

        self.proj = nn.Linear(init_output_dim, embed_dim)

        # 2. Transformer encoder (`n_layers` layers, `n_heads` heads, pre-norm)
        self.transformer = nn.ModuleList([
            TransformerLayer(
                embed_dim,
                n_heads,
                ff_dim,
                dropout,
                activation,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)   # post-encoder norm

        # 3. Per-track projection (`embed_dim` -> `pool_dim`)
        self.track_proj = nn.Linear(embed_dim, pool_dim)

        # 4. Attention pooling (`pool_dim`)
        self.pool = AttentionPooling(pool_dim)

        # 5. Task heads (3 hidden layers: 128 -> 64 -> 32)
        self.jet_head = _mlp(pool_dim, self.head_hidden_dims, n_classes, activation)

        # count all trainable parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("GN2 initialised - embed=%s, layers=%s, heads=%s, ff=%s  |  params: %s",
                    embed_dim, n_layers, n_heads, ff_dim, f"{n_params:,}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> "GN2":
        """
        Load a GN2 model from a checkpoint file.

        Args:
            checkpoint_path (str | Path): path to the checkpoint file.
            device (torch.device): device to load the model on.

        Returns:
            model (GN2): GN2 model loaded with the checkpoint weights.
        
        Raises:
            OSError: if loading the checkpoint fails.
            KeyError: if expected keys are missing in the checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        config       = checkpoint["config"]
        model_config = config.get("model", {})
        data_config  = config.get("data", {})

        label_map = parse_label_map(data_config["label_map"])

        model = cls(
            n_jet_vars       = len(data_config["jet_features"]),
            n_track_vars     = len(data_config["track_features"]),
            n_classes        = len(label_map),
            init_hidden_dim  = model_config.get("initialiser_hidden_dim", 256),
            init_output_dim  = model_config.get("initialiser_output_dim", 256),
            embed_dim        = model_config.get("transformer_embed_dim", 256),
            n_heads          = model_config.get("transformer_n_heads", 8),
            n_layers         = model_config.get("transformer_n_layers", 4),
            ff_dim           = model_config.get("transformer_ff_dim", 512),
            pool_dim         = model_config.get("pooling_dim", 128),
            dropout          = model_config.get("transformer_dropout", 0.0),
            head_hidden_dims = model_config.get("head_hidden_dims", None),
            activation       = model_config.get("activation", "relu"),
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        logger.info(
            "GN2 loaded from %s (epoch %s, val_loss=%.4f)",
            checkpoint_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_loss", float("nan")),
        )
        return model

    def forward(
        self,
        jet_features: torch.Tensor,
        track_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the GN2 model.

        Args:
            jet_features (torch.Tensor): shape ``(B, n_jet_vars)``.
            track_features (torch.Tensor): shape ``(B, T, n_track_vars)``.
            mask (torch.Tensor): shape ``(B, T)``,  ``True`` = real track, ``False`` = padding.

        Returns:
            dict with keys:
              ``jet_outputs`` : (torch.Tensor): shape ``(B, n_classes)``
        """
        _, num_tracks, _ = track_features.shape

        # 1. Concatenate jet features (broadcast) with track features
        # unsqueeze jet features to (B, 1, J) and expand to (B, T, J) to
        # concatenate with track features (-1 = keep original size)
        jet_expanded = jet_features.unsqueeze(1).expand(-1, num_tracks, -1)   # (B, T, J)
        # concatenate along feature dimension
        combined     = torch.cat([jet_expanded, track_features], dim=-1)      # (B, T, J+K)

        # 2. Per-track initialisation
        x = self.track_init(combined)
        x = self.proj(x)                # (B, T, embed_dim)

        # 3. Transformer encoder
        for layer in self.transformer:
            x = layer(x, key_padding_mask=mask)
        x = self.final_norm(x)          # (B, T, embed_dim)

        # 4. Project track down to pool_dim
        x = self.track_proj(x)          # (B, T, pool_dim)

        # 5. Attention pooling: global jet representation
        jet_rep = self.pool(x, padding_mask=mask)   # (B, pool_dim)

        # 6. Primary head: jet classification
        jet_outputs = self.jet_head(jet_rep)         # (B, n_classes)

        return {
            "jet_outputs" : jet_outputs,
        }

    @torch.no_grad()        # for inference
    def predict_proba(
        self,
        jet_features: torch.Tensor,
        track_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class probabilities for the jet classification head.

        Args:
            jet_features (torch.Tensor): shape ``(B, n_jet_vars)``.
            track_features (torch.Tensor): shape ``(B, T, n_track_vars)``.
            mask (torch.Tensor): shape ``(B, T)``,  ``True`` = real track, ``False`` = padding.
        
        Returns:
            (torch.Tensor): shape ``(B, n_classes)``, predicted class probabilities.
        """
        self.eval()
        out = self.forward(jet_features, track_features, mask)
        return torch.softmax(out["jet_outputs"], dim=-1)

    @torch.no_grad()
    def discriminant_db(
        self,
        proba: torch.Tensor,
        fc: float = 0.2,
        ftau: float = 0.05,
        flavour_map: dict[str, int] = None,
    ) -> torch.Tensor:
        """
        Compute the b-tagging discriminant D_b:

        .. math::
            D_b = \\log\\left(\\frac{p_b}{f_c p_c + f_\\tau p_\\tau + (1-f_c-f_\\tau) p_u}\\right)

        Args:
            proba (torch.Tensor): shape ``(B, n_classes)``, predicted class probabilities.
            fc (float): contamination fraction of c-jets (default ``0.2``).
            ftau (float): contamination fraction of tau-jets (default ``0.05``).
            flavour_map (dict[str, int], optional): mapping ``{class_name: class_index}``.
                Must contain ``"b-jet"``, ``"c-jet"``, ``"light-jet"``, ``"tau-jet"``.
                Defaults to ``FLAVOUR_LABELS_INV``.

        Returns:
            (torch.Tensor): shape ``(B,)``, the b-tagging discriminant D_b.
        
        Raises:
            ValueError: if *flavour_map* is missing any of the required class names.
        """
        flavour_map = flavour_map if flavour_map is not None else FLAVOUR_LABELS_INV
        if not all(k in flavour_map for k in ("b-jet", "c-jet", "light-jet", "tau-jet")):
            raise ValueError("`flavour_map` must contain: "
                             "['b-jet', 'c-jet','light-jet', 'tau-jet']")
        pb    = proba[:, flavour_map["b-jet"]]
        pc    = proba[:, flavour_map["c-jet"]]
        pu    = proba[:, flavour_map["light-jet"]]
        ptau  = proba[:, flavour_map["tau-jet"]]
        denom = fc * pc + ftau * ptau + (1 - fc - ftau) * pu
        return torch.log((pb / denom).clamp(min=1e-8))

    @torch.no_grad()
    def discriminant_dc(
        self,
        proba: torch.Tensor,
        fb: float = 0.3,
        ftau: float = 0.01,
        flavour_map: dict[str, int] = None,
    ) -> torch.Tensor:
        """
        Compute the c-tagging discriminant D_c:

        .. math::
            D_c = \\log\\left(\\frac{p_c}{f_b p_b + f_\\tau p_\\tau + (1-f_b-f_\\tau) p_u}\\right)

        Args:
            proba (torch.Tensor): shape ``(B, n_classes)``, predicted class probabilities for
                the jet classification head.
            fb (float): contamination fraction of b-jets (default ``0.3``).
            ftau (float): contamination fraction of tau-jets (default ``0.01``).
            flavour_map (dict[str, int], optional): mapping ``{class_name: class_index}``.
                Must contain ``"b-jet"``, ``"c-jet"``, ``"light-jet"``, ``"tau-jet"``.
                Defaults to ``FLAVOUR_LABELS_INV``.

        Returns:
            (torch.Tensor): shape ``(B,)``, the c-tagging discriminant D_c.
        
        Raises:
            ValueError: if *flavour_map* is missing any of the required class names.
        """
        flavour_map = flavour_map if flavour_map is not None else FLAVOUR_LABELS_INV
        if not all(k in flavour_map for k in ("b-jet", "c-jet", "light-jet", "tau-jet")):
            raise ValueError("`flavour_map` must contain: "
                             "['b-jet', 'c-jet', 'light-jet', 'tau-jet']")
        pb    = proba[:, flavour_map["b-jet"]]
        pc    = proba[:, flavour_map["c-jet"]]
        pu    = proba[:, flavour_map["light-jet"]]
        ptau  = proba[:, flavour_map["tau-jet"]]
        denom = fb * pb + ftau * ptau + (1 - fb - ftau) * pu
        return torch.log((pc / denom).clamp(min=1e-8))
