import torch
import torch.nn as nn


class CdPredictor(nn.Module):
    """
    Multi-Layer Perceptron (MLP) regressor for predicting global aerodynamic coefficients
    (e.g., drag coefficient Cd) directly from a compact latent representation (z).

    This model takes the graph-level latent embedding produced by the frozen GAT Autoencoder
    encoder and maps it through non-linear layers with batch normalization and dropout
    regularization to estimate macroscopic aerodynamic quantities.

    Attributes:
        net (nn.Sequential): Sequential feed-forward layers mapping latent_dim -> 1.
    """

    def __init__(self, latent_dim: int = 25, hidden_dim: int = 64, dropout: float = 0.2):
        """
        Initializes the CdPredictor.

        Args:
            latent_dim (int): Dimensionality of the input latent vector z (default: 25).
            hidden_dim (int): Number of hidden units in the first dense layer (default: 64).
            dropout (float): Dropout rate for regularization (default: 0.2).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z (torch.Tensor): Latent representation tensor of shape [batch_size, latent_dim].

        Returns:
            torch.Tensor: Predicted aerodynamic drag coefficient of shape [batch_size, 1].
        """
        return self.net(z)

