import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm
from torch_geometric.data import Data


class PhysicsScoutGATLite(nn.Module):
    """
    Lightweight Graph Attention Network (GAT) designed as a physics-guided scout model
    for surface mesh pressure field estimation.

    In the context of aerodynamic surrogate modeling, this network operates directly
    on 3D surface discretized meshes (e.g., vehicle half-body geometries) to rapidly
    predict point-wise surface pressure distributions from geometric features
    (3D coordinates and surface normal vectors). It leverages multi-head graph attention
    layers paired with GraphNorm to capture local surface curvature, flow alignment,
    and geometric spatial dependencies before projecting features through a non-linear
    regression head.

    Attributes:
        conv1 (GATConv): First multi-head graph attention convolutional layer.
        norm1 (GraphNorm): Graph normalization layer for the first stage.
        conv2 (GATConv): Second multi-head graph attention convolutional layer.
        norm2 (GraphNorm): Graph normalization layer for the second stage.
        conv3 (GATConv): Single-head graph attention convolutional layer aggregating features.
        norm3 (GraphNorm): Graph normalization layer for the bottleneck stage.
        pressure_head (nn.Sequential): Multi-layer perceptron mapping latent node features to local pressure scalar.
    """

    def __init__(self, in_features: int = 6, hidden_dim: int = 32, heads: int = 2):
        """
        Initializes the PhysicsScoutGATLite model.

        Args:
            in_features (int): Dimensionality of input node features (default: 6, representing pos [x, y, z] + norm [nx, ny, nz]).
            hidden_dim (int): Base hidden feature dimension for GAT layers (default: 32).
            heads (int): Number of multi-head attention mechanisms for intermediate layers (default: 2).
        """
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, concat=True)
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.norm2 = GraphNorm(hidden_dim * heads)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.norm3 = GraphNorm(hidden_dim)
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for predicting nodal pressure fields.

        Args:
            data (torch_geometric.data.Data): Graph batch containing:
                - data.pos (torch.Tensor): 3D node coordinates of shape [N, 3].
                - data.norm (torch.Tensor): Surface normal vectors of shape [N, 3].
                - data.edge_index (torch.Tensor): Graph adjacency indices of shape [2, E].

        Returns:
            torch.Tensor: Predicted nodal pressure tensor of shape [N, 1].
        """
        # Concatenate 3D spatial coordinates and normal vectors
        x = torch.cat([data.pos, data.norm], dim=1)
        edge_index = data.edge_index

        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.elu(self.norm2(self.conv2(x, edge_index)))
        x = F.elu(self.norm3(self.conv3(x, edge_index)))

        return self.pressure_head(x)

