import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class PressureAutoencoderGNN(nn.Module):
    """
    Graph Attention Network Autoencoder (GAT-AE) for mesh-based physical field estimation.

    This model compresses discretized surface mesh representations into a compact global
    latent representation (z) and reconstructs point-wise scalar physical fields
    (e.g., surface pressure distributions).

    Architecture Overview:
        - Encoder: Multi-layer GAT convolutions with multi-head attention and batch
          normalization capturing localized geometric surface features.
        - Bottleneck: Combined global mean and max graph pooling followed by an MLP
          latent projector producing a compact latent vector z.
        - Decoder: Latent broadcast across nodes concatenated with encoder skip connections,
          processed by GAT layers and an output projection head.

    Attributes:
        conv1 (GATConv): First encoder graph attention convolution.
        bn1 (nn.BatchNorm1d): Batch normalization layer for first encoder stage.
        conv2 (GATConv): Second encoder graph attention convolution.
        bn2 (nn.BatchNorm1d): Batch normalization layer for second encoder stage.
        enc_head (nn.Sequential): MLP projecting pooled graph features into latent space.
        dec_entry (nn.Sequential): MLP lifting latent vector to node feature dimension.
        dec_conv1 (GATConv): First decoder graph attention convolution with skip features.
        dec_bn1 (nn.BatchNorm1d): Batch normalization layer for decoder stage.
        dec_conv2 (GATConv): Final decoder graph attention aggregation layer.
        out_local (nn.Linear): Linear projection layer predicting local scalar values (e.g. pressure).
    """

    def __init__(
        self,
        in_features: int = 6,
        latent_dim: int = 25,
        out_local: int = 1,
        heads: int = 2,
        hidden: int = 64,
        dropout: float = 0.05,
    ):
        """
        Initializes the PressureAutoencoderGNN.

        Args:
            in_features (int): Number of input node features (e.g., 6 for 3D pos + normals, or 4 for 2D pos + normals).
            latent_dim (int): Dimensionality of the global latent bottleneck vector z.
            out_local (int): Dimensionality of local nodal target (default: 1 for surface pressure).
            heads (int): Number of attention heads for GAT layers.
            hidden (int): Base hidden feature dimension across GAT layers.
            dropout (float): Dropout probability applied in GAT layers.
        """
        super().__init__()
        self.dropout = dropout

        # ---- ENCODER ----
        self.conv1 = GATConv(in_features, hidden, heads=heads, dropout=self.dropout)
        self.bn1 = nn.BatchNorm1d(hidden * heads)

        self.conv2 = GATConv(hidden * heads, hidden * 2, heads=heads, dropout=self.dropout)
        self.bn2 = nn.BatchNorm1d(hidden * 2 * heads)

        # Bottleneck Latent Projector (Mean + Max pooling concatenation)
        input_dim_head = (hidden * 2 * heads) * 2
        self.enc_head = nn.Sequential(
            nn.Linear(input_dim_head, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # ---- DECODER ----
        self.dec_entry = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden * heads),
        )

        # Concatenation of broadcasted latent (hidden * heads) + skip connection x1 (hidden * heads)
        self.dec_conv1 = GATConv(hidden * heads * 2, hidden, heads=heads, dropout=self.dropout)
        self.dec_bn1 = nn.BatchNorm1d(hidden * heads)

        self.dec_conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=self.dropout)
        self.out_local = nn.Linear(hidden, out_local)

    def encode(self, data: Data) -> torch.Tensor:
        """
        Encodes graph into a compact latent representation z.

        Args:
            data (torch_geometric.data.Data): Input graph batch.

        Returns:
            torch.Tensor: Global latent tensor z of shape [batch_size, latent_dim].
        """
        x = self._get_node_features(data)
        edge_index = data.edge_index
        batch = self._get_batch(data, x)

        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))

        g = torch.cat([global_mean_pool(x2, batch), global_max_pool(x2, batch)], dim=1)
        z = self.enc_head(g)
        return z

    def forward(self, data: Data, return_latent: bool = False):
        """
        Forward pass predicting reconstructed point-wise scalar fields.

        Args:
            data (torch_geometric.data.Data): Input graph batch containing node features and edge indices.
            return_latent (bool): If True, returns tuple (out_p, z); otherwise returns out_p.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Predicted local field [N, out_local]
            and optionally latent code z [batch_size, latent_dim].
        """
        x = self._get_node_features(data)
        edge_index = data.edge_index
        batch = self._get_batch(data, x)

        # 1. Encoder
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))

        # 2. Global Pooling & Latent Bottleneck
        g = torch.cat([global_mean_pool(x2, batch), global_max_pool(x2, batch)], dim=1)
        z = self.enc_head(g)

        # 3. Decoder with Skip Connection from x1
        z_exp = self.dec_entry(z)[batch]
        x_cat = torch.cat([z_exp, x1], dim=1)

        xd = F.elu(self.dec_bn1(self.dec_conv1(x_cat, edge_index)))
        xd = F.elu(self.dec_conv2(xd, edge_index))
        out_p = self.out_local(xd)

        if return_latent:
            return out_p, z
        return out_p

    def _get_node_features(self, data: Data) -> torch.Tensor:
        """Helper to extract or construct node feature matrix."""
        if hasattr(data, "x") and data.x is not None:
            return data.x
        if hasattr(data, "pos") and hasattr(data, "norm") and data.norm is not None:
            return torch.cat([data.pos, data.norm], dim=1)
        if hasattr(data, "pos"):
            return data.pos
        raise ValueError("Data object must contain 'x' or 'pos' (and optional 'norm') attributes.")

    def _get_batch(self, data: Data, x: torch.Tensor) -> torch.Tensor:
        """Helper to safely retrieve batch vector for single or batched graphs."""
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)


class MultiTaskGNN(nn.Module):
    """
    Multi-Task Graph Attention Network for simultaneous local field reconstruction
    and global aerodynamic coefficient prediction (e.g. drag coefficient Cd).

    Attributes:
        conv_enc1 (GATConv): First encoder GAT layer.
        bn_enc1 (nn.BatchNorm1d): First encoder BatchNorm.
        conv_enc2 (GATConv): Second encoder GAT layer.
        bn_enc2 (nn.BatchNorm1d): Second encoder BatchNorm.
        encoder_head (nn.Sequential): Projector into global latent representation z.
        decoder_entry (nn.Sequential): Expands latent representation back to node features.
        conv_dec1 (GATConv): Decoder GAT layer with skip connection from encoder layer 1.
        bn_dec1 (nn.BatchNorm1d): Decoder BatchNorm.
        conv_dec2 (GATConv): Final decoder GAT layer.
        output_local (nn.Linear): Linear layer for node-level physical scalar prediction.
        drag_head (nn.Sequential): Deep MLP mapping latent representation z to global drag Cd.
    """

    def __init__(
        self,
        in_features: int = 6,
        latent_dim: int = 25,
        out_features_local: int = 1,
        heads: int = 4,
        hidden_internal: int = 48,
        dropout_rate: float = 0.025,
    ):
        """
        Initializes the MultiTaskGNN.

        Args:
            in_features (int): Number of input node features.
            latent_dim (int): Dimensionality of the global latent space z.
            out_features_local (int): Number of output features per node (e.g., pressure = 1).
            heads (int): Number of attention heads for GAT layers.
            hidden_internal (int): Base internal feature dimension.
            dropout_rate (float): Dropout probability for attention convolutions.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # ---- ENCODER (GAT) ----
        self.conv_enc1 = GATConv(in_features, hidden_internal, heads=heads, dropout=self.dropout_rate)
        self.bn_enc1 = nn.BatchNorm1d(hidden_internal * heads)

        self.conv_enc2 = GATConv(hidden_internal * heads, hidden_internal * 2, heads=heads, dropout=self.dropout_rate)
        self.bn_enc2 = nn.BatchNorm1d(hidden_internal * 2 * heads)

        # ---- LATENT PROJECTOR ----
        input_dim_head = (hidden_internal * 2 * heads) * 2
        self.encoder_head = nn.Sequential(
            nn.Linear(input_dim_head, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # ---- DECODER ----
        self.decoder_entry = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_internal * heads),
        )

        self.conv_dec1 = GATConv(hidden_internal * heads * 2, hidden_internal, heads=heads, dropout=self.dropout_rate)
        self.bn_dec1 = nn.BatchNorm1d(hidden_internal * heads)

        self.conv_dec2 = GATConv(hidden_internal * heads, hidden_internal, heads=1, dropout=self.dropout_rate)
        self.output_local = nn.Linear(hidden_internal, out_features_local)

        # ---- GLOBAL DRAG HEAD ----
        self.drag_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data):
        """
        Forward pass predicting local field, global scalar, and latent vector.

        Args:
            data (torch_geometric.data.Data): Graph batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out_p: Predicted point-wise field of shape [N, out_features_local].
                - out_cd: Predicted global scalar (e.g. Cd) of shape [batch_size, 1].
                - z: Latent embedding vector of shape [batch_size, latent_dim].
        """
        x = self._get_node_features(data)
        edge_index = data.edge_index
        batch = self._get_batch(data, x)

        # 1. ENCODER
        x1 = F.elu(self.bn_enc1(self.conv_enc1(x, edge_index)))
        x2 = F.elu(self.bn_enc2(self.conv_enc2(x1, edge_index)))

        # 2. POOLING (Mean + Max)
        g_mean = global_mean_pool(x2, batch)
        g_max = global_max_pool(x2, batch)

        x_global = torch.cat([g_mean, g_max], dim=1)
        z = self.encoder_head(x_global)

        # 3. TASKS
        out_cd = self.drag_head(z)

        z_expanded = self.decoder_entry(z)
        z_nodes = z_expanded[batch]
        x_cat = torch.cat([z_nodes, x1], dim=1)

        xd = F.elu(self.bn_dec1(self.conv_dec1(x_cat, edge_index)))
        xd = F.elu(self.conv_dec2(xd, edge_index))
        out_p = self.output_local(xd)

        return out_p, out_cd, z

    def _get_node_features(self, data: Data) -> torch.Tensor:
        """Helper to extract or construct node feature matrix."""
        if hasattr(data, "x") and data.x is not None:
            return data.x
        if hasattr(data, "pos") and hasattr(data, "norm") and data.norm is not None:
            return torch.cat([data.pos, data.norm], dim=1)
        if hasattr(data, "pos"):
            return data.pos
        raise ValueError("Data object must contain 'x' or 'pos' (and optional 'norm') attributes.")

    def _get_batch(self, data: Data, x: torch.Tensor) -> torch.Tensor:
        """Helper to safely retrieve batch vector for single or batched graphs."""
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)



class PhysicsScoutGATLite(nn.Module):
    def __init__(self, in_features=2, hidden_dim=32, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, dropout=0.1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.1)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x if hasattr(data, 'x') and data.x is not None else data.pos
        edge_index = data.edge_index
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.conv2(x, edge_index))
        return self.out(x)
