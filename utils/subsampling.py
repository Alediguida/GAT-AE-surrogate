import numpy as np
import torch
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from utils.vtk_processing import load_vtk_mesh, compute_vtk_gradient


def robust_normalize(x: np.ndarray, lower_percentile: float = 0.0, upper_percentile: float = 99.9) -> np.ndarray:
    """
    Robustly scales an array into the [0, 1] range using percentiles to mitigate outlier distortion.

    Args:
        x (np.ndarray): Input numpy array.
        lower_percentile (float): Lower percentile threshold (default: 0.0, minimum).
        upper_percentile (float): Upper percentile threshold (default: 99.9).

    Returns:
        np.ndarray: Normalized array clipped to [0, 1].
    """
    p_high = np.percentile(x, upper_percentile)
    p_low = x.min() if lower_percentile == 0.0 else np.percentile(x, lower_percentile)

    if p_high - p_low < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - p_low) / (p_high - p_low), 0.0, 1.0)


def compute_physical_importance(
    score_abs: np.ndarray,
    score_grad: np.ndarray,
    weight_abs: float = 0.80,
    weight_grad: float = 0.20,
    exaggeration_power: float = 5.0,
) -> np.ndarray:
    """
    Computes a non-linear physical importance metric combining absolute scalar values and gradient magnitudes.

    Args:
        score_abs (np.ndarray): Normalized magnitude of local physical scalar (e.g. pressure).
        score_grad (np.ndarray): Normalized magnitude of physical field spatial gradients.
        weight_abs (float): Weight assigned to absolute value (default: 0.80).
        weight_grad (float): Weight assigned to gradient intensity (default: 0.20).
        exaggeration_power (float): Non-linear exponent to accentuate high-importance peaks (default: 5.0).

    Returns:
        np.ndarray: Calculated physical importance scores.
    """
    raw_importance = (weight_abs * score_abs) + (weight_grad * score_grad)
    return raw_importance ** exaggeration_power


def subsample_by_importance(
    pos: np.ndarray,
    norm: np.ndarray,
    y: np.ndarray,
    importance: np.ndarray,
    p_low: float = 40.0,
    p_high: float = 80.0,
    ratio_high: float = 1.00,
    ratio_med: float = 0.20,
    ratio_low: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs tiered subsampling based on physical importance percentiles.

    Args:
        pos (np.ndarray): 3D coordinates [N, 3].
        norm (np.ndarray): Normal vectors [N, 3].
        y (np.ndarray): Ground truth values [N,].
        importance (np.ndarray): Importance weights [N,].
        p_low (float): Low importance percentile threshold (default: 40.0).
        p_high (float): High importance percentile threshold (default: 80.0).
        ratio_high (float): Fraction of top tier nodes to retain (default: 1.0 = 100%).
        ratio_med (float): Fraction of mid tier nodes to retain (default: 0.2 = 20%).
        ratio_low (float): Fraction of low tier nodes to retain (default: 0.01 = 1%).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - pos_sub: Subsampled node coordinates.
            - norm_sub: Subsampled normal vectors.
            - y_sub: Subsampled target values.
            - imp_sub: Subsampled importance scores.
            - final_indices: Retained vertex indices from original mesh.
    """
    th_low = np.percentile(importance, p_low)
    th_high = np.percentile(importance, p_high)

    low_idx = np.where(importance <= th_low)[0]
    med_idx = np.where((importance > th_low) & (importance <= th_high))[0]
    high_idx = np.where(importance > th_high)[0]

    # Sampling each tier
    sel_high = high_idx if ratio_high >= 1.0 else np.random.choice(high_idx, max(1, int(len(high_idx) * ratio_high)), replace=False)
    sel_med = np.random.choice(med_idx, max(1, int(len(med_idx) * ratio_med)), replace=False) if len(med_idx) > 0 else np.empty(0, dtype=int)
    sel_low = np.random.choice(low_idx, max(1, int(len(low_idx) * ratio_low)), replace=False) if len(low_idx) > 0 else np.empty(0, dtype=int)

    final_idx = np.sort(np.concatenate([sel_high, sel_med, sel_low]).astype(int))

    return pos[final_idx], norm[final_idx], y[final_idx], importance[final_idx], final_idx


def build_knn_graph(pos: np.ndarray, k_neighbors: int = 6) -> torch.Tensor:
    """
    Constructs a k-Nearest Neighbors graph from 3D spatial coordinates.

    Args:
        pos (np.ndarray): Node coordinates of shape [N, 3].
        k_neighbors (int): Number of nearest neighbors per node (default: 6).

    Returns:
        torch.Tensor: PyTorch LongTensor of shape [2, E] representing edge indices.
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto", n_jobs=1).fit(pos)
    adj = nbrs.kneighbors_graph(mode="connectivity").tocoo()
    return torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)


def refine_mesh_with_scout(
    vtp_path: str,
    cd_val: float,
    scout_model: torch.nn.Module,
    device: torch.device,
    k_neighbors_scout: int = 6,
    k_neighbors_final: int = 6,
    weight_abs: float = 0.80,
    weight_grad: float = 0.20,
    exaggeration_power: float = 5.0,
    p_low: float = 40.0,
    p_high: float = 80.0,
    ratio_high: float = 1.00,
    ratio_med: float = 0.20,
    ratio_low: float = 0.01,
) -> Data:
    """
    Refines a raw surface mesh into a prioritized, subsampled graph using a pre-trained Scout GNN.

    Args:
        vtp_path (str): Filepath to the raw .vtp or .vtu surface mesh.
        cd_val (float): Aerodynamic drag coefficient for this run.
        scout_model (torch.nn.Module): Trained Scout GNN model.
        device (torch.device): Compute device (CPU or CUDA).
        k_neighbors_scout (int): Number of neighbors for initial scout inference graph.
        k_neighbors_final (int): Number of neighbors for subsampled final graph.

    Returns:
        Data: Refined PyTorch Geometric Data object.
    """
    pos_full, norm_full, _, y_full_truth, polydata_full = load_vtk_mesh(vtp_path)

    # 1. Scout Inference on Full Mesh
    edge_index_full = build_knn_graph(pos_full, k_neighbors=k_neighbors_scout).to(device)
    pos_full_t = torch.tensor(pos_full, dtype=torch.float, device=device)
    norm_full_t = torch.tensor(norm_full, dtype=torch.float, device=device)

    scout_model.eval()
    with torch.no_grad():
        dummy_data = Data(pos=pos_full_t, norm=norm_full_t, edge_index=edge_index_full)
        p_pred_scout = scout_model(dummy_data)

    p_interp_full = p_pred_scout.cpu().numpy().flatten()

    # 2. Physical Importance Computation (Pressure Magnitude + Spatial Gradient)
    score_abs = robust_normalize(np.abs(p_interp_full))
    grad_mag_full = compute_vtk_gradient(polydata_full, p_interp_full, "PredPressure")
    score_grad = robust_normalize(grad_mag_full)

    importance = compute_physical_importance(
        score_abs=score_abs,
        score_grad=score_grad,
        weight_abs=weight_abs,
        weight_grad=weight_grad,
        exaggeration_power=exaggeration_power,
    )

    # 3. Aggressive Tiered Subsampling
    pos_fin, norm_fin, y_fin, imp_fin, _ = subsample_by_importance(
        pos=pos_full,
        norm=norm_full,
        y=y_full_truth,
        importance=importance,
        p_low=p_low,
        p_high=p_high,
        ratio_high=ratio_high,
        ratio_med=ratio_med,
        ratio_low=ratio_low,
    )

    # 4. Final k-NN Graph Connectivity
    edge_index_fin = build_knn_graph(pos_fin, k_neighbors=k_neighbors_final)

    return Data(
        pos=torch.tensor(pos_fin, dtype=torch.float),
        norm=torch.tensor(norm_fin, dtype=torch.float),
        y_local=torch.tensor(y_fin, dtype=torch.float).view(-1, 1),
        y_global=torch.tensor([[cd_val]], dtype=torch.float),
        edge_index=edge_index_fin,
        importance=torch.tensor(imp_fin, dtype=torch.float).view(-1, 1),
    )

