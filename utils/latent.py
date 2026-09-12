import torch
from typing import Tuple
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader


def extract_latent_features(
    loader: DataLoader,
    encoder_model: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts global latent vectors (z) and corresponding aerodynamic ground-truth targets (y_global)
    from a graph dataset using a pre-trained, frozen autoencoder.

    Args:
        loader (DataLoader): PyTorch Geometric DataLoader providing batched graph Data objects.
        encoder_model (torch.nn.Module): Pre-trained GNN model with an `.encode()` method or forward pass returning latent z.
        device (torch.device): Device on which inference is performed (CPU or CUDA).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - X: Latent feature matrix of shape [N_samples, latent_dim].
            - y: Target values tensor of shape [N_samples, target_dim].
    """
    encoder_model.eval()
    x_list, y_list = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting Latent Space", leave=False):
            data = data.to(device)

            if hasattr(encoder_model, "encode"):
                z = encoder_model.encode(data)
            elif hasattr(encoder_model, "forward"):
                out = encoder_model(data, return_latent=True)
                z = out[1] if isinstance(out, tuple) else out
            else:
                raise AttributeError("The encoder model must implement either an 'encode' or 'forward' method.")

            x_list.append(z.cpu())
            if hasattr(data, "y_global") and data.y_global is not None:
                y_list.append(data.y_global.cpu())
            elif hasattr(data, "y") and data.y is not None:
                y_list.append(data.y.cpu())

    x_tensor = torch.cat(x_list, dim=0) if len(x_list) > 0 else torch.empty(0)
    y_tensor = torch.cat(y_list, dim=0) if len(y_list) > 0 else torch.empty(0)

    return x_tensor, y_tensor

