import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

from models.mlp_predictor import CdPredictor


def train_mlp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    latent_dim: int = 25,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 1500,
    patience: int = 200,
    save_path: str = "checkpoints/best_cd_mlp.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> CdPredictor:
    """
    Trains the CdPredictor MLP on extracted latent representations with Early Stopping.

    Args:
        X_train (torch.Tensor): Training latent features [N_train, latent_dim].
        y_train (torch.Tensor): Training aerodynamic targets [N_train, 1].
        X_val (torch.Tensor): Validation latent features [N_val, latent_dim].
        y_val (torch.Tensor): Validation aerodynamic targets [N_val, 1].
        latent_dim (int): Dimensionality of latent embeddings.
        hidden_dim (int): Hidden units in MLP.
        dropout (float): Dropout probability.
        batch_size (int): Batch size for tabular latent data.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization weight decay.
        max_epochs (int): Maximum training epochs.
        patience (int): Early stopping patience epochs.
        save_path (str): Filepath to save best performing model weights.
        device (torch.device): Device on which training runs.

    Returns:
        CdPredictor: Trained model loaded with best validation weights.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = CdPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training CdPredictor (Latent Dim: {latent_dim}, Hidden: {hidden_dim}) on {device}...")

    for epoch in range(1, max_epochs + 1):
        # ---- Training Phase ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validation Phase ----
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_preds.append(pred.cpu())
                val_trues.append(yb.cpu())
        val_loss /= len(val_loader.dataset)

        # Compute Validation R2 score
        val_p_arr = torch.cat(val_preds).numpy().flatten()
        val_t_arr = torch.cat(val_trues).numpy().flatten()
        r2 = r2_score(val_t_arr, val_p_arr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            improved = "[Saved Best]"
        else:
            patience_counter += 1
            improved = ""

        if epoch % 50 == 0 or epoch == 1 or improved:
            print(f"Epoch {epoch:04d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val R2: {r2:.4f} {improved}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.5f}")
            break

    print(f"Training completed. Best checkpoint saved to {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Autoencoder Latent Embeddings")
    parser.add_argument("--latent_train_path", type=str, default="data/latent_train.pt", help="Path to saved latent train tensor")
    parser.add_argument("--latent_val_path", type=str, default="data/latent_val.pt", help="Path to saved latent val tensor")
    parser.add_argument("--latent_dim", type=int, default=25, help="Latent bottleneck dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden units")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Tabular batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=1500, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_cd_mlp.pt", help="Output model path")
    args = parser.parse_args()

    if os.path.exists(args.latent_train_path) and os.path.exists(args.latent_val_path):
        train_data = torch.load(args.latent_train_path, weights_only=False)
        val_data = torch.load(args.latent_val_path, weights_only=False)
        train_mlp(
            X_train=train_data["X"],
            y_train=train_data["y"],
            X_val=val_data["X"],
            y_val=val_data["y"],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            save_path=args.save_path,
        )
    else:
        print("Usage: Pass pre-extracted latent data tensors or call train_mlp() programmatically from your pipeline.")

