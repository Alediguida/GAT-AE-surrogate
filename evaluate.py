import os
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models.gnn_autoencoder import PressureAutoencoderGNN
from models.mlp_predictor import CdPredictor
from utils.latent import extract_latent_features
from utils.metrics_plotting import compute_regression_metrics, plot_parity


def evaluate_pipeline(
    ae_model: PressureAutoencoderGNN,
    mlp_model: CdPredictor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    scaler=None,
    save_plot_path: str = "plots/parity_cd_prediction.png",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Evaluates the combined Autoencoder + Latent MLP pipeline on Validation and Test sets.

    Args:
        ae_model (PressureAutoencoderGNN): Pre-trained GNN autoencoder for latent extraction.
        mlp_model (CdPredictor): Trained MLP regressor mapping latent z -> Cd.
        val_loader (DataLoader): PyG DataLoader for validation graph instances.
        test_loader (DataLoader): PyG DataLoader for test graph instances.
        scaler: Optional fitted scikit-learn scaler with .inverse_transform().
        save_plot_path (str): Filepath to save parity plot.
        device (torch.device): Compute device.
    """
    ae_model.to(device).eval()
    mlp_model.to(device).eval()

    def run_inference(loader: DataLoader, name: str):
        print(f"🔍 Extracting and predicting on {name} set...")
        x_latent, y_true = extract_latent_features(loader, ae_model, device)
        x_latent = x_latent.to(device)

        with torch.no_grad():
            y_pred = mlp_model(x_latent).cpu().numpy()

        y_true = y_true.numpy()

        if scaler is not None:
            y_pred = scaler.inverse_transform(y_pred).flatten()
            y_true = scaler.inverse_transform(y_true).flatten()
        else:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()

        metrics = compute_regression_metrics(y_true, y_pred)
        print(f"  • {name} Results: R2 = {metrics['r2']:.4f} | RMSE = {metrics['rmse']:.4f} | MAE = {metrics['mae']:.4f}")
        return y_true, y_pred, metrics

    val_true, val_pred, _ = run_inference(val_loader, "Validation")
    test_true, test_pred, _ = run_inference(test_loader, "Test")

    # Generate Parity Plot
    plot_parity(
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        title="Latent Space Drag Prediction",
        xlabel="True $C_d$ (CFD)",
        ylabel="Predicted $C_d$ (Latent-MLP)",
        save_path=save_plot_path,
        show=True,
    )
    print(f"📊 Parity plot successfully generated and saved to '{save_plot_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN-AE + MLP Drag Predictor")
    parser.add_argument("--ae_checkpoint", type=str, default="checkpoints/pure_ae_model.pt", help="Path to AE weights")
    parser.add_argument("--mlp_checkpoint", type=str, default="checkpoints/best_cd_mlp.pt", help="Path to MLP weights")
    parser.add_argument("--latent_dim", type=int, default=25, help="Latent bottleneck dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="MLP hidden dimension")
    parser.add_argument("--save_plot", type=str, default="plots/parity_cd_prediction.png", help="Plot save path")
    args = parser.parse_args()

    print("💡 Evaluation module initialized. Run with loaded datasets or call evaluate_pipeline() programmatically.")

