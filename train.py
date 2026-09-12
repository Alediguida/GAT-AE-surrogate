import os
import argparse
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from models.gnn_autoencoder import PressureAutoencoderGNN
from data.dataset import create_dataloaders


def custom_punitive_loss(pred: torch.Tensor, target: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
    """
    Computes a weighted MSE loss for the Autoencoder.
    Uses the importance map raised to the third power as a multiplicative factor.

    Args:
        pred (torch.Tensor): Predicted field [N, 1].
        target (torch.Tensor): Ground-truth field [N, 1].
        importance (torch.Tensor): Physical importance scores from the Scout model [N, 1].

    Returns:
        torch.Tensor: Weighted scalar loss.
    """
    weights = importance ** 3
    loss = (((pred - target) ** 2) * weights).mean()
    return loss


def train_autoencoder(
    data_dir: str,
    split_info_path: str,
    in_features: int = 6,
    latent_dim: int = 25,
    hidden_dim: int = 64,
    batch_size: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 10,
    save_path: str = "checkpoints/pure_ae_model.pt",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Trains the PressureAutoencoderGNN using a physics-informed punitive loss.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # 1. Load Data
    print("Initializing Dataloaders...")
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        data_dir=data_dir,
        split_info_path=split_info_path,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # 2. Initialize Model
    model = PressureAutoencoderGNN(
        in_features=in_features, 
        latent_dim=latent_dim, 
        out_local=1, 
        heads=2, 
        hidden=hidden_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting PUNITIVE Autoencoder Training on {device}...")

    for epoch in range(1, max_epochs + 1):
        # ---- Training Phase ----
        model.train()
        train_loss_tot = 0.0

        pbar = tqdm(train_loader, leave=False, desc=f"Ep {epoch}")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()

            out_p = model(data)
            
            # Apply punitive loss
            loss = custom_punitive_loss(out_p, data.y_local, data.importance)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_tot += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_tot / len(train_loader)

        # ---- Validation Phase ----
        model.eval()
        val_loss_tot = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out_p = model(data)
                val_loss = custom_punitive_loss(out_p, data.y_local, data.importance)
                val_loss_tot += val_loss.item()

        avg_val_loss = val_loss_tot / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]

        # Save Best Model
        msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            msg = "BEST VAL"
        else:
            epochs_no_improve += 1

        print(f"Ep {epoch:03d} | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {msg}")

        if epochs_no_improve >= 25:
            print(f"Early stopping triggered! Val loss did not improve for 25 epochs.")
            break
        if current_lr < 1e-4:
            print(f"Early stopping triggered! Learning rate dropped below 1e-4.")
            break

    print(f"Autoencoder Training Completed! Best model saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT Autoencoder with Punitive Loss")
    parser.add_argument("--data_dir", type=str, default="data/refined_graphs", help="Directory containing .pt graph files")
    parser.add_argument("--split_info", type=str, default="data/refined_graphs/split_info.json", help="Path to split_info.json")
    parser.add_argument('--in_features', type=int, default=4, help="Input features per node")
    parser.add_argument('--latent_dim', type=int, default=10, help="Latent space dimension")
    parser.add_argument("--batch_size", type=int, default=12, help="Graph batch size")
    parser.add_argument("--max_epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--save_path", type=str, default="checkpoints/pure_ae_model.pt", help="Output model path")
    
    args = parser.parse_args()

    # Pre-flight check
    if os.path.exists(args.data_dir) and os.path.exists(args.split_info):
        train_autoencoder(
            data_dir=args.data_dir,
            split_info_path=args.split_info,
            in_features=args.in_features,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            save_path=args.save_path
        )
    else:
        print(f"💡 Usage: Ensure data directory '{args.data_dir}' and split info '{args.split_info}' exist.")
        print("💡 E.g.: python train.py --data_dir /path/to/graphs --split_info /path/to/split.json")

