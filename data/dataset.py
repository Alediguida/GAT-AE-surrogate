import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from utils.vtk_processing import load_cd_map


class RefinedDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading pre-processed and subsampled aerodynamic surface graphs.

    Applies feature assembly (e.g. concatenating 3D pos and normals into node features x)
    and normalizes local nodal targets (pressure) and global aerodynamic coefficients (Cd).

    Attributes:
        files (List[str]): List of absolute filepaths to .pt graph files.
        cd_map (Dict[int, float]): Mapping from run IDs to ground-truth drag coefficients Cd.
        scaler_p (Optional[StandardScaler]): Fitted scaler for local nodal pressure values.
        scaler_cd (Optional[StandardScaler]): Fitted scaler for global drag coefficient Cd.
    """

    def __init__(
        self,
        files: List[str],
        cd_map: Optional[Dict[int, float]] = None,
        scaler_p: Optional[StandardScaler] = None,
        scaler_cd: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.files = [f for f in files if os.path.exists(f)]
        self.cd_map = cd_map if cd_map is not None else {}
        self.scaler_p = scaler_p
        self.scaler_cd = scaler_cd

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Data:
        file_path = self.files[idx]
        data: Data = torch.load(file_path, weights_only=False)

        # Assemble node feature matrix x from pos and norm if not already defined
        if not hasattr(data, "x") or data.x is None:
            data.x = torch.cat([data.pos, data.norm], dim=1)

        # Apply target normalization for nodal pressure (y_local)
        if self.scaler_p is not None and hasattr(data, "y_local") and data.y_local is not None:
            scaled_p = self.scaler_p.transform(data.y_local.numpy())
            data.y_local = torch.tensor(scaled_p, dtype=torch.float)

        # Parse run id and apply normalization for global drag coefficient (y_global)
        try:
            run_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
        except Exception:
            run_id = None

        if run_id is not None and run_id in self.cd_map:
            cd_raw = np.array([[self.cd_map[run_id]]], dtype=np.float32)
        elif hasattr(data, "y_global") and data.y_global is not None:
            cd_raw = data.y_global.numpy().reshape(1, 1)
        else:
            cd_raw = np.array([[0.0]], dtype=np.float32)

        if self.scaler_cd is not None:
            scaled_cd = self.scaler_cd.transform(cd_raw)
            data.y_global = torch.tensor(scaled_cd, dtype=torch.float)
        else:
            data.y_global = torch.tensor(cd_raw, dtype=torch.float)

        return data


def fit_scalers_on_training_set(
    train_files: List[str],
    cd_map: Optional[Dict[int, float]] = None,
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fits standard scalers exclusively on the training set to prevent data leakage.

    Args:
        train_files (List[str]): List of file paths belonging to the training partition.
        cd_map (Optional[Dict[int, float]]): Mapping of run IDs to Cd values.

    Returns:
        Tuple[StandardScaler, StandardScaler]: Fitted (scaler_p, scaler_cd).
    """
    scaler_p = StandardScaler()
    scaler_cd = StandardScaler()
    cd_values: List[float] = []

    print(f"Fitting scalers on {len(train_files)} training graphs...")
    for f in tqdm(train_files, desc="Fitting Scalers", leave=False):
        try:
            data: Data = torch.load(f, weights_only=False)
            if hasattr(data, "y_local") and data.y_local is not None:
                scaler_p.partial_fit(data.y_local.numpy())

            try:
                run_id = int(os.path.basename(f).split("_")[1].split(".")[0])
            except Exception:
                run_id = None

            if cd_map and run_id in cd_map:
                cd_values.append(cd_map[run_id])
            elif hasattr(data, "y_global") and data.y_global is not None:
                cd_values.append(float(data.y_global.item()))
        except Exception as e:
            print(f"Error reading {f} during scaler fitting: {e}")

    if len(cd_values) > 0:
        scaler_cd.fit(np.array(cd_values, dtype=np.float32).reshape(-1, 1))

    return scaler_p, scaler_cd


def create_dataloaders(
    data_dir: str,
    split_info_path: str,
    root_raw_dir: Optional[str] = None,
    batch_size: int = 12,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Loads dataset split info, fits scalers on the train partition, and instantiates DataLoaders.

    Args:
        data_dir (str): Directory containing preprocessed .pt graph files.
        split_info_path (str): Path to split_info.json containing 'train', 'val', 'test' lists.
        root_raw_dir (Optional[str]): Path to raw dataset root to parse Cd maps.
        batch_size (int): Batch size for train/val loaders.
        num_workers (int): DataLoader multiprocessing workers.
        pin_memory (bool): Whether to pin host memory for GPU transfers.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
            (train_loader, val_loader, test_loader, scaler_p, scaler_cd)
    """
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"split_info.json not found at: {split_info_path}")

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    train_files = [os.path.join(data_dir, os.path.basename(f)) for f in split_info["train"]]
    val_files = [os.path.join(data_dir, os.path.basename(f)) for f in split_info["val"]]
    test_files = [os.path.join(data_dir, os.path.basename(f)) for f in split_info["test"]]

    cd_map = load_cd_map(root_raw_dir) if root_raw_dir else {}
    scaler_p, scaler_cd = fit_scalers_on_training_set(train_files, cd_map)

    train_dataset = RefinedDataset(train_files, cd_map=cd_map, scaler_p=scaler_p, scaler_cd=scaler_cd)
    val_dataset = RefinedDataset(val_files, cd_map=cd_map, scaler_p=scaler_p, scaler_cd=scaler_cd)
    test_dataset = RefinedDataset(test_files, cd_map=cd_map, scaler_p=scaler_p, scaler_cd=scaler_cd)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"DataLoaders initialized. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, scaler_p, scaler_cd

