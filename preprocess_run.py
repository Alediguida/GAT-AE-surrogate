import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from utils.vtk_processing import load_vtk_mesh

def preprocess_vtu_dataset(
    raw_dir: str = "data/dataset_run",
    out_dir: str = "data/processed_data",
    k_neighbors: int = 6
):
    """
    Parses OpenFOAM VTU volume meshes, extracts 2D coords, builds KNN graphs, 
    and saves them as PyG .pt files. (FULL GRAPHS FOR SCOUT)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(raw_dir, "dataset_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} runs in CSV. Starting conversion to graphs...")
    
    processed_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        run_id = int(row['run_id'])
        cd_val = float(row['Cd'])
        
        vtu_path = os.path.join(raw_dir, f"run_{run_id}_field.vtu")
        if not os.path.exists(vtu_path):
            continue
            
        try:
            # 1. Load VTU file (extracts position and pressure)
            pos, norm, area, p_truth, _ = load_vtk_mesh(vtu_path)
            
            # 2. Extract 2D coordinates and generate k-NN Graph
            pos_2d = pos[:, :2]
            n_points = len(pos_2d)
            
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', n_jobs=1).fit(pos_2d)
            adj = nbrs.kneighbors_graph(mode='connectivity').tocoo()
            edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
            
            # 3. Tensor Creation
            # in_features=2 (x, y)
            x_features = torch.tensor(pos_2d, dtype=torch.float)
            y_local = torch.tensor(p_truth, dtype=torch.float).view(-1, 1)
            y_global = torch.tensor([[cd_val]], dtype=torch.float)
            
            # Dummy importance vector (1.0 everywhere)
            importance = torch.ones((n_points, 1), dtype=torch.float)
            
            data = Data(
                x=x_features,
                edge_index=edge_index,
                y_local=y_local,
                y_global=y_global,
                importance=importance,
                pos=torch.tensor(pos_2d, dtype=torch.float)
            )
            
            # 4. Save to disk
            out_file = f"run_{run_id}.pt"
            torch.save(data, os.path.join(out_dir, out_file))
            processed_files.append(out_file)
            
        except Exception as e:
            print(f"Error on run_{run_id}: {e}")
            
    # 5. Generate split_info.json (80% train, 10% val, 10% test)
    import random
    random.seed(42)
    random.shuffle(processed_files)
    
    n = len(processed_files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    split_info = {
        "train": processed_files[:n_train],
        "val": processed_files[n_train:n_train+n_val],
        "test": processed_files[n_train+n_val:]
    }
    
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=4)
        
    print(f"Conversion complete! {n} graphs saved in {out_dir}")

if __name__ == "__main__":
    preprocess_vtu_dataset()
