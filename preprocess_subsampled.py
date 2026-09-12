import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from utils.vtk_processing import load_vtk_mesh, compute_vtk_gradient
from models.gnn_autoencoder import PhysicsScoutGATLite

def robust_normalize(x):
    p99 = np.percentile(x, 99.9)
    p01 = x.min()
    if p99 - p01 < 1e-8: return np.zeros_like(x)
    return np.clip((x - p01) / (p99 - p01), 0, 1)

def run_subsampling(
    in_dir: str = "data/processed_data",
    out_dir: str = "data/refined_graphs",
    raw_dir: str = "data/dataset_run",
    scout_path: str = "checkpoints/scout_model.pt",
    k_neighbors: int = 6
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load trained Scout model
    if not os.path.exists(scout_path):
        print(f"Error: Scout model not found in {scout_path}. Train it first!")
        return
        
    print(f"Loading Scout model from {scout_path}...")
    scout = PhysicsScoutGATLite(in_features=2, hidden_dim=32, heads=2).to(device)
    scout.load_state_dict(torch.load(scout_path, map_location=device, weights_only=True))
    scout.eval()
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(in_dir, "split_info.json"), "r") as f:
        split_info = json.load(f)
    
    all_files = split_info["train"] + split_info["val"] + split_info["test"]
    
    for pt_file in tqdm(all_files, desc="Generating Subsampled Graphs"):
        run_id = pt_file.split('_')[1].split('.')[0]
        in_path = os.path.join(in_dir, pt_file)
        
        # Load FULL graph
        data_full = torch.load(in_path, weights_only=False).to(device)
        
        # 1. Scout Prediction
        with torch.no_grad():
            p_pred_scout = scout(data_full).cpu().numpy().flatten()
            
        # 2. Compute VTK Gradients (fetching original mesh)
        vtu_path = os.path.join(raw_dir, f"run_{run_id}_field.vtu")
        _, _, _, _, polydata_full = load_vtk_mesh(vtu_path)
        
        score_abs = robust_normalize(np.abs(p_pred_scout))
        grad_mag_full = compute_vtk_gradient(polydata_full, p_pred_scout, "PredPressure")
        score_grad = robust_normalize(grad_mag_full)
        
        # 50% gradient, 50% absolute value
        raw_importance = (0.5 * score_abs) + (0.5 * score_grad)
        
        # 3. Subsampling Logic (Top 30% per 10k nodi -> ~3000 nodi fitti, + random background)
        p_low = np.percentile(raw_importance, 40)
        p_high = np.percentile(raw_importance, 70)
        
        low_idx = np.where(raw_importance <= p_low)[0]
        med_idx = np.where((raw_importance > p_low) & (raw_importance <= p_high))[0]
        high_idx = np.where(raw_importance > p_high)[0]
        
        sel_high = high_idx # Keep all top 30%
        sel_med = np.random.choice(med_idx, int(len(med_idx) * 0.30), replace=False) # 30% of the middle zone
        sel_low = np.random.choice(low_idx, int(len(low_idx) * 0.10), replace=False) # 10% of the far field
        
        final_idx = np.sort(np.concatenate([sel_high, sel_med, sel_low]))
        
        # 4. Create Subsampled Graph
        pos_full = data_full.pos.cpu().numpy()
        y_local_full = data_full.y_local.cpu().numpy()
        
        pos_fin = pos_full[final_idx]
        y_local_fin = y_local_full[final_idx]
        importance_fin = raw_importance[final_idx]
        
        nbrs_fin = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=1).fit(pos_fin)
        adj_fin = nbrs_fin.kneighbors_graph(mode='connectivity').tocoo()
        
        data_refined = Data(
            x=torch.tensor(pos_fin, dtype=torch.float),
            edge_index=torch.tensor(np.vstack((adj_fin.row, adj_fin.col)), dtype=torch.long),
            y_local=torch.tensor(y_local_fin, dtype=torch.float).view(-1, 1),
            y_global=data_full.y_global.cpu(),
            pos=torch.tensor(pos_fin, dtype=torch.float),
            importance=torch.tensor(importance_fin, dtype=torch.float).view(-1, 1),
            original_idx=torch.tensor(final_idx, dtype=torch.long) # CRITICAL FOR L'UPSAMPLING
        )
        
        torch.save(data_refined, os.path.join(out_dir, pt_file))
        
    # Copy split_info.json così the test set dataloader remains consistent
    import shutil
    shutil.copy(os.path.join(in_dir, "split_info.json"), os.path.join(out_dir, "split_info.json"))
    print("✅ Subsampled dataset generation completed in data/refined_graphs!")

if __name__ == "__main__":
    run_subsampling()

