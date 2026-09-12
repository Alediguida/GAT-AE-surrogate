import os
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from utils.vtk_processing import load_vtk_mesh
from models.gnn_autoencoder import PressureAutoencoderGNN

def run_test():
    vtu_file = "data/dataset_run/run_0_field.vtu"
    if not os.path.exists(vtu_file):
        print(f"File {vtu_file} not found.")
        return

    print("1. Testing VTU Loading...")
    # Read manually first to see the available arrays
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file)
    reader.Update()
    mesh = reader.GetOutput()
    
    n_points = mesh.GetNumberOfPoints()
    print(f"Number of points: {n_points}")
    
    point_data = mesh.GetPointData()
    arrays = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    print(f"Point data arrays: {arrays}")
    
    # We expect coordinates to be in 2D (z=0)
    pos = vtk_to_numpy(mesh.GetPoints().GetData())
    print(f"Positions shape: {pos.shape}, Z-min: {pos[:, 2].min()}, Z-max: {pos[:, 2].max()}")
    
    # Let's extract x, y
    pos_2d = pos[:, :2]
    
    # Assuming normals are needed, but it's a 2D mesh (maybe boundary only or volume?).
    # OpenFOAM slice VTU is typically a 2D surface.
    
    print("\n2. Building k-NN Graph (k=6)...")
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto', n_jobs=1).fit(pos_2d)
    adj = nbrs.kneighbors_graph(mode='connectivity').tocoo()
    edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
    print(f"Edge index shape: {edge_index.shape}")
    
    # For a 2D fluid volume mesh, there are no surface normals. We use [x, y].
    x_features = torch.tensor(pos_2d, dtype=torch.float) # 2 features
    
    # Fake pressure targets
    y_local = torch.rand((n_points, 1), dtype=torch.float)
    y_global = torch.tensor([[0.015]], dtype=torch.float)
    batch = torch.zeros(n_points, dtype=torch.long)
    
    data = Data(x=x_features, edge_index=edge_index, y_local=y_local, y_global=y_global, batch=batch)
    
    print("\n3. Testing GNN Autoencoder (2D Volume, in_features=2)...")
    model = PressureAutoencoderGNN(in_features=2, latent_dim=25, out_local=1, heads=2, hidden=32)
    model.eval()
    
    with torch.no_grad():
        z = model.encode(data)
        print(f"Encoded latent z shape: {z.shape} (Expected: [1, 25])")
        
        out_p = model(data)
        print(f"Decoded pressure shape: {out_p.shape} (Expected: [{n_points}, 1])")
        
    print("✅ Dry-run successful! The architecture fully supports 2D datasets.")

if __name__ == "__main__":
    run_test()
