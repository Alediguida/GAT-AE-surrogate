import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def load_vtk_mesh(vtk_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, vtk.vtkPointSet]:
    """
    Loads a 3D/2D surface mesh from a VTP or VTU file, computes point normals and cell-to-point areas,
    and extracts nodal coordinates, surface normals, point areas, and ground-truth pressure fields.

    Args:
        vtk_path (str): Path to the .vtp or .vtu XML file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, vtk.vtkPointSet]:
            - pos: Node Cartesian coordinates of shape [N, 3].
            - norm: Node surface unit normal vectors of shape [N, 3].
            - area: Nodal area weights of shape [N,].
            - p: Ground-truth surface pressure array of shape [N,].
            - mesh: Processed VTK object containing point data arrays.
    """
    if not os.path.exists(vtk_path):
        raise FileNotFoundError(f"VTK file not found at: {vtk_path}")

    if vtk_path.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()

    reader.SetFileName(vtk_path)
    reader.Update()
    mesh = reader.GetOutput()

    # Compute point normals if not already present
    if not mesh.GetPointData().GetNormals():
        if isinstance(mesh, vtk.vtkPolyData):
            norm_filter = vtk.vtkPolyDataNormals()
            norm_filter.SetInputData(mesh)
            norm_filter.ComputePointNormalsOn()
            norm_filter.SplittingOff()
            norm_filter.Update()
            mesh = norm_filter.GetOutput()
        else:
            # Cannot compute surface normals on a volume mesh directly.
            pass

    # Compute cell areas and interpolate to point data
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(mesh)
    cell_size.ComputeAreaOn()
    cell_size.Update()

    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(cell_size.GetOutput())
    c2p.PassCellDataOn()
    c2p.Update()
    final_mesh = c2p.GetOutput()

    pos = vtk_to_numpy(final_mesh.GetPoints().GetData())
    
    normals_array = final_mesh.GetPointData().GetNormals()
    if normals_array:
        norm = vtk_to_numpy(normals_array)
    else:
        norm = np.zeros_like(pos)

    # Extract point area
    if final_mesh.GetPointData().GetArray("Area"):
        area = vtk_to_numpy(final_mesh.GetPointData().GetArray("Area"))
    else:
        area = np.zeros(len(pos), dtype=np.float32)
    area = np.nan_to_num(area, nan=0.0)

    # Extract pressure field (pMean or p)
    if final_mesh.GetPointData().GetArray("pMean"):
        p = vtk_to_numpy(final_mesh.GetPointData().GetArray("pMean"))
    elif final_mesh.GetPointData().GetArray("p"):
        p = vtk_to_numpy(final_mesh.GetPointData().GetArray("p"))
    else:
        p = np.zeros(len(pos), dtype=np.float32)

    return pos, norm, area, p, final_mesh


def compute_vtk_gradient(
    polydata: vtk.vtkPolyData,
    scalar_array: np.ndarray,
    array_name: str = "ScoutPressure",
) -> np.ndarray:
    """
    Computes spatial gradient magnitude of a nodal scalar field on a VTK surface mesh.

    Args:
        polydata (vtk.vtkPolyData): VTK PolyData surface geometry.
        scalar_array (np.ndarray): 1D array of scalar values at each vertex.
        array_name (str): Temporary array identifier in VTK point data.

    Returns:
        np.ndarray: 1D array representing the L2 norm of spatial gradients at each vertex of shape [N,].
    """
    vtk_arr = numpy_to_vtk(scalar_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName(array_name)
    polydata.GetPointData().AddArray(vtk_arr)
    polydata.GetPointData().SetActiveScalars(array_name)

    grad_filter = vtk.vtkGradientFilter()
    grad_filter.SetInputData(polydata)
    grad_filter.SetInputScalars(0, array_name)
    grad_filter.SetResultArrayName("Gradient")
    grad_filter.Update()

    grads = vtk_to_numpy(grad_filter.GetOutput().GetPointData().GetArray("Gradient"))
    return np.linalg.norm(grads, axis=1)


def get_cd_from_csv(run_folder_path: str, run_id: int) -> float:
    """
    Parses aerodynamic drag coefficient (Cd) from run simulation CSV file.

    Args:
        run_folder_path (str): Folder containing the simulation results for this run.
        run_id (int): Run numerical identifier.

    Returns:
        float: Aerodynamic drag coefficient Cd.
    """
    csv_path = os.path.join(run_folder_path, f"Cd_{run_id}.csv")
    try:
        if not os.path.exists(csv_path):
            return 0.0
        df = pd.read_csv(csv_path, header=None)
        return float(df.iloc[0, 1])
    except Exception:
        return 0.0


def load_cd_map(root_dir: str) -> Dict[int, float]:
    """
    Scans simulation directory to map all available run IDs to their respective Cd values.

    Args:
        root_dir (str): Root dataset directory containing 'Risultati_Simulazioni'.

    Returns:
        Dict[int, float]: Mapping from run ID to drag coefficient Cd.
    """
    cd_map: Dict[int, float] = {}
    sim_dir = os.path.join(root_dir, "Risultati_Simulazioni")
    if not os.path.exists(sim_dir):
        return cd_map

    run_dirs = [d for d in os.listdir(sim_dir) if d.startswith("Run_")]
    for d in run_dirs:
        try:
            run_id = int(d.split("_")[1])
            folder_path = os.path.join(sim_dir, d)
            cd_val = get_cd_from_csv(folder_path, run_id)
            cd_map[run_id] = cd_val
        except Exception:
            continue
    return cd_map

