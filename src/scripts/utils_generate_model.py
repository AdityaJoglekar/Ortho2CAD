import json
import os
import random
import subprocess
import sys

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree as KDTree
from trimesh.sample import sample_surface

def read_jsonl(file_path, *keys):
    """
    Reads a JSONL file and extracts specific keys from each dictionary.

    Args:
        file_path (str): Path to the JSONL file.
        *keys (str): One or more keys to extract from each JSON object.

    Returns:
        tuple: A tuple of lists, each corresponding to the extracted values for a given key.
    """
    results = {key: [] for key in keys}  # Create a dictionary to store lists for each key

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # Parse each line as a JSON object
            for key in keys:
                results[key].append(data.get(key, None))  # Append value or None if key is missing

    return tuple(results[key] for key in keys)  # Return lists as a tuple


def write_python_file(file_content, py_path):
    with open(py_path, "w", encoding="utf-8") as file:
        file.write(file_content)
    return

def run_python_script(py_path):
    try:
        result = subprocess.run(
            ["python", py_path],  # Use "python3" if needed
            capture_output=True,      # Capture stdout and stderr
            text=True,                # Decode output as text
            check=True                # Raise an exception if the script fails
        )
        return True
    except subprocess.CalledProcessError as e:
        if e.stdout:
            sys.stderr.write(e.stdout)
            if not e.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if e.stderr:
            sys.stderr.write(e.stderr)
            if not e.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.write(f"error: {e}\n")
        sys.stderr.flush()
        return False
    
# Writing ply file, from GenCAD/Ferdous's repo
def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
    return

# From DeepCAD
def convert_stl_to_point_cloud(stl_path, point_cloud_path, n_points, seed=42):
    np.random.seed(seed)
    out_mesh = trimesh.load(stl_path) # load the stl as a mesh
    out_pc, _ = sample_surface(out_mesh, n_points) # convert to a point cloud
    write_ply(out_pc, point_cloud_path)
    return out_pc