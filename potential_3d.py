"""
3D Coulomb potential from nuclei positions (synced with app.py via nuclei_state.json).
Coulomb law: electron–nucleus potential V_i(r) = -Z_i / |r - r_i| (atomic units).
Total potential V(x,y,z) = sum over all nuclei of V_i.
"""
import json
import os
import numpy as np
from periodic_data import ELEMENTS_1_82

# Path to shared state file written by app.py; read by potential and render
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUCLEI_STATE_PATH = os.path.join(_SCRIPT_DIR, "nuclei_state.json")

# Symbol -> atomic number (Z) for elements 1-82
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(ELEMENTS_1_82)}

# Small cutoff to avoid 1/r singularity at nucleus
EPS = 1e-6


def load_nuclei_from_app(state_path=None):
    """
    Load nucleus positions from the file written by app.py (Sync nuclei to potential).
    state_path: optional path to JSON file; default is NUCLEI_STATE_PATH.
    Returns: list of (symbol, (x, y, z)); empty list if file missing or invalid.
    """
    path = state_path or NUCLEI_STATE_PATH
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    out = []
    for item in data:
        s = item.get("symbol", "H")
        x = float(item.get("x", 0))
        y = float(item.get("y", 0))
        z = float(item.get("z", 0))
        out.append((s, (x, y, z)))
    return out


def _nuclei_to_charges_positions(nuclei):
    """
    nuclei: list of (symbol, (x, y, z))
    Returns: (Z_array shape (N,), positions shape (N, 3))
    """
    Z_list = []
    pos_list = []
    for symbol, (x, y, z) in nuclei:
        Z_list.append(SYMBOL_TO_Z.get(symbol, 1))
        pos_list.append([float(x), float(y), float(z)])
    return np.array(Z_list), np.array(pos_list)


def potential_3d(nuclei):
    """
    Build the 3D Coulomb potential V(x,y,z) from a list of nuclei.
    Coulomb law: V = sum_i ( -Z_i / |r - r_i| ) in atomic units.
    The charge magnitude Z_i is the atomic number (1 for H, 6 for C, etc.), so different
    atoms contribute according to their nuclear charge.

    nuclei: list of (symbol, (x, y, z))
    Returns: callable V(x, y, z) that returns the potential in atomic units.
             Uses small epsilon in denominator to avoid singularity at nuclei.
    """
    Z, pos = _nuclei_to_charges_positions(nuclei)
    if len(Z) == 0:
        def V_empty(x, y, z):
            return np.zeros_like(np.atleast_1d(x))
        return V_empty

    def V(x, y, z):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.zeros_like(x)
        for i in range(len(Z)):
            dx = x - pos[i, 0]
            dy = y - pos[i, 1]
            dz = z - pos[i, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz + EPS ** 2)
            out -= Z[i] / r
        return out

    return V


def potential_1d_cuts(nuclei, x0=None, y0=None, z0=None):
    """
    One energy function per dimension: 1D cuts through V(x,y,z) at a center.

    nuclei: list of (symbol, (x, y, z))
    x0, y0, z0: center for the cuts. If None, use the centroid of nuclei.

    Returns: (V_x, V_y, V_z) where
      V_x(x) = V(x, y0, z0)
      V_y(y) = V(x0, y, z0)
      V_z(z) = V(x0, y0, z)
    """
    Z, pos = _nuclei_to_charges_positions(nuclei)
    if pos.size == 0:
        cx = cy = cz = 0.0
    else:
        cx = float(np.mean(pos[:, 0]))
        cy = float(np.mean(pos[:, 1]))
        cz = float(np.mean(pos[:, 2]))
    x0 = cx if x0 is None else float(x0)
    y0 = cy if y0 is None else float(y0)
    z0 = cz if z0 is None else float(z0)

    V = potential_3d(nuclei)

    def V_x(x):
        return V(np.atleast_1d(x), y0, z0)

    def V_y(y):
        return V(x0, np.atleast_1d(y), z0)

    def V_z(z):
        return V(x0, y0, np.atleast_1d(z))

    return V_x, V_y, V_z


def potential_on_grid(nuclei, x_1d, y_1d, z_1d):
    """
    Evaluate V(x,y,z) on a 3D grid. Useful for visualization.

    nuclei: list of (symbol, (x, y, z))
    x_1d, y_1d, z_1d: 1D arrays defining the grid.

    Returns: (X, Y, Z, V) where X,Y,Z are 3D arrays from np.meshgrid and V is the potential.
    """
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
    V = potential_3d(nuclei)
    V_grid = V(X.ravel(), Y.ravel(), Z.ravel()).reshape(X.shape)
    return X, Y, Z, V_grid
