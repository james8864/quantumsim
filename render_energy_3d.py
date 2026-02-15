"""
3D Coulomb potential from nuclei positions.

Shared state file (nuclei_state.json) can be written by any front-end
(app.py sidebar **or** pubchem_chatbot.py).  Both ``save_nuclei`` and
``load_nuclei`` work with the same format so that potential_3d,
schrodinger_3d, and render_energy_3d always stay in sync.

Coulomb law: V_i(r) = -Z_i / |r - r_i| (atomic units).
"""
import json # file to store the nuclei positions
import os # operating system (file explorer)
import numpy as np 
from periodic_data import ELEMENTS_1_118

# Path to shared state file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUCLEI_STATE_PATH = os.path.join(_SCRIPT_DIR, "nuclei_state.json")

# Symbol → atomic number (Z) for elements 1-118  (single source of truth)
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(ELEMENTS_1_118)}

# Atomic number → symbol (reverse lookup)
Z_TO_SYMBOL = {z: s for s, z in SYMBOL_TO_Z.items()}

# Small cutoff to avoid 1/r singularity at nucleus
EPS = 1e-6


# ─────────────────────  Shared state I/O  ─────────────────────
def save_nuclei(nuclei, state_path=None):
    """
    Persist a list of nuclei to the shared JSON file.

    nuclei : [(symbol, (x, y, z)), ...]   — positions in Bohr
    state_path : optional override; default is NUCLEI_STATE_PATH.

    Called by app.py (_sync_nuclei_to_potential) and
    pubchem_chatbot.py (_on_molecule_received) so that every
    consumer of this file always sees the latest geometry.
    """
    path = state_path or NUCLEI_STATE_PATH
    data = [{"symbol": s, "x": x, "y": y, "z": z}
            for s, (x, y, z) in nuclei]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_nuclei(state_path=None):
    """
    Load nucleus positions from the shared JSON file.

    Returns: list of (symbol, (x, y, z)); empty list if file missing/invalid.
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


# Backward-compatible alias used by render_energy_3d.py
load_nuclei_from_app = load_nuclei

# associating atom to nuclear charge
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


def _potential_single_atom(Z_i, x_i, y_i, z_i):
    """
    Potential energy from one nucleus: V_i(x,y,z) = -Z_i / |r - r_i|.

    Returns a callable V_i(x, y, z) for that atom only.
    """
    def V_i(x, y, z):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        dx = x - x_i
        dy = y - y_i
        dz = z - z_i
        r = np.sqrt(dx * dx + dy * dy + dz * dz + EPS ** 2)
        return -Z_i / r
    return V_i


def potential_per_atom(nuclei):
    """
    One potential function per atom: V_i(x,y,z) = -Z_i / |r - r_i|.

    nuclei: list of (symbol, (x, y, z))
    Returns: list of callables [V_0, V_1, ...], one per nucleus.
    """
    Z, pos = _nuclei_to_charges_positions(nuclei)
    return [_potential_single_atom(Z[i], pos[i, 0], pos[i, 1], pos[i, 2])
            for i in range(len(Z))]


def potential_3d(nuclei):
    """
    Total 3D Coulomb potential: V(x,y,z) = sum_i V_i(x,y,z).

    For each atom i, V_i(x,y,z) = -Z_i / |r - r_i|.
    The total is the sum of per-atom contributions.

    nuclei: list of (symbol, (x, y, z))
    Returns: callable V(x, y, z) in atomic units.
    """
    per_atom = potential_per_atom(nuclei)
    if len(per_atom) == 0:
        def V_empty(x, y, z):
            return np.zeros_like(np.atleast_1d(x))
        return V_empty

    def V(x, y, z):
        out = np.zeros_like(np.atleast_1d(np.asarray(x, dtype=float)))
        for V_i in per_atom:
            out += V_i(x, y, z)
        return out

    return V


def potential_1d_cuts(nuclei, x0=None, y0=None, z0=None):
    """
    One energy function per dimension, each built by summing per-atom contributions.

    For each atom i: V_i(x,y,z) = -Z_i / |r - r_i|.
    For each dimension d, the cut is V_d(coord) = sum_i V_i evaluated along that line:

      V_x(x) = sum_i V_i(x,  y0, z0)   — x varies, y,z fixed at center
      V_y(y) = sum_i V_i(x0, y,  z0)   — y varies, x,z fixed
      V_z(z) = sum_i V_i(x0, y0, z )   — z varies, x,y fixed

    nuclei: list of (symbol, (x, y, z))
    x0, y0, z0: center for cuts. If None, use centroid of nuclei.

    Returns: (V_x, V_y, V_z)
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

    per_atom = potential_per_atom(nuclei)

    def V_x(x):
        """V_x(x) = sum_i V_i(x, y0, z0)."""
        out = np.zeros_like(np.atleast_1d(np.asarray(x, dtype=float)))
        for V_i in per_atom:
            out += V_i(np.atleast_1d(x), y0, z0)
        return out

    def V_y(y):
        """V_y(y) = sum_i V_i(x0, y, z0)."""
        out = np.zeros_like(np.atleast_1d(np.asarray(y, dtype=float)))
        for V_i in per_atom:
            out += V_i(x0, np.atleast_1d(y), z0)
        return out

    def V_z(z):
        """V_z(z) = sum_i V_i(x0, y0, z)."""
        out = np.zeros_like(np.atleast_1d(np.asarray(z, dtype=float)))
        for V_i in per_atom:
            out += V_i(x0, y0, np.atleast_1d(z))
        return out

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
