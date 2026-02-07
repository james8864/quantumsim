"""
Solve the 3D Schrödinger equation H ψ = E ψ on a grid.
H = -½∇² + V(r) in atomic units; V from potential_3d (Coulomb potential with nuclear charge Z).
Returns eigenvalues (energy levels) and eigenstates (wavefunctions); probability density |ψ|².
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from potential_3d import potential_on_grid


def _bounding_box(nuclei, padding=3.0):
    """Return (x_min, x_max, y_min, y_max, z_min, z_max) with padding."""
    if not nuclei:
        return -4, 4, -4, 4, -4, 4
    xs = [n[1][0] for n in nuclei]
    ys = [n[1][1] for n in nuclei]
    zs = [n[1][2] for n in nuclei]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    half = max(
        max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 2.0
    ) / 2 + padding
    return (
        cx - half, cx + half,
        cy - half, cy + half,
        cz - half, cz + half,
    )


def _index_to_linear(i, j, k, n_x, n_y, n_z):
    return i + n_x * j + n_x * n_y * k


def _build_hamiltonian(nuclei, x_1d, y_1d, z_1d):
    """
    Build sparse Hamiltonian H = -½∇² + V on the 3D grid.
    Dirichlet BC (ψ=0 at boundary). Returns scipy sparse matrix (csr).
    """
    n_x, n_y, n_z = len(x_1d), len(y_1d), len(z_1d)
    dx = float(x_1d[1] - x_1d[0]) if n_x > 1 else 1.0
    dy = float(y_1d[1] - y_1d[0]) if n_y > 1 else 1.0
    dz = float(z_1d[1] - z_1d[0]) if n_z > 1 else 1.0

    N = n_x * n_y * n_z
    X, Y, Z, V_grid = potential_on_grid(nuclei, x_1d, y_1d, z_1d)
    V_flat = V_grid.ravel()

    # Kinetic -½∇²: 7-point stencil. Diagonal 1/dx²+1/dy²+1/dz², off-diag -1/(2*dx²) etc.
    diag = np.zeros(N)
    rows, cols, data = [], [], []

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                idx = _index_to_linear(i, j, k, n_x, n_y, n_z)
                diag[idx] = 1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)
                # x neighbors
                if i + 1 < n_x:
                    jdx = _index_to_linear(i + 1, j, k, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dx * dx))
                if i - 1 >= 0:
                    jdx = _index_to_linear(i - 1, j, k, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dx * dx))
                # y
                if j + 1 < n_y:
                    jdx = _index_to_linear(i, j + 1, k, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dy * dy))
                if j - 1 >= 0:
                    jdx = _index_to_linear(i, j - 1, k, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dy * dy))
                # z
                if k + 1 < n_z:
                    jdx = _index_to_linear(i, j, k + 1, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dz * dz))
                if k - 1 >= 0:
                    jdx = _index_to_linear(i, j, k - 1, n_x, n_y, n_z)
                    rows.append(idx); cols.append(jdx); data.append(-0.5 / (dz * dz))

    T = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    H = T + sparse.diags(diag + V_flat, format="csr")
    return H, (x_1d, y_1d, z_1d), (n_x, n_y, n_z)


def solve_schrodinger(nuclei, n_grid=18, num_states=8):
    """
    Solve H ψ = E ψ for the lowest num_states eigenvalues and eigenvectors.
    nuclei: list of (symbol, (x, y, z))
    n_grid: number of grid points per axis (total points = n_grid^3)
    num_states: number of eigenstates to compute
    Returns: (energies, wavefunctions, grid_info)
      energies: 1D array of length num_states (hartree)
      wavefunctions: list of 3D arrays, shape (n_x, n_y, n_z), each normalized so ∫|ψ|² dV = 1
      grid_info: dict with 'x_1d', 'y_1d', 'z_1d', 'n_x', 'n_y', 'n_z', 'dx', 'dy', 'dz'
    """
    if not nuclei:
        return np.array([]), [], {}

    x_min, x_max, y_min, y_max, z_min, z_max = _bounding_box(nuclei)
    x_1d = np.linspace(x_min, x_max, n_grid)
    y_1d = np.linspace(y_min, y_max, n_grid)
    z_1d = np.linspace(z_min, z_max, n_grid)

    H, (x_1d, y_1d, z_1d), (n_x, n_y, n_z) = _build_hamiltonian(nuclei, x_1d, y_1d, z_1d)
    N = n_x * n_y * n_z
    num_states = min(num_states, N - 1)
    if num_states < 1:
        return np.array([]), [], {}

    dx = float(x_1d[1] - x_1d[0]) if n_x > 1 else 1.0
    dy = float(y_1d[1] - y_1d[0]) if n_y > 1 else 1.0
    dz = float(z_1d[1] - z_1d[0]) if n_z > 1 else 1.0
    dV = dx * dy * dz

    try:
        eigenvalues, eigenvectors = eigsh(H, k=num_states, which="SA")
    except Exception:
        return np.array([]), [], {}

    wavefunctions = []
    for ev in eigenvectors.T:
        psi_flat = np.asarray(ev).ravel()
        norm = np.sqrt(np.sum(psi_flat ** 2) * dV)
        if norm > 1e-20:
            psi_flat = psi_flat / norm
        psi_3d = psi_flat.reshape((n_x, n_y, n_z), order="C")
        wavefunctions.append(psi_3d)

    grid_info = {
        "x_1d": x_1d, "y_1d": y_1d, "z_1d": z_1d,
        "n_x": n_x, "n_y": n_y, "n_z": n_z,
        "dx": dx, "dy": dy, "dz": dz, "dV": dV,
    }
    return eigenvalues, wavefunctions, grid_info


def sample_positions_from_probability(psi_3d, grid_info, num_dots=2000, jitter=0.0):
    """
    Sample (x,y,z) positions from the probability density |ψ|².
    More samples in regions where |ψ|² is larger (more dots = higher probability).
    Returns: (x, y, z) arrays of length num_dots.
    """
    n_x, n_y, n_z = grid_info["n_x"], grid_info["n_y"], grid_info["n_z"]
    x_1d, y_1d, z_1d = grid_info["x_1d"], grid_info["y_1d"], grid_info["z_1d"]
    dx, dy, dz = grid_info["dx"], grid_info["dy"], grid_info["dz"]

    prob = (np.abs(psi_3d) ** 2).ravel()
    prob = np.maximum(prob, 0)
    s = prob.sum()
    if s < 1e-30:
        prob = np.ones_like(prob) / prob.size
    else:
        prob = prob / s

    indices = np.random.choice(prob.size, size=num_dots, replace=True, p=prob)
    ii = indices % n_x
    jj = (indices // n_x) % n_y
    kk = indices // (n_x * n_y)

    x = x_1d[ii] + (np.random.rand(num_dots) - 0.5) * jitter * dx
    y = y_1d[jj] + (np.random.rand(num_dots) - 0.5) * jitter * dy
    z = z_1d[kk] + (np.random.rand(num_dots) - 0.5) * jitter * dz
    return x, y, z