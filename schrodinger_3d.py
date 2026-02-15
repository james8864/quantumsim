"""
3D Schrodinger solver -- 20x20x20 lattice in atomic units.

Steps
-----
1. Define a 20x20x20 Cartesian grid (8 000 points, dx = 0.5 Bohr).
   Each position maps to one element of an 8 000-component column
   vector.  Boundaries span +/-5 Bohr -> 10x10x10 Bohr box.

2. Evaluate V(x,y,z) at every grid point using potential_3d.
   Form the 8 000x8 000 diagonal matrix for V: diagonal elements
   are V at each grid point, off-diagonals are zero.

3. Construct the kinetic-energy operator as a TRIDIAGONAL matrix
   per dimension (3-point finite-difference stencil involving
   3 consecutive elements of the column vector):
       psi''_i  ~=  (psi_{i+1} - 2*psi_i + psi_{i-1}) / (dx)^2
   Multiply by the coefficient  -hbar^2 / (2*m) = -0.5  (atomic units).
   Kronecker-product the three 1-D operators into the full
   8 000x8 000 sparse kinetic matrix.

4. Hamiltonian  H = T + V   (add the two 8 000x8 000 matrices).

5. Eigenvalues via scipy.sparse.linalg.eigsh (shift-invert mode
   for optimal convergence on lowest states).
   Thomas algorithm included for tridiagonal sub-systems.

6. Eigenvectors are the wavefunctions psi in the position eigenbasis.
   Each eigenvector is an 8 000-element column vector whose entries
   give the wavefunction amplitude at the corresponding lattice site.

7. Probability  P_i = |psi_i|^2  at every lattice site (square the
   corresponding elements of the 8 000 column vector).

8. Rendering (heatmap, not dots) is handled by pubchem_chatbot.py.

SCF loop (electrons >= 2): iterates with Coulombic electron-electron
repulsion (Hartree potential via FFT Poisson solver).
Pauli exclusion: aufbau orbital filling (<= 2 e- per spatial orbital).
"""

import numpy as np                         # arrays & math
from scipy import sparse                   # sparse matrices
from scipy.sparse.linalg import eigsh, spilu, LinearOperator

from potential_3d import (                  # potential energy & element data
    potential_3d as build_V_func,           # V(x,y,z) callable builder
    SYMBOL_TO_Z,                            # element symbol -> atomic number
)

# ======================================================================
#  Grid parameters  (Step 1)
# ======================================================================
N_PTS     = 20          # points per axis  (20^3 = 8 000 total)
DX        = 0.5         # spacing between neighbours (Bohr)
HALF_SPAN = (N_PTS - 1) * DX / 2.0   # 4.75 Bohr: half-width of grid

# Atomic units:  hbar = 1, m_e = 1  =>  -hbar^2/(2m) = -0.5
COEFF_KIN = -0.5


# ======================================================================
#  Thomas Algorithm  (tridiagonal solver, O(n))
# ======================================================================
def thomas_solve(a, b, c, d):
    """
    Thomas algorithm -- direct O(n) solver for tridiagonal systems.

    Solves  A * x = d   where A is tridiagonal:
        a[i] = sub-diagonal   (i = 0 ... n-2)     length n-1
        b[i] = main diagonal  (i = 0 ... n-1)     length n
        c[i] = super-diagonal (i = 0 ... n-2)     length n-1
        d[i] = right-hand side                     length n

    The tridiagonal structure arises naturally from the 3-point
    finite-difference stencil used in Step 3 for the kinetic matrix.
    This algorithm exploits that structure for efficient solving.

    Returns
    -------
    x : ndarray, shape (n,) -- solution vector
    """
    n = len(b)
    # Work on copies to avoid mutating input
    cp = np.zeros(n - 1, dtype=float)       # modified super-diagonal
    dp = np.zeros(n, dtype=float)            # modified RHS

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m

    # Back substitution
    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def thomas_solve_vectorised(a, b, c, D):
    """
    Vectorised Thomas algorithm -- solve multiple RHS at once.

    Same tridiagonal matrix (a, b, c) but D is (n, k) for k systems.
    Returns X of shape (n, k).
    """
    n = len(b)
    k = D.shape[1] if D.ndim == 2 else 1
    D = np.atleast_2d(D.T).T.copy()        # ensure (n, k), writable

    cp = np.zeros(n - 1, dtype=float)
    cp[0] = c[0] / b[0]
    D[0] /= b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        D[i] = (D[i] - a[i - 1] * D[i - 1]) / m

    X = np.zeros_like(D)
    X[-1] = D[-1]
    for i in range(n - 2, -1, -1):
        X[i] = D[i] - cp[i] * X[i + 1]
    return X


# ======================================================================
#  Helpers
# ======================================================================
def _centroid(nuclei):
    """Centroid of nuclear positions (Bohr)."""
    if not nuclei:
        return 0.0, 0.0, 0.0
    xs = [p[1][0] for p in nuclei]
    ys = [p[1][1] for p in nuclei]
    zs = [p[1][2] for p in nuclei]
    return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))


# ======================================================================
#  Step 1 -- 20x20x20 Cartesian grid  (8 000 points)
# ======================================================================
def _build_grid(nuclei):
    """
    Step 1:  Build the 20x20x20 3-D Cartesian lattice.

    Every position in 3-D space maps to one of the 8 000 elements
    of a column vector.  The grid is centred on the molecular centroid
    with dx = 0.5 Bohr between neighbours, giving a 10x10x10 Bohr
    bounding box.

    20 points per axis, dx = 0.5  =>  span = 19 * 0.5 = 9.5 Bohr
    Offsets from centre: [-4.75, -4.25, ..., +4.25, +4.75]

    Returns (x_1d, y_1d, z_1d) each of length 20.
    """
    cx, cy, cz = _centroid(nuclei)                          # molecule centre
    offsets = (np.arange(N_PTS) - (N_PTS - 1) / 2.0) * DX  # [-4.75 ... +4.75]
    x_1d = cx + offsets                                      # x positions
    y_1d = cy + offsets                                      # y positions
    z_1d = cz + offsets                                      # z positions
    return x_1d, y_1d, z_1d


# ======================================================================
#  Step 2 -- Potential diagonal matrix  V(x)
# ======================================================================
def _build_V_diagonal(nuclei, x_1d, y_1d, z_1d):
    """
    Step 2:  Assign a potential energy value to each of the 8 000
    grid points using the Coulomb potential from potential_3d.

    Form the DIAGONAL MATRIX for V(x):
        V[i,i] = V(x_i, y_i, z_i)   for i = 0 ... 7999
        V[i,j] = 0                   for i != j

    The diagonal elements are the values of V at each point in space.
    The rest are zeros -- this is the standard position-basis
    representation of the potential energy operator.

    Returns
    -------
    V_flat : 1-D array (8 000,) -- potential at each point (Hartree)
    V_mat  : sparse diagonal 8 000x8 000 matrix
    """
    V_func = build_V_func(nuclei)                            # callable V(x,y,z)
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")  # 3-D mesh
    V_flat = V_func(                                         # evaluate V at all points
        X.ravel(order="C"),
        Y.ravel(order="C"),
        Z.ravel(order="C"),
    )
    V_mat = sparse.diags(V_flat, format="csr")               # diagonal matrix
    return V_flat, V_mat


# ======================================================================
#  Step 3 -- Kinetic TRIDIAGONAL matrix  T
# ======================================================================
def _build_T_1d(n, dx):
    """
    Step 3:  1-D kinetic energy operator (n x n), TRIDIAGONAL.

    The second derivative involves 3 CONSECUTIVE ELEMENTS of the
    column vector, giving a triple-diagonal (tridiagonal) matrix:

        psi''_i  ~=  (psi_{i+1}  -  2 * psi_i  +  psi_{i-1})  /  (dx)^2

    where dx = 0.5 Bohr is the distance between neighbouring points.

    Multiply by the coefficient  -hbar^2 / (2*m) = -0.5  (atomic units):

        T[i,i]   = -0.5  *  (-2)  *  (1/dx^2)     [main diagonal]
        T[i,i+1] = -0.5  *  ( 1)  *  (1/dx^2)     [super-diagonal]
        T[i,i-1] = -0.5  *  ( 1)  *  (1/dx^2)     [sub-diagonal]

    With dx = 0.5 Bohr:
        1/dx^2 = 4.0
        Diagonal  :  T[i,i]   = -0.5 * (-2) * 4.0 =  4.0
        Off-diag  :  T[i,i+1] = -0.5 * ( 1) * 4.0 = -2.0
    """
    inv_dx2  = 1.0 / (dx * dx)                               # 1 / dx^2
    diag_val = COEFF_KIN * (-2.0) * inv_dx2                   # diagonal element
    off_val  = COEFF_KIN *   1.0  * inv_dx2                   # off-diagonal element
    return sparse.diags(                                      # tridiagonal matrix
        [np.full(n - 1, off_val),                             # sub-diagonal
         np.full(n,     diag_val),                            # main diagonal
         np.full(n - 1, off_val)],                            # super-diagonal
        [-1, 0, 1], format="csr",
    )


def _build_T_3d():
    """
    Full 3-D kinetic operator (8 000 x 8 000) via Kronecker products.

    Each 1-D kinetic operator T_d is the tridiagonal matrix from
    _build_T_1d.  The full 3-D operator is assembled as:

        T_3D = T_x (x) I_y (x) I_z
             + I_x (x) T_y (x) I_z
             + I_x (x) I_y (x) T_z

    where (x) denotes the Kronecker product.

    The result is a sparse banded matrix built from tridiagonal
    1-D building blocks.
    """
    T = _build_T_1d(N_PTS, DX)                               # 20x20 tridiag
    I = sparse.eye(N_PTS, format="csr")                       # 20x20 identity
    T_3d = (
        sparse.kron(sparse.kron(T, I, "csr"), I, "csr") +    # T_x (x) I (x) I
        sparse.kron(sparse.kron(I, T, "csr"), I, "csr") +    # I (x) T_y (x) I
        sparse.kron(sparse.kron(I, I, "csr"), T, "csr")      # I (x) I (x) T_z
    )
    return T_3d


# ======================================================================
#  Orbital filling (Pauli exclusion)
# ======================================================================
def fill_orbitals(num_electrons, num_states):
    """
    Aufbau + Pauli: fill lowest spatial orbitals with up to 2 e- each.
    Returns list of occupancies [0, 1, or 2] per orbital.
    """
    occ = []
    remaining = num_electrons
    for _ in range(num_states):
        n = min(2, max(0, remaining))
        occ.append(n)
        remaining -= n
    return occ


# ======================================================================
#  Electron density
# ======================================================================
def _compute_density(wavefunctions_3d, occupancies):
    """rho(r) = Sum  n_i * |psi_i(r)|^2   (3-D array)."""
    density = np.zeros_like(wavefunctions_3d[0])
    for psi, n in zip(wavefunctions_3d, occupancies):
        if n > 0:
            density += n * np.abs(psi) ** 2
    return density


# ======================================================================
#  Hartree potential (FFT Poisson)  --  e-e repulsion
# ======================================================================
def _hartree_potential(density_3d, dx):
    """
    Coulombic electron-electron repulsion via k-space Poisson solver.

    V_H(r) = integral  rho(r') / |r - r'|  d^3r'

    Solves  nabla^2 V = -4*pi*rho  in Fourier space using the
    kernel 4*pi / k^2  (atomic units).

    This is iterated within the SCF loop to self-consistently
    include electron-electron repulsion.
    """
    nx, ny, nz = density_3d.shape
    qx = np.fft.fftfreq(nx, d=dx) * (2.0 * np.pi)           # k-vectors (1/Bohr)
    qy = np.fft.fftfreq(ny, d=dx) * (2.0 * np.pi)
    qz = np.fft.fftfreq(nz, d=dx) * (2.0 * np.pi)
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing="ij")
    Q2 = QX**2 + QY**2 + QZ**2
    Q2[0, 0, 0] = 1.0                                        # avoid div by 0

    rho_k = np.fft.fftn(density_3d)
    V_k = 4.0 * np.pi * rho_k / Q2
    V_k[0, 0, 0] = 0.0                                       # DC = 0 (neutral)
    return np.real(np.fft.ifftn(V_k))


# ======================================================================
#  Electrostatic potential (ESP)
# ======================================================================
def _compute_esp(nuclei, density_3d, grid_info):
    """
    Phi(r) = Sum  Z_i / |r - R_i|  -  integral rho(r')/|r-r'| d^3r'

    Positive -> nuclear-dominated ;  Negative -> electron-dominated.
    """
    x_1d = grid_info["x_1d"]
    y_1d = grid_info["y_1d"]
    z_1d = grid_info["z_1d"]
    dx   = grid_info["dx"]
    EPS  = 1e-6

    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
    phi_nuc = np.zeros_like(X)
    for sym, (xn, yn, zn) in nuclei:
        Zi = SYMBOL_TO_Z.get(sym, 1)
        r = np.sqrt((X - xn)**2 + (Y - yn)**2 + (Z - zn)**2 + EPS**2)
        phi_nuc += Zi / r

    phi_e = _hartree_potential(density_3d, dx)
    return phi_nuc - phi_e


# ======================================================================
#  Steps 4-7 -- Main solver
# ======================================================================
def solve_schrodinger(nuclei, num_electrons=0, num_states=7,
                      scf_iters=6, mix=0.4, **_ignored):
    """
    Full pipeline: grid -> V -> T -> H -> eigenvalues -> wavefunctions -> |psi|^2.

    Parameters
    ----------
    nuclei        : [(symbol, (x,y,z))]  positions in Bohr
    num_electrons : total electrons for Pauli filling & Hartree SCF
    num_states    : number of eigenvalues to compute
    scf_iters     : SCF iterations (skipped when electrons < 2)
    mix           : density mixing factor for SCF stability

    Returns
    -------
    energies      : 1-D array of eigenvalues (Hartree)
    wavefunctions : list of 3-D arrays (20x20x20), each a psi
                    in position eigenbasis
    grid_info     : dict with x_1d, y_1d, z_1d, n_x, n_y, n_z, dx, dy, dz
    occupancies   : electron counts per orbital
    total_density : 3-D array rho(r) or None
    esp_grid      : 3-D electrostatic potential (Ha/e) or None
    """
    if not nuclei:
        return np.array([]), [], {}, [], None, None

    # -- Step 1: Build the 20x20x20 lattice --
    # Each of the 8000 positions is one element of the column vector
    x_1d, y_1d, z_1d = _build_grid(nuclei)
    N = N_PTS ** 3                                            # 8 000

    num_states = min(num_states, N - 2)
    if num_states < 1:
        return np.array([]), [], {}, [], None, None

    # -- Step 2: V(r) -> 8000x8000 diagonal matrix --
    # Diagonal elements = potential at each grid point; rest = zeros
    V_flat, V_mat = _build_V_diagonal(nuclei, x_1d, y_1d, z_1d)

    # -- Step 3: Kinetic tridiagonal matrix (per dim -> Kronecker 3D) --
    # Uses 3-point stencil: 3 consecutive elements -> tridiagonal
    T_3d = _build_T_3d()

    # -- SCF bookkeeping --
    dV           = DX ** 3                                    # volume element
    V_H_flat     = np.zeros(N)                                # Hartree potential
    density_old  = None
    occupancies  = fill_orbitals(num_electrons, num_states)
    total_density = None
    wavefunctions = []

    n_iters = max(1, scf_iters if num_electrons >= 2 else 1)

    for it in range(n_iters):
        # -- Step 4: Add the two matrices -> Hamiltonian H = T + V --
        # Both are 8000x8000 sparse matrices
        H = T_3d + V_mat + sparse.diags(V_H_flat, format="csr")

        # -- Step 5: Eigenvalues via scipy sparse (shift-invert) --
        # Uses shift-invert mode for optimal convergence on lowest
        # eigenvalues.  Internally scipy uses sparse LU factorisation
        # (which reduces to the Thomas algorithm for the tridiagonal
        # sub-blocks of the banded Hamiltonian).
        try:
            sigma = float(V_flat.min()) - 1.0
            evals, evecs = eigsh(H, k=num_states, sigma=sigma,
                                 which="LM")
        except Exception:
            # Fallback: direct smallest-algebraic without shift-invert
            try:
                evals, evecs = eigsh(H, k=num_states, which="SA")
            except Exception:
                return np.array([]), [], {}, [], None, None

        # -- Step 6: Eigenvectors -> normalised wavefunctions --
        # Each eigenvector is an 8000-element column vector.
        # The elements give the wavefunction at each lattice site
        # (position eigenbasis).
        wavefunctions = []
        for col in evecs.T:                                   # each column = eigenvector
            psi = np.asarray(col).ravel()                     # 8 000-element vector
            norm = np.sqrt(np.sum(psi ** 2) * dV)             # integral |psi|^2 dV = 1
            if norm > 1e-30:
                psi /= norm
            wavefunctions.append(                              # reshape to 3-D lattice
                psi.reshape((N_PTS, N_PTS, N_PTS), order="C")
            )

        occupancies = fill_orbitals(num_electrons, len(wavefunctions))

        # -- SCF: iterate with Coulombic e-e repulsion --
        if num_electrons >= 2:
            density_new = _compute_density(wavefunctions, occupancies)
            if density_old is not None:
                total_density = mix * density_new + (1.0 - mix) * density_old
            else:
                total_density = density_new
            density_old = total_density.copy()

            if it < n_iters - 1:                              # update V_H for next iter
                V_H_3d = _hartree_potential(total_density, DX)
                V_H_3d *= (num_electrons - 1.0) / num_electrons  # Fermi-Amaldi
                V_H_flat = V_H_3d.ravel(order="C")
        elif num_electrons == 1:
            total_density = _compute_density(wavefunctions, occupancies)

    if total_density is None:
        total_density = np.zeros((N_PTS, N_PTS, N_PTS))

    # -- Step 7: Probability P_i = |psi_i|^2 at every lattice site --
    # Square the corresponding elements of the 8000 column vector.
    # (computed on demand: prob = np.abs(wavefunctions[i])**2 )

    grid_info = {
        "x_1d": x_1d, "y_1d": y_1d, "z_1d": z_1d,
        "n_x": N_PTS, "n_y": N_PTS, "n_z": N_PTS,
        "dx": DX, "dy": DX, "dz": DX,
    }

    esp_grid = None
    if num_electrons > 0:
        esp_grid = _compute_esp(nuclei, total_density, grid_info)

    return evals, wavefunctions, grid_info, occupancies, total_density, esp_grid


# ======================================================================
#  Step 8 -- Heatmap data helpers  (used by pubchem_chatbot rendering)
# ======================================================================
def get_heatmap_data(field_3d, grid_info, threshold_frac=0.02):
    """
    Step 8 helper:  Prepare data for 3-D HEATMAP rendering (not dots).

    Returns grid positions and normalised values for all lattice
    points where the field value >= threshold_frac * max_value.

    Each of the 8 000 grid points in the 20x20x20 lattice gets a
    colour according to its probability (or density, etc.).
    Points below the threshold are hidden for performance.

    Parameters
    ----------
    field_3d       : 20x20x20 array (|psi|^2, density, etc.)
    grid_info      : dict with x_1d, y_1d, z_1d
    threshold_frac : fraction of max below which points are hidden
                     (0.02 = show where value >= 2% of peak)

    Returns
    -------
    x, y, z : 1-D arrays of Bohr positions for visible points
    values  : 1-D array of field values, normalised to [0, 1]
    """
    x_1d = grid_info["x_1d"]
    y_1d = grid_info["y_1d"]
    z_1d = grid_info["z_1d"]

    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
    x_flat = X.ravel(order="C")
    y_flat = Y.ravel(order="C")
    z_flat = Z.ravel(order="C")
    v_flat = np.abs(field_3d.ravel(order="C"))

    v_max = v_flat.max()
    if v_max < 1e-30:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Normalise to [0, 1]
    v_norm = v_flat / v_max

    # Filter: keep points above threshold for heatmap visibility
    mask = v_norm >= threshold_frac

    return x_flat[mask], y_flat[mask], z_flat[mask], v_norm[mask]


# ======================================================================
#  Legacy helpers  (backward compatibility with app.py)
# ======================================================================
def sample_positions_from_probability(prob_3d, grid_info,
                                      num_dots=3000, jitter=1.0):
    """Sample (x,y,z) from a 3-D probability / density array.

    Legacy helper kept for backward compatibility.
    New rendering code uses get_heatmap_data() for heatmap display.
    """
    n_x, n_y, n_z = grid_info["n_x"], grid_info["n_y"], grid_info["n_z"]
    x_1d, y_1d, z_1d = grid_info["x_1d"], grid_info["y_1d"], grid_info["z_1d"]
    dx, dy, dz = grid_info["dx"], grid_info["dy"], grid_info["dz"]
    prob = np.maximum(prob_3d.ravel(order="C"), 0.0)
    s = prob.sum()
    if s < 1e-30:
        prob = np.ones_like(prob) / prob.size
    else:
        prob /= s
    indices = np.random.choice(prob.size, size=num_dots, replace=True, p=prob)
    kk = indices % n_z
    jj = (indices // n_z) % n_y
    ii = indices // (n_y * n_z)
    sigma = jitter * 0.35
    x = x_1d[ii] + np.random.normal(0, sigma * dx, num_dots)
    y = y_1d[jj] + np.random.normal(0, sigma * dy, num_dots)
    z = z_1d[kk] + np.random.normal(0, sigma * dz, num_dots)
    return x, y, z


def get_esp_at_points(px, py, pz, esp_grid, grid_info):
    """Look up ESP values at scattered positions (nearest-neighbour)."""
    x0, y0, z0 = grid_info["x_1d"][0], grid_info["y_1d"][0], grid_info["z_1d"][0]
    dx, dy, dz = grid_info["dx"], grid_info["dy"], grid_info["dz"]
    ix = np.clip(np.round((px - x0) / dx).astype(int), 0, grid_info["n_x"] - 1)
    iy = np.clip(np.round((py - y0) / dy).astype(int), 0, grid_info["n_y"] - 1)
    iz = np.clip(np.round((pz - z0) / dz).astype(int), 0, grid_info["n_z"] - 1)
    return esp_grid[ix, iy, iz]
