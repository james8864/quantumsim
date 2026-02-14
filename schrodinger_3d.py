"""
3D Schrödinger solver with self-consistent field (SCF).

Features
--------
- Electron-electron Coulomb repulsion  (Hartree potential via FFT)
- Pauli exclusion principle  (aufbau orbital filling, 2 e⁻ per spatial orbital)
- Combined electron density  (sum over all occupied orbitals)
- Electrostatic potential map  (charge polarity: nuclear vs electron)

Pipeline
--------
1. Positions in **Bohr** → convert to **SI** (m, J) for computation.
2. Build 1-D kinetic operators per dimension; combine via Kronecker products.
3. Nuclear Coulomb potential  V_nuc = −Σ Z_i k_e e² / |r − R_i|.
4. SCF loop (when num_electrons ≥ 2):
       a.  H = T + V_nuc + V_H
       b.  eigsh → 7 lowest eigenstates
       c.  Fill orbitals (aufbau / Pauli) with N electrons
       d.  ρ(r) = Σ n_i |ψ_i|²   (electron density)
       e.  V_H via FFT Poisson solver  (electron-electron repulsion)
       f.  Fermi-Amaldi self-interaction correction
       g.  Density mixing for stability
5. Convert eigenvalues  J → **Hartree**;  grid → **Bohr**.
6. Electrostatic potential  Φ = Σ Z/r − ∫ ρ/|r−r′| d³r′  (atomic units).

Grid: 16³ = 4 096 position eigenstates (≈4× of 1 000).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from potential_3d import SYMBOL_TO_Z

# ═══════════════════════  SI constants  ═══════════════════════
A0   = 5.29177210903e-11       # Bohr radius  (m)
HA   = 4.3597447222071e-18     # Hartree      (J)
M_E  = 9.1093837015e-31        # Electron mass (kg)
HBAR = 1.054571817e-34         # ℏ  (J·s)
K_E  = 8.9875517873681764e9    # 1/(4πε₀)     (N·m²/C²)
Q_E  = 1.602176634e-19         # e             (C)
EPS_R = 1e-12                  # singularity guard  (Bohr)


# ═══════════════════════  Unit helpers  ═══════════════════════
def _bohr_to_m(r):
    return r * A0

def _j_to_ha(E):
    return E / HA


# ═══════════════════════  Bounding box  ═══════════════════════
def _bounding_box_bohr(nuclei):
    """
    Adaptive box (Bohr).

    With SCF screening, outer orbitals extend much further than bare-Z predicts.
    A fixed 8-Bohr pad around the nuclei captures screened orbitals comfortably
    while keeping the 16-point spacing at ≈ 1 Bohr.
    """
    if not nuclei:
        return -8, 8, -8, 8, -8, 8
    xs = [p[1][0] for p in nuclei]
    ys = [p[1][1] for p in nuclei]
    zs = [p[1][2] for p in nuclei]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 0.0)

    pad = 8.0
    half = max(span / 2.0 + pad, pad)
    return cx - half, cx + half, cy - half, cy + half, cz - half, cz + half


# ═══════════════════  1-D kinetic operator  ═══════════════════
def _build_1d_kinetic_si(n, dx_m):
    """T = −ℏ²/(2m) d²/dx²  (3-point stencil, CSR, Joules)."""
    c = HBAR ** 2 / (2.0 * M_E * dx_m ** 2)
    return sparse.diags(
        [np.full(n - 1, -c), np.full(n, 2.0 * c), np.full(n - 1, -c)],
        [-1, 0, 1], format="csr",
    )


# ═══════════════════  Nuclear Coulomb  ═══════════════════════
def _nuclear_potential_si(nuclei, x_m, y_m, z_m):
    """V_nuc(r) = −Σ Z_i k_e e² / |r − R_i|  (Joules, flat C-order)."""
    X, Y, Z = np.meshgrid(x_m, y_m, z_m, indexing="ij")
    V = np.zeros_like(X)
    for sym, (xb, yb, zb) in nuclei:
        Zi = SYMBOL_TO_Z.get(sym, 1)
        r = np.sqrt(
            (X - _bohr_to_m(xb)) ** 2 +
            (Y - _bohr_to_m(yb)) ** 2 +
            (Z - _bohr_to_m(zb)) ** 2 +
            (EPS_R * A0) ** 2
        )
        V -= Zi * K_E * Q_E ** 2 / r
    return V.ravel(order="C")


# ═══════════════════  Hartree potential (FFT)  ════════════════
def _coulomb_convolution_fft(density_3d, dx, dy, dz, prefactor=1.0):
    """
    V(r) = prefactor × ∫ ρ(r′)/|r−r′| d³r′   via k-space Coulomb kernel.

    Solves the Poisson equation  ∇²V = −4π·prefactor·ρ  in Fourier space
    using the analytic kernel  4π/k².  This is inherently symmetric for
    any grid parity (even or odd) and avoids spatial-kernel centering
    artefacts that break molecular symmetry on even grids.
    """
    n_x, n_y, n_z = density_3d.shape

    # k-space wave-vectors (rad / length-unit)
    qx = np.fft.fftfreq(n_x, d=dx) * (2.0 * np.pi)
    qy = np.fft.fftfreq(n_y, d=dy) * (2.0 * np.pi)
    qz = np.fft.fftfreq(n_z, d=dz) * (2.0 * np.pi)
    QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing="ij")
    Q2 = QX ** 2 + QY ** 2 + QZ ** 2
    Q2[0, 0, 0] = 1.0                    # avoid division by zero

    # Poisson solver:  V(k) = 4π · prefactor · ρ(k) / k²
    rho_k = np.fft.fftn(density_3d)
    V_k = 4.0 * np.pi * prefactor * rho_k / Q2
    V_k[0, 0, 0] = 0.0                   # charge-neutrality (zero mean)

    return np.real(np.fft.ifftn(V_k))


# ═══════════════════  Grid-symmetry enforcement  ═════════════
def _detect_grid_symmetries(V_3d, rtol=1e-10):
    """
    Detect which of the 48 Oh operations leave *V_3d* invariant
    on a cubic grid (n_x = n_y = n_z, same range on all axes).

    Each operation is ``(perm, signs)`` where *perm* is a 3-axis
    permutation tuple and *signs* is a boolean triple indicating
    which axes are reversed.

    Returns at least the identity operation.
    """
    n_x, n_y, n_z = V_3d.shape
    # Only test permutations when all dimensions are equal
    if not (n_x == n_y == n_z):
        return [((0, 1, 2), (False, False, False))]

    scale = max(abs(V_3d.max()), abs(V_3d.min()), 1e-30)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
             (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    ops = []
    for perm in perms:
        for s0 in (False, True):
            for s1 in (False, True):
                for s2 in (False, True):
                    a = np.transpose(V_3d, perm)
                    if s0:
                        a = a[::-1, :, :]
                    if s1:
                        a = a[:, ::-1, :]
                    if s2:
                        a = a[:, :, ::-1]
                    if np.allclose(a, V_3d, rtol=rtol, atol=rtol * scale):
                        ops.append((perm, (s0, s1, s2)))
    return ops if ops else [((0, 1, 2), (False, False, False))]


def _symmetrize_3d(arr, sym_ops):
    """Average a 3-D array over the detected symmetry operations."""
    if len(sym_ops) <= 1:
        return arr
    result = np.zeros_like(arr)
    for perm, (s0, s1, s2) in sym_ops:
        a = np.transpose(arr, perm)
        if s0:
            a = a[::-1, :, :]
        if s1:
            a = a[:, ::-1, :]
        if s2:
            a = a[:, :, ::-1]
        result += a
    result /= len(sym_ops)
    return result


# ═══════════════════  Orbital filling (Pauli)  ═══════════════
def fill_orbitals(num_electrons, num_states):
    """
    Aufbau + Pauli exclusion: fill lowest-energy spatial orbitals
    with up to 2 electrons each (spin ↑↓).

    Returns a list of occupancy counts  [0, 1, or 2]  per orbital.
    """
    occ = []
    remaining = num_electrons
    for _ in range(num_states):
        n = min(2, max(0, remaining))
        occ.append(n)
        remaining -= n
    return occ


# ═══════════════════  Electron density  ═══════════════════════
def _compute_density(wavefunctions, occupancies):
    """ρ(r) = Σ n_i |ψ_i(r)|²   (same units as |ψ|²)."""
    density = np.zeros_like(wavefunctions[0])
    for psi, n in zip(wavefunctions, occupancies):
        if n > 0:
            density += n * np.abs(psi) ** 2
    return density


# ═══════════════════  Electrostatic potential  ════════════════
def _compute_esp(nuclei, density_si_3d, grid_info):
    """
    Electrostatic potential Φ(r) on the grid  (atomic units, Ha/e).

        Φ = Σ Z_i / |r − R_i|  −  ∫ ρ(r′) / |r − r′| d³r′

    Positive → nuclear-dominated (electrophilic).
    Negative → electron-dominated (nucleophilic).
    """
    x_1d = grid_info["x_1d"]
    y_1d = grid_info["y_1d"]
    z_1d = grid_info["z_1d"]
    dx, dy, dz = grid_info["dx"], grid_info["dy"], grid_info["dz"]

    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    # Nuclear contribution (positive, atomic units)
    phi_nuc = np.zeros_like(X)
    for sym, (xn, yn, zn) in nuclei:
        Zi = SYMBOL_TO_Z.get(sym, 1)
        r = np.sqrt((X - xn) ** 2 + (Y - yn) ** 2 + (Z - zn) ** 2 + EPS_R ** 2)
        phi_nuc += Zi / r

    # Electron contribution via FFT convolution (atomic units)
    # SI density → AU:  ρ_au = ρ_SI × a₀³
    rho_au = density_si_3d * A0 ** 3
    phi_e = _coulomb_convolution_fft(rho_au, dx, dy, dz, prefactor=1.0)

    return phi_nuc - phi_e


# ═══════════════════════  Main solver  ════════════════════════
def solve_schrodinger(nuclei, num_electrons=0, n_grid=16, num_states=7,
                      scf_iters=6, mix=0.4):
    """
    Self-consistent field (SCF) Schrödinger solver.

    Parameters
    ----------
    nuclei        : [(symbol, (x,y,z))]  positions in Bohr
    num_electrons : total electrons (for Pauli filling & Hartree)
    n_grid        : grid per dimension  (16 → 4 096 total)
    num_states    : eigenvalues to compute
    scf_iters     : SCF iterations (skipped when num_electrons < 2)
    mix           : density mixing  (0.3–0.5 typical)

    Returns
    -------
    energies_ha     : 1-D array, eigenvalues in Hartree
    wavefunctions   : list of 3-D arrays  (SI normalisation)
    grid_info       : dict with Bohr grids / spacings
    occupancies     : list of electron counts per orbital
    total_density   : 3-D array, electron density (SI m⁻³) or None
    esp_grid        : 3-D array, electrostatic potential (Ha/e) or None
    """
    if not nuclei:
        return np.array([]), [], {}, [], None, None

    # ── bounding box (Bohr) ──
    x_min, x_max, y_min, y_max, z_min, z_max = _bounding_box_bohr(nuclei)
    n_x = n_y = n_z = n_grid

    # ── SI grids ──
    x_m = np.linspace(_bohr_to_m(x_min), _bohr_to_m(x_max), n_x)
    y_m = np.linspace(_bohr_to_m(y_min), _bohr_to_m(y_max), n_y)
    z_m = np.linspace(_bohr_to_m(z_min), _bohr_to_m(z_max), n_z)
    dx_m = float(x_m[1] - x_m[0]) if n_x > 1 else A0
    dy_m = float(y_m[1] - y_m[0]) if n_y > 1 else A0
    dz_m = float(z_m[1] - z_m[0]) if n_z > 1 else A0
    dV_m = dx_m * dy_m * dz_m

    # ── 3-D kinetic (Kronecker products) ──
    T_x = _build_1d_kinetic_si(n_x, dx_m)
    T_y = _build_1d_kinetic_si(n_y, dy_m)
    T_z = _build_1d_kinetic_si(n_z, dz_m)
    Ix = sparse.eye(n_x, format="csr")
    Iy = sparse.eye(n_y, format="csr")
    Iz = sparse.eye(n_z, format="csr")

    T_3d = (
        sparse.kron(sparse.kron(T_x, Iy, "csr"), Iz, "csr") +
        sparse.kron(sparse.kron(Ix, T_y, "csr"), Iz, "csr") +
        sparse.kron(sparse.kron(Ix, Iy, "csr"), T_z, "csr")
    )

    # ── nuclear potential (fixed across SCF) ──
    V_nuc = _nuclear_potential_si(nuclei, x_m, y_m, z_m)

    # ── detect symmetry operations of the nuclear potential ──
    V_nuc_3d = V_nuc.reshape((n_x, n_y, n_z), order="C")
    sym_ops = _detect_grid_symmetries(V_nuc_3d)

    N = n_x * n_y * n_z
    num_states = min(num_states, N - 2)
    if num_states < 1:
        return np.array([]), [], {}, [], None, None

    # ── SCF loop ──
    V_H_flat = np.zeros(N)
    density_old = None
    wavefunctions = []
    occupancies = fill_orbitals(num_electrons, num_states)
    total_density = None

    # Only iterate if there are ≥ 2 electrons (self-repulsion is unphysical)
    n_iters = max(1, scf_iters if num_electrons >= 2 else 1)

    for it in range(n_iters):
        # ── H = T + V_nuc + V_H ──
        H = T_3d + sparse.diags(V_nuc + V_H_flat, format="csr")

        try:
            evals_j, evecs = eigsh(H, k=num_states, which="SA")
        except Exception:
            return np.array([]), [], {}, [], None, None

        # ── normalise wavefunctions ──
        wavefunctions = []
        for col in evecs.T:
            psi = np.asarray(col).ravel()
            norm = np.sqrt(np.sum(psi ** 2) * dV_m)
            if norm > 1e-30:
                psi /= norm
            wavefunctions.append(psi.reshape((n_x, n_y, n_z), order="C"))

        # ── fill orbitals ──
        occupancies = fill_orbitals(num_electrons, len(wavefunctions))

        # ── density + Hartree ──
        if num_electrons >= 2:
            density_new = _compute_density(wavefunctions, occupancies)
            # density mixing
            if density_old is not None:
                total_density = mix * density_new + (1.0 - mix) * density_old
            else:
                total_density = density_new
            # enforce molecular symmetry on the density
            if len(sym_ops) > 1:
                total_density = _symmetrize_3d(total_density, sym_ops)
            density_old = total_density.copy()

            # update Hartree for next iteration (skip on last)
            if it < n_iters - 1:
                V_H_3d = _coulomb_convolution_fft(
                    total_density, dx_m, dy_m, dz_m,
                    prefactor=K_E * Q_E ** 2,
                )
                # Fermi-Amaldi self-interaction correction
                V_H_3d *= (num_electrons - 1.0) / num_electrons
                V_H_flat = V_H_3d.ravel(order="C")
        elif num_electrons == 1:
            total_density = _compute_density(wavefunctions, occupancies)
            if len(sym_ops) > 1:
                total_density = _symmetrize_3d(total_density, sym_ops)

    if total_density is None:
        total_density = np.zeros((n_x, n_y, n_z))

    # ── energies → Hartree ──
    energies_ha = _j_to_ha(evals_j)

    # ── output grid (Bohr) ──
    x_1d = np.linspace(x_min, x_max, n_x)
    y_1d = np.linspace(y_min, y_max, n_y)
    z_1d = np.linspace(z_min, z_max, n_z)

    grid_info = {
        "x_1d": x_1d, "y_1d": y_1d, "z_1d": z_1d,
        "n_x": n_x, "n_y": n_y, "n_z": n_z,
        "dx": float(x_1d[1] - x_1d[0]) if n_x > 1 else 1.0,
        "dy": float(y_1d[1] - y_1d[0]) if n_y > 1 else 1.0,
        "dz": float(z_1d[1] - z_1d[0]) if n_z > 1 else 1.0,
    }

    # ── electrostatic potential (atomic units) ──
    esp_grid = None
    if num_electrons > 0:
        esp_grid = _compute_esp(nuclei, total_density, grid_info)

    return energies_ha, wavefunctions, grid_info, occupancies, total_density, esp_grid


# ═══════════════════  Probability sampling  ═══════════════════
def sample_positions_from_probability(prob_3d, grid_info,
                                      num_dots=3000, jitter=1.0):
    """
    Sample (x,y,z) from a 3-D probability / density array.

    Works for single-orbital |ψ|², combined electron density, or any
    non-negative 3-D field.  Gaussian jitter smooths the discrete grid.
    """
    n_x = grid_info["n_x"]
    n_y = grid_info["n_y"]
    n_z = grid_info["n_z"]
    x_1d = grid_info["x_1d"]
    y_1d = grid_info["y_1d"]
    z_1d = grid_info["z_1d"]
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
    """Look up electrostatic-potential values at scattered dot positions."""
    x0, y0, z0 = grid_info["x_1d"][0], grid_info["y_1d"][0], grid_info["z_1d"][0]
    dx, dy, dz = grid_info["dx"], grid_info["dy"], grid_info["dz"]
    ix = np.clip(np.round((px - x0) / dx).astype(int), 0, grid_info["n_x"] - 1)
    iy = np.clip(np.round((py - y0) / dy).astype(int), 0, grid_info["n_y"] - 1)
    iz = np.clip(np.round((pz - z0) / dz).astype(int), 0, grid_info["n_z"] - 1)
    return esp_grid[ix, iy, iz]
