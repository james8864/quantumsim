"""
Schrödinger solver — 50x50x50 grid, potential + kinetic, eigenvalues, electron density.

Can be run standalone (reads nuclei from nuclei_state.json) or imported
and called with nuclei from PubChem: compute_schrodinger_cloud(nuclei)
+ open_schrodinger_cloud_window(data, parent, mol_name).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from potential_3d import load_nuclei, potential_3d

# ─────────────────────  Default parameters  ─────────────────────
DEFAULT_N = 50
DEFAULT_L = 10.0
NUM_EIGENVALUES = 10
BG = "#1e1e2e"
BG_PLOT = "#181825"
FG = "#cdd6f4"


def compute_schrodinger_cloud(nuclei, n=None, l_span=None, num_states=None):
    """
    Run full Schrödinger pipeline: grid → V → T → H → eigsh → psi → probabilities.

    nuclei : [(symbol, (x,y,z)), ...]  in Bohr
    n      : grid points per axis (default 50)
    l_span : total span per axis in Bohr (default 10)
    num_states : number of eigenvalues (default 10)

    Returns dict with: x_flat, y_flat, z_flat, nuclei, V_vec, prob_list (list of
    |psi_i|^2), prob_total, eigenvalues, n, n_total
    """
    if not nuclei:
        return None
    n = n or DEFAULT_N
    l_span = l_span or DEFAULT_L
    num_states = num_states or NUM_EIGENVALUES

    n_total = n ** 3
    n2 = n * n

    cx = np.mean([p[1][0] for p in nuclei])
    cy = np.mean([p[1][1] for p in nuclei])
    cz = np.mean([p[1][2] for p in nuclei])

    x_1d = np.linspace(cx - l_span / 2, cx + l_span / 2, n)
    y_1d = np.linspace(cy - l_span / 2, cy + l_span / 2, n)
    z_1d = np.linspace(cz - l_span / 2, cz + l_span / 2, n)
    dx = x_1d[1] - x_1d[0]
    dx2 = dx ** 2

    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")
    x_flat = X.ravel(order="C")
    y_flat = Y.ravel(order="C")
    z_flat = Z.ravel(order="C")

    V_func = potential_3d(nuclei)
    V_vec = V_func(x_flat, y_flat, z_flat)
    V_mat = sparse.diags(V_vec, 0, shape=(n_total, n_total), format="csr")

    coeff = -0.5
    diag_val = coeff * (-2.0) / dx2
    off_val = coeff * (1.0) / dx2
    main_diag = np.full(n_total, 3.0 * diag_val)

    off_z = np.full(n_total - 1, off_val)
    for k in range(1, n_total):
        if k % n == 0:
            off_z[k - 1] = 0.0

    off_y = np.full(n_total - n, off_val)
    for k in range(n, n_total):
        if k % n2 < n:
            off_y[k - n] = 0.0

    off_x = np.full(n_total - n2, off_val)
    T_mat = sparse.diags(
        [off_x, off_y, off_z, main_diag, off_z, off_y, off_x],
        [-n2, -n, -1, 0, 1, n, n2],
        shape=(n_total, n_total),
        format="csr",
    )
    H_mat = T_mat + V_mat

    num_states = min(num_states, n_total - 2)
    if num_states < 1:
        return None
    eigenvalues, eigenvectors = eigsh(H_mat, k=num_states, which="SA")

    psi = [eigenvectors[:, i] for i in range(num_states)]
    prob_list = [np.abs(psi[i]) ** 2 for i in range(num_states)]
    prob_total = np.sum(prob_list, axis=0)

    return {
        "x_flat": x_flat,
        "y_flat": y_flat,
        "z_flat": z_flat,
        "nuclei": nuclei,
        "V_vec": V_vec,
        "prob_list": prob_list,
        "prob_total": prob_total,
        "eigenvalues": eigenvalues,
        "n": n,
        "n_total": n_total,
        "x_1d": x_1d,
        "y_1d": y_1d,
        "z_1d": z_1d,
    }


def open_schrodinger_cloud_window(data, parent=None, mol_name=""):
    """
    Open a Toplevel (or Tk) window with heatmap buttons: V(r), single |psi_i|^2,
    combined density.

    data : result from compute_schrodinger_cloud()
    parent : optional Tk window (creates Toplevel child)
    mol_name : for window title
    """
    x_flat = data["x_flat"]
    y_flat = data["y_flat"]
    z_flat = data["z_flat"]
    nuclei = data["nuclei"]
    V_vec = data["V_vec"]
    prob_list = data["prob_list"]
    prob_total = data["prob_total"]
    eigenvalues = data["eigenvalues"]
    n = data["n"]
    n_total = data["n_total"]
    num_states = len(prob_list)

    def _plot_heatmap(parent_frame, values, title, cbar_label, cmap="inferno"):
        flat = np.asarray(values)
        v_lo = float(np.percentile(flat, 1))
        v_hi = float(np.percentile(flat, 99))
        if v_lo >= v_hi:
            v_lo, v_hi = float(flat.min()), float(flat.max())
        fig = Figure(figsize=(9, 7), dpi=100, facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG_PLOT)
        sc = ax.scatter(
            x_flat, y_flat, z_flat,
            c=flat, cmap=cmap, vmin=v_lo, vmax=v_hi,
            s=2, alpha=0.4, edgecolors="none",
        )
        for sym, (nx, ny, nz) in nuclei:
            ax.scatter(nx, ny, nz, c="red", s=120, edgecolors="darkred",
                       linewidths=0.8, zorder=5)
            ax.text(nx, ny, nz, f"  {sym}", fontsize=8, color="white",
                    fontweight="bold", zorder=6)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label(cbar_label, color=FG, fontsize=10)
        cbar.ax.tick_params(colors=FG, labelsize=8)
        ax.set_xlabel("x (Bohr)", color=FG, fontsize=9)
        ax.set_ylabel("y (Bohr)", color=FG, fontsize=9)
        ax.set_zlabel("z (Bohr)", color=FG, fontsize=9)
        ax.set_title(title, color=FG, fontsize=12, pad=14)
        ax.tick_params(colors=FG, labelsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    if parent:
        win = tk.Toplevel(parent)
        win.transient(parent)
    else:
        win = tk.Tk()
    win.title(f"Schr\u00f6dinger cloud — {mol_name}" if mol_name else "Schr\u00f6dinger cloud")
    win.geometry("1050x780")
    win.configure(bg=BG)

    toolbar = tk.Frame(win, bg="#313244", height=44)
    toolbar.pack(fill="x", padx=6, pady=(6, 0))
    plot_frame = tk.Frame(win, bg=BG)
    plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def show_V():
        for w in plot_frame.winfo_children():
            w.destroy()
        _plot_heatmap(plot_frame, V_vec,
                     f"Potential V(r) — {n}x{n}x{n}  ({n_total} points)",
                     "V (Hartree)", cmap="inferno")

    def show_orbital(i):
        def _():
            for w in plot_frame.winfo_children():
                w.destroy()
            E = eigenvalues[i] if i < len(eigenvalues) else 0
            _plot_heatmap(plot_frame, prob_list[i],
                         f"|psi_{i}|^2  (E{i} = {E:+.4f} Ha)",
                         f"|psi_{i}|^2", cmap="hot")
        return _

    def show_combined():
        for w in plot_frame.winfo_children():
            w.destroy()
        _plot_heatmap(plot_frame, prob_total,
                     f"Electron density (sum of {num_states} orbitals)",
                     "density", cmap="hot")

    btn_V = tk.Button(toolbar, text="Potential V(r)", bg="#89b4fa", fg="#1e1e2e",
                      font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                      command=show_V)
    btn_V.pack(side="left", padx=(10, 6), pady=6, ipady=3, ipadx=8)

    tk.Label(toolbar, text="Orbital:", bg="#313244", fg=FG,
             font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
    for i in range(num_states):
        b = tk.Button(toolbar, text=f"|psi_{i}|^2", bg="#a6e3a1", fg="#1e1e2e",
                      font=("Segoe UI", 9), relief="flat", cursor="hand2",
                      command=show_orbital(i))
        b.pack(side="left", padx=2, pady=6, ipady=2, ipadx=4)

    btn_comb = tk.Button(toolbar, text="Combined", bg="#f9e2af", fg="#1e1e2e",
                         font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                         command=show_combined)
    btn_comb.pack(side="left", padx=(8, 6), pady=6, ipady=3, ipadx=8)

    show_V()
    return win


# ─────────────────────  Standalone script  ─────────────────────
if __name__ == "__main__":
    nuclei = load_nuclei()
    if not nuclei:
        raise SystemExit("nuclei_state.json is empty. Load a molecule first.")
    data = compute_schrodinger_cloud(nuclei)
    win = open_schrodinger_cloud_window(data, parent=None, mol_name="(nuclei_state.json)")
    win.mainloop()
