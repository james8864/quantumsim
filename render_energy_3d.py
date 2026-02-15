"""
Render the potential energy V(x,y,z) from potential_3d as 2-D slice heatmaps.

Three panels show the actual V = sum_i(-Z_i / |r - R_i|) evaluated on a
dense grid via ``potential_on_grid``:

  • xy slice  (z = z₀)
  • xz slice  (y = y₀)
  • yz slice  (x = x₀)

Slices pass through the centroid of the nuclei by default.
A slider lets the user move the slice position in real time.
Nuclei that lie near the slice plane are marked as circles.
"""
import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from potential_3d import (
    potential_3d,
    potential_on_grid,
    SYMBOL_TO_Z,
    load_nuclei_from_app,
)


# ─────────────────────  Grid helper  ─────────────────────────
def _adaptive_grid(nuclei, n_pts=120, pad=6.0):
    """
    Dense grid that encloses all nuclei with padding.
    Returns (x_1d, y_1d, z_1d, center).
    """
    if not nuclei:
        lin = np.linspace(-5, 5, n_pts)
        return lin, lin, lin, (0.0, 0.0, 0.0)

    xs = [p[1][0] for p in nuclei]
    ys = [p[1][1] for p in nuclei]
    zs = [p[1][2] for p in nuclei]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys),
               max(zs) - min(zs), 0.0)
    half = span / 2.0 + pad

    return (np.linspace(cx - half, cx + half, n_pts),
            np.linspace(cy - half, cy + half, n_pts),
            np.linspace(cz - half, cz + half, n_pts),
            (cx, cy, cz))


# ─────────────────────  Color range  ─────────────────────────
def _clamp_range(V_3d):
    """Pick a nice color range that shows structure without singularity."""
    flat = V_3d.ravel()
    v_lo = np.percentile(flat, 1)
    v_hi = np.percentile(flat, 99)
    if v_lo >= v_hi:
        v_lo, v_hi = flat.min(), flat.max()
    if v_lo >= v_hi:
        v_lo, v_hi = -1.0, 0.0
    return v_lo, v_hi


# ─────────────────────  Nearest index  ───────────────────────
def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


# ─────────────────────  Main renderer  ───────────────────────
def render_energy_3d(
    nuclei=None,
    cmap_name="RdBu_r",
    show_nuclei=True,
    title="Énergie potentielle V(x,y,z) — coupes 2D",
    show_point_query=True,
):
    """
    Display the potential from potential_3d as three 2-D slice heatmaps.

    nuclei : list of (symbol, (x,y,z)) or None → load from nuclei_state.json
    """
    if nuclei is None:
        nuclei = load_nuclei_from_app()
        if not nuclei:
            print("Aucun noyau dans nuclei_state.json. Exemple H2.")
            nuclei = [("H", (-0.7, 0.0, 0.0)), ("H", (0.7, 0.0, 0.0))]

    # ── evaluate V on a dense grid ──
    x_1d, y_1d, z_1d, (cx, cy, cz) = _adaptive_grid(nuclei, n_pts=120, pad=6.0)
    X, Y, Z, V_grid = potential_on_grid(nuclei, x_1d, y_1d, z_1d)
    V_func = potential_3d(nuclei)
    v_lo, v_hi = _clamp_range(V_grid)

    n_contours = 30

    # ── Tkinter window ──
    root = tk.Tk()
    root.wm_title("Potentiel V(r) — coupes 2D")
    root.geometry("1300x780")
    root.configure(bg="#1e1e2e")

    # ── figure: 3 panels ──
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(
        1, 3, figsize=(14, 4.5), dpi=100,
        facecolor="#1e1e2e")
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.12, top=0.88,
                        wspace=0.35)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── state: current slice positions ──
    slice_z_val = [cz]   # mutable for closures
    slice_y_val = [cy]
    slice_x_val = [cx]

    # ── drawing function ──
    def draw_slices():
        sz, sy, sx = slice_z_val[0], slice_y_val[0], slice_x_val[0]

        iz = _nearest_idx(z_1d, sz)
        iy = _nearest_idx(y_1d, sy)
        ix = _nearest_idx(x_1d, sx)

        for a in (ax_xy, ax_xz, ax_yz):
            a.clear()

        # ── xy slice (z = sz) ──
        Vxy = V_grid[:, :, iz]  # shape (nx, ny)
        im1 = ax_xy.pcolormesh(x_1d, y_1d, Vxy.T,
                                cmap=cmap_name, vmin=v_lo, vmax=v_hi,
                                shading="auto")
        ax_xy.contour(x_1d, y_1d, Vxy.T, levels=n_contours,
                       colors="k", linewidths=0.3, alpha=0.5)
        ax_xy.set_xlabel("x (Bohr)")
        ax_xy.set_ylabel("y (Bohr)")
        ax_xy.set_title(f"Coupe xy  (z = {sz:.2f})", color="#cdd6f4", fontsize=10)
        ax_xy.set_aspect("equal")
        ax_xy.set_facecolor("#181825")
        ax_xy.tick_params(colors="#cdd6f4", labelsize=7)
        for spine in ax_xy.spines.values():
            spine.set_color("#45475a")

        # ── xz slice (y = sy) ──
        Vxz = V_grid[:, iy, :]  # shape (nx, nz)
        ax_xz.pcolormesh(x_1d, z_1d, Vxz.T,
                          cmap=cmap_name, vmin=v_lo, vmax=v_hi,
                          shading="auto")
        ax_xz.contour(x_1d, z_1d, Vxz.T, levels=n_contours,
                       colors="k", linewidths=0.3, alpha=0.5)
        ax_xz.set_xlabel("x (Bohr)")
        ax_xz.set_ylabel("z (Bohr)")
        ax_xz.set_title(f"Coupe xz  (y = {sy:.2f})", color="#cdd6f4", fontsize=10)
        ax_xz.set_aspect("equal")
        ax_xz.set_facecolor("#181825")
        ax_xz.tick_params(colors="#cdd6f4", labelsize=7)
        for spine in ax_xz.spines.values():
            spine.set_color("#45475a")

        # ── yz slice (x = sx) ──
        Vyz = V_grid[ix, :, :]  # shape (ny, nz)
        ax_yz.pcolormesh(y_1d, z_1d, Vyz.T,
                          cmap=cmap_name, vmin=v_lo, vmax=v_hi,
                          shading="auto")
        ax_yz.contour(y_1d, z_1d, Vyz.T, levels=n_contours,
                       colors="k", linewidths=0.3, alpha=0.5)
        ax_yz.set_xlabel("y (Bohr)")
        ax_yz.set_ylabel("z (Bohr)")
        ax_yz.set_title(f"Coupe yz  (x = {sx:.2f})", color="#cdd6f4", fontsize=10)
        ax_yz.set_aspect("equal")
        ax_yz.set_facecolor("#181825")
        ax_yz.tick_params(colors="#cdd6f4", labelsize=7)
        for spine in ax_yz.spines.values():
            spine.set_color("#45475a")

        # ── nuclei markers (project onto slices if close) ──
        if show_nuclei:
            tol = (x_1d[-1] - x_1d[0]) / len(x_1d) * 2.5  # ~2.5 grid cells
            for sym, (nx, ny, nz) in nuclei:
                # xy slice: show if |nz - sz| < tol
                if abs(nz - sz) < tol:
                    ax_xy.plot(nx, ny, "o", color="red", markersize=7,
                               markeredgecolor="darkred", markeredgewidth=1)
                    ax_xy.text(nx, ny, f" {sym}", fontsize=7, color="white",
                               fontweight="bold")
                # xz slice: show if |ny - sy| < tol
                if abs(ny - sy) < tol:
                    ax_xz.plot(nx, nz, "o", color="red", markersize=7,
                               markeredgecolor="darkred", markeredgewidth=1)
                    ax_xz.text(nx, nz, f" {sym}", fontsize=7, color="white",
                               fontweight="bold")
                # yz slice: show if |nx - sx| < tol
                if abs(nx - sx) < tol:
                    ax_yz.plot(ny, nz, "o", color="red", markersize=7,
                               markeredgecolor="darkred", markeredgewidth=1)
                    ax_yz.text(ny, nz, f" {sym}", fontsize=7, color="white",
                               fontweight="bold")

        canvas.draw_idle()

    # ── colorbar (draw once) ──
    sm = cm.ScalarMappable(cmap=cmap_name,
                           norm=plt.Normalize(vmin=v_lo, vmax=v_hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_xy, ax_xz, ax_yz],
                        shrink=0.85, pad=0.03, aspect=30)
    cbar.set_label("V (Ha)", color="#cdd6f4", fontsize=9)
    cbar.ax.tick_params(colors="#cdd6f4", labelsize=7)

    # ── control panel ──
    ctrl = tk.Frame(root, bg="#313244")
    ctrl.pack(fill="x", padx=6, pady=(0, 6))

    # Slider ranges
    z_lo, z_hi = float(z_1d[0]), float(z_1d[-1])
    y_lo, y_hi = float(y_1d[0]), float(y_1d[-1])
    x_lo_v, x_hi_v = float(x_1d[0]), float(x_1d[-1])

    def _make_slider(parent, label, lo, hi, init, callback):
        f = tk.Frame(parent, bg="#313244")
        f.pack(side="left", padx=12, pady=4)
        tk.Label(f, text=label, bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI", 9)).pack(side="left")
        var = tk.DoubleVar(value=init)
        s = tk.Scale(f, from_=lo, to=hi, resolution=0.05,
                     orient="horizontal", length=200,
                     variable=var, command=callback,
                     bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
                     highlightthickness=0)
        s.pack(side="left", padx=4)
        return var

    def on_z(val):
        slice_z_val[0] = float(val)
        draw_slices()

    def on_y(val):
        slice_y_val[0] = float(val)
        draw_slices()

    def on_x(val):
        slice_x_val[0] = float(val)
        draw_slices()

    _make_slider(ctrl, "z coupe xy :", z_lo, z_hi, cz, on_z)
    _make_slider(ctrl, "y coupe xz :", y_lo, y_hi, cy, on_y)
    _make_slider(ctrl, "x coupe yz :", x_lo_v, x_hi_v, cx, on_x)

    # ── point query ──
    if show_point_query:
        qf = tk.LabelFrame(root, text="V au point (x, y, z)",
                           font=("Segoe UI", 10, "bold"),
                           bg="#313244", fg="#cdd6f4")
        qf.pack(fill="x", padx=6, pady=(0, 6))
        row = tk.Frame(qf, bg="#313244")
        row.pack(anchor="w", pady=4, padx=4)

        entries = {}
        for name in ("x", "y", "z"):
            tk.Label(row, text=f"{name}:", bg="#313244",
                     fg="#cdd6f4").pack(side="left", padx=(0, 2))
            var = tk.StringVar(value="0")
            tk.Entry(row, textvariable=var, width=8,
                     bg="#45475a", fg="#cdd6f4",
                     insertbackground="#cdd6f4",
                     relief="flat").pack(side="left", padx=(0, 10))
            entries[name] = var

        result_lbl = tk.Label(row, text="V = —",
                              font=("Consolas", 10),
                              bg="#313244", fg="#89dceb")
        result_lbl.pack(side="left", padx=(16, 0))

        def query_v():
            try:
                xv = float(entries["x"].get().strip() or 0)
                yv = float(entries["y"].get().strip() or 0)
                zv = float(entries["z"].get().strip() or 0)
            except ValueError:
                result_lbl.config(text="V = (invalide)")
                return
            v = V_func(xv, yv, zv)
            v = float(np.atleast_1d(v).ravel()[0])
            result_lbl.config(text=f"V = {v:.6f} Ha")

        tk.Button(row, text="Calculer",
                  bg="#89b4fa", fg="#1e1e2e",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2",
                  command=query_v).pack(side="left", padx=(8, 0))

    # ── initial draw ──
    draw_slices()
    root.mainloop()


if __name__ == "__main__":
    render_energy_3d()