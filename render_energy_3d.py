"""
Render the 3D potential (energy) from nuclei.
No boundaries: uses parametric spheres for Coulomb isosurfaces V = -Z/r.
Potential decays naturally (eclipses) with distance - no limits or boxes.
"""
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from potential_3d import potential_3d, SYMBOL_TO_Z, load_nuclei_from_app


def _sphere_mesh(cx, cy, cz, r, n_theta=32, n_phi=16):
    """Parametric sphere centered at (cx,cy,cz) with radius r. No bounds."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    T, P = np.meshgrid(theta, phi)
    x = r * np.sin(P) * np.cos(T) + cx
    y = r * np.sin(P) * np.sin(T) + cy
    z = r * np.cos(P) + cz
    return x, y, z


def _build_3d_figure(nuclei, cmap_name, show_nuclei, title):
    """
    Draw potential as concentric spheres per nucleus. No grid, no box, no boundaries.
    V = -Z/r → isosurface at V0 is sphere of radius r = -Z/V0. Decays naturally outward.
    """
    if not nuclei:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    try:
        cmap = plt.colormaps[cmap_name]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap(cmap_name)

    v_min, v_max = -5.0, -0.02
    for symbol, (cx, cy, cz) in nuclei:
        Z = SYMBOL_TO_Z.get(symbol, 1)
        n_shells = 8
        # Radii from 0.5 to 40 Bohr - potential eclipses naturally with distance, no cutoff
        radii = np.linspace(0.5, 40, n_shells)
        for i, r in enumerate(radii):
            if r < 0.05:
                continue
            v0 = -Z / r
            x, y, z = _sphere_mesh(cx, cy, cz, r)
            norm = plt.Normalize(vmin=v_min, vmax=v_max)
            # Outer shells more transparent - natural eclipse
            alpha = max(0.15, 0.5 - 0.04 * i)
            rgba = np.array(cmap(norm(float(v0)))).reshape(1, 1, -1)
            facecolors = np.broadcast_to(rgba, (x.shape[0] - 1, x.shape[1] - 1, 4))
            ax.plot_surface(x, y, z, facecolors=facecolors, alpha=alpha, shade=False)

    if show_nuclei:
        nx = [n[1][0] for n in nuclei]
        ny = [n[1][1] for n in nuclei]
        nz = [n[1][2] for n in nuclei]
        ax.scatter(nx, ny, nz, c="red", s=80, marker="o", edgecolors="darkred", label="Nuclei")

    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
    ax.set_zlabel("z (Bohr)")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    norm = plt.Normalize(vmin=v_min, vmax=v_max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("V (a.u.)")
    fig.tight_layout()
    return fig, ax


def render_energy_3d(
    nuclei=None,
    cmap_name="viridis",
    show_nuclei=True,
    title="Potential energy V(x,y,z)",
    show_point_query=True,
):
    """
    Plot the potential energy as concentric spheres (no boundaries; decays naturally).
    If nuclei is None, loads positions from app.py state (nuclei_state.json).

    nuclei: list of (symbol, (x, y, z)) or None to load from app sync file
    cmap_name: matplotlib colormap (e.g. 'viridis', 'plasma', 'coolwarm')
    show_nuclei: if True, scatter the nucleus positions on the plot
    title: plot title
    show_point_query: if True, show panel to enter x,y,z and get potential at that point
    """
    if nuclei is None:
        nuclei = load_nuclei_from_app()
        if not nuclei:
            print("No nuclei in app state (run app.py and use 'Sync nuclei to potential' first). Using example.")
            nuclei = [("H", (-0.5, 0.0, 0.0)), ("H", (0.5, 0.0, 0.0))]
    V_func = potential_3d(nuclei)

    if not show_point_query:
        fig, _ = _build_3d_figure(nuclei, cmap_name, show_nuclei, title)
        plt.show()
        return

    root = tk.Tk()
    root.wm_title("Potential energy – 3D colormap")
    root.geometry("900x720")

    fig, ax = _build_3d_figure(nuclei, cmap_name, show_nuclei, title)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    query_frame = tk.LabelFrame(root, text="Potential at point (x, y, z)", font=("Segoe UI", 10, "bold"))
    query_frame.pack(side="bottom", fill="x", padx=8, pady=8)
    row = tk.Frame(query_frame)
    row.pack(anchor="w", pady=4)
    tk.Label(row, text="x:").pack(side="left", padx=(0, 4))
    x_var = tk.StringVar(value="0")
    tk.Entry(row, textvariable=x_var, width=10).pack(side="left", padx=(0, 12))
    tk.Label(row, text="y:").pack(side="left", padx=(0, 4))
    y_var = tk.StringVar(value="0")
    tk.Entry(row, textvariable=y_var, width=10).pack(side="left", padx=(0, 12))
    tk.Label(row, text="z:").pack(side="left", padx=(0, 4))
    z_var = tk.StringVar(value="0")
    tk.Entry(row, textvariable=z_var, width=10).pack(side="left", padx=(0, 12))
    result_label = tk.Label(row, text="V = —", font=("Segoe UI", 10))
    result_label.pack(side="left", padx=(16, 0))

    def get_potential():
        try:
            x = float(x_var.get().strip() or 0)
            y = float(y_var.get().strip() or 0)
            z = float(z_var.get().strip() or 0)
        except ValueError:
            result_label.config(text="V = (invalid x,y,z)")
            return
        v = V_func(x, y, z)
        v = float(np.atleast_1d(v).ravel()[0])
        result_label.config(text=f"V = {v:.6f} a.u.")

    tk.Button(query_frame, text="Get potential", command=get_potential).pack(anchor="w", pady=(0, 4))
    root.mainloop()


if __name__ == "__main__":
    render_energy_3d(nuclei=None, cmap_name="viridis", show_point_query=True)
