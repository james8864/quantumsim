"""
Render the 3D potential (energy) from nuclei as a colormap in 3D.
Uses nuclei synced from app.py (nuclei_state.json). Includes a panel to query V(x,y,z) at any point.
"""
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from potential_3d import potential_on_grid, potential_3d, load_nuclei_from_app


def _bounding_box(nuclei, padding=2.0):
    """Return (x_min, x_max, y_min, y_max, z_min, z_max) with padding."""
    if not nuclei:
        return -3, 3, -3, 3, -3, 3
    xs = [n[1][0] for n in nuclei]
    ys = [n[1][1] for n in nuclei]
    zs = [n[1][2] for n in nuclei]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    half = max(
        max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0
    ) / 2 + padding
    return (
        cx - half, cx + half,
        cy - half, cy + half,
        cz - half, cz + half,
    )


def _build_3d_figure(nuclei, n_grid, slice_center, cmap_name, show_nuclei, title):
    """Build the 3D colormap figure (no display). Returns (fig, ax)."""
    if not nuclei:
        x_min, x_max, y_min, y_max, z_min, z_max = -3, 3, -3, 3, -3, 3
        cx = cy = cz = 0.0
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = _bounding_box(nuclei)
        xs = [n[1][0] for n in nuclei]
        ys = [n[1][1] for n in nuclei]
        zs = [n[1][2] for n in nuclei]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2

    if slice_center is None:
        slice_center = (cx, cy, cz)
    x0, y0, z0 = slice_center

    x_1d = np.linspace(x_min, x_max, n_grid)
    y_1d = np.linspace(y_min, y_max, n_grid)
    z_1d = np.linspace(z_min, z_max, n_grid)

    X, Y, Z, V = potential_on_grid(nuclei, x_1d, y_1d, z_1d)
    v_min, v_max = np.nanmin(V), np.nanmax(V)
    if v_min >= v_max:
        v_max = v_min + 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        cmap = plt.colormaps[cmap_name]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=v_min, vmax=v_max)

    # Plane y = y0: (x, z) vary -> X and Z 2D, Y constant
    ix_y = np.argmin(np.abs(y_1d - y0))
    V_y_slice = V[:, ix_y, :]  # (nx, nz)
    X_y, Z_y = np.meshgrid(x_1d, z_1d, indexing="ij")
    Y_y = np.full_like(X_y, y0)
    ax.plot_surface(
        X_y, Y_y, Z_y, facecolors=cmap(norm(V_y_slice)), rstride=1, cstride=1,
        shade=False, antialiased=False
    )

    # Plane z = z0: (x, y) vary
    ix_z = np.argmin(np.abs(z_1d - z0))
    V_z_slice = V[:, :, ix_z]  # (nx, ny)
    X_z, Y_z = np.meshgrid(x_1d, y_1d, indexing="ij")
    Z_z = np.full_like(X_z, z0)
    ax.plot_surface(
        X_z, Y_z, Z_z, facecolors=cmap(norm(V_z_slice)), rstride=1, cstride=1,
        shade=False, antialiased=False
    )

    # Plane x = x0: (y, z) vary
    ix_x = np.argmin(np.abs(x_1d - x0))
    V_x_slice = V[ix_x, :, :]  # (ny, nz)
    Y_x, Z_x = np.meshgrid(y_1d, z_1d, indexing="ij")
    X_x = np.full_like(Y_x, x0)
    ax.plot_surface(
        X_x, Y_x, Z_x, facecolors=cmap(norm(V_x_slice)), rstride=1, cstride=1,
        shade=False, antialiased=False
    )

    if show_nuclei and nuclei:
        nx = [n[1][0] for n in nuclei]
        ny = [n[1][1] for n in nuclei]
        nz = [n[1][2] for n in nuclei]
        ax.scatter(nx, ny, nz, c="red", s=80, marker="o", edgecolors="darkred", label="Nuclei")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("V (a.u.)")
    fig.tight_layout()
    return fig, ax


def render_energy_3d(
    nuclei=None,
    n_grid=40,
    slice_center=None,
    cmap_name="viridis",
    show_nuclei=True,
    title="Potential energy V(x,y,z)",
    show_point_query=True,
):
    """
    Plot the potential energy on three orthogonal slice planes in 3D with a colormap.
    If nuclei is None, loads positions from app.py state (nuclei_state.json).
    When show_point_query is True, opens a window with x,y,z entry and displays V(x,y,z).

    nuclei: list of (symbol, (x, y, z)) or None to load from app sync file
    n_grid: number of grid points per axis (smaller = faster)
    slice_center: (x0, y0, z0) for the three planes; default is centroid of nuclei
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
        fig, _ = _build_3d_figure(nuclei, n_grid, slice_center, cmap_name, show_nuclei, title)
        plt.show()
        return

    root = tk.Tk()
    root.wm_title("Potential energy – 3D colormap")
    root.geometry("900x720")

    fig, ax = _build_3d_figure(nuclei, n_grid, slice_center, cmap_name, show_nuclei, title)
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
        v = float(np.asarray(v).flat[0])
        result_label.config(text=f"V = {v:.6f} a.u.")

    tk.Button(query_frame, text="Get potential", command=get_potential).pack(anchor="w", pady=(0, 4))
    root.mainloop()


if __name__ == "__main__":
    # Load from app sync file if available; else use example nuclei
    render_energy_3d(
        nuclei=None,
        n_grid=35,
        cmap_name="viridis",
        title="Potential energy V(x,y,z) (Coulomb)",
        show_point_query=True,
    )
