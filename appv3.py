"""
Quantum Simulator UI - Python
Right sidebar: periodic table (1-82), add atoms with x,y,z. Left: 3D nuclei view.
Schrödinger solver: energy levels and position probability per orbital (dots = higher probability).
"""
import json
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from periodic_data import ELEMENTS_1_82
from potential_3d import NUCLEI_STATE_PATH
from schrodinger_3d import solve_schrodinger, sample_positions_from_probability


class ScrollableSidebar(tk.Frame):
    """Scrollable frame for the periodic table sidebar."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas)
        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)


class NucleusRow:
    """One nucleus: symbol + x,y,z entries."""
    def __init__(self, parent, symbol, on_coord_change):
        self.symbol = symbol
        self.on_coord_change = on_coord_change
        self.frame = tk.Frame(parent)
        self.x_var = tk.StringVar(value="0")
        self.y_var = tk.StringVar(value="0")
        self.z_var = tk.StringVar(value="0")
        tk.Label(self.frame, text="x:", width=2).pack(side="left", padx=(8, 0))
        self.x_entry = tk.Entry(self.frame, textvariable=self.x_var, width=6)
        self.x_entry.pack(side="left", padx=2)
        tk.Label(self.frame, text="y:", width=2).pack(side="left", padx=(4, 0))
        self.y_entry = tk.Entry(self.frame, textvariable=self.y_var, width=6)
        self.y_entry.pack(side="left", padx=2)
        tk.Label(self.frame, text="z:", width=2).pack(side="left", padx=(4, 0))
        self.z_entry = tk.Entry(self.frame, textvariable=self.z_var, width=6)
        self.z_entry.pack(side="left", padx=2)
        for v in (self.x_var, self.y_var, self.z_var):
            v.trace_add("write", lambda *_: on_coord_change())

    def get_xyz(self):
        try:
            x = float(self.x_var.get().strip() or 0)
        except ValueError:
            x = 0
        try:
            y = float(self.y_var.get().strip() or 0)
        except ValueError:
            y = 0
        try:
            z = float(self.z_var.get().strip() or 0)
        except ValueError:
            z = 0
        return x, y, z

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def destroy(self):
        self.frame.destroy()


class ElementBlock:
    """One element in the sidebar: symbol, plus button, and optional nucleus rows."""
    def __init__(self, parent, z, symbol, on_add_nucleus):
        self.z = z
        self.symbol = symbol
        self.on_add_nucleus = on_add_nucleus
        self.nucleus_rows = []
        self.frame = tk.Frame(parent)
        self.row_frame = tk.Frame(self.frame)  # holds symbol + plus
        self.row_frame.pack(anchor="w")
        tk.Label(self.row_frame, text=f"{z}", width=3, anchor="e").pack(side="left", padx=(0, 4))
        tk.Label(self.row_frame, text=symbol, width=3, font=("Segoe UI", 10, "bold")).pack(side="left", padx=2)
        self.plus_btn = tk.Button(
            self.row_frame, text="+", width=2, cursor="hand2",
            command=self._add_nucleus
        )
        self.plus_btn.pack(side="left", padx=2)
        self.coords_frame = tk.Frame(self.frame)  # holds all x,y,z rows for this element
        self.coords_frame.pack(anchor="w", fill="x")

    def _add_nucleus(self):
        row = NucleusRow(self.coords_frame, self.symbol, self.on_add_nucleus)
        row.pack(anchor="w", pady=1)
        self.nucleus_rows.append(row)
        self.on_add_nucleus()

    def get_nuclei(self):
        return [(self.symbol, r.get_xyz()) for r in self.nucleus_rows]

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quantum Simulator – Nuclei (1–82)")
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)

        # Main layout: 3D left, sidebar right
        self.main = tk.PanedWindow(self.root, orient="horizontal")
        self.main.pack(fill="both", expand=True, padx=4, pady=4)

        # Left: 3D plot
        self.plot_frame = tk.Frame(self.main, width=700)
        self.plot_frame.pack(side="left", fill="both", expand=True)
        self._setup_3d_plot()

        # Right: sidebar
        self.sidebar_frame = tk.Frame(self.main, width=280)
        self.sidebar_frame.pack(side="right", fill="y", padx=(8, 0))
        self.num_electrons_var = tk.IntVar(value=0)
        self._setup_sidebar()

        self.element_blocks = []
        self._build_periodic_sidebar()

        # Schrödinger state (filled on "Compute energy levels")
        self.schrodinger_energies = []
        self.schrodinger_wavefunctions = []
        self.schrodinger_grid_info = {}
        self.selected_orbital_var = tk.IntVar(value=-1)  # -1 = nuclei only

        self._update_electron_label()
        self._update_3d()
        self.root.mainloop()

    def _setup_3d_plot(self):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title("Nuclei positions")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        btn_frame = tk.Frame(self.plot_frame)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Update 3D view", command=self._update_3d).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Sync nuclei to potential", command=self._sync_nuclei_to_potential).pack(side="left", padx=2)
        tk.Button(btn_frame, text="Reset atoms", command=self._reset_atoms).pack(side="left", padx=2)

    def _setup_sidebar(self):
        # Number of electrons slider
        electron_frame = tk.Frame(self.sidebar_frame)
        electron_frame.pack(anchor="w", pady=(0, 8))
        tk.Label(electron_frame, text="Number of electrons", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.electron_count_label = tk.Label(electron_frame, text="0", font=("Segoe UI", 9))
        self.electron_count_label.pack(anchor="w")
        self.electron_slider = tk.Scale(
            electron_frame, from_=0, to=500, orient="horizontal",
            variable=self.num_electrons_var, showvalue=0, length=220
        )
        self.electron_slider.pack(anchor="w", pady=(2, 0))
        self.num_electrons_var.trace_add("write", lambda *_: self._update_electron_label())

        # Schrödinger: compute levels, select orbital, show position probability
        schro_frame = tk.LabelFrame(self.sidebar_frame, text="Schrödinger", font=("Segoe UI", 10, "bold"))
        schro_frame.pack(anchor="w", pady=(0, 8))
        tk.Label(schro_frame, text="E0 = ground (most bound); higher = excited, farther out.", font=("Segoe UI", 8), fg="gray").pack(anchor="w")
        tk.Button(schro_frame, text="Compute energy levels", command=self._compute_schrodinger).pack(anchor="w", pady=2)
        self.energy_listbox = tk.Listbox(schro_frame, height=4, width=28, font=("Consolas", 8))
        self.energy_listbox.pack(anchor="w", pady=2)
        self.energy_listbox.bind("<<ListboxSelect>>", self._on_orbital_select)
        tk.Label(schro_frame, text="View position probability:", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        self.orbital_view_var = tk.StringVar(value="Nuclei only")
        self.orbital_combo = ttk.Combobox(schro_frame, textvariable=self.orbital_view_var, state="readonly", width=26)
        self.orbital_combo.pack(anchor="w", pady=2)
        self.orbital_combo.bind("<<ComboboxSelected>>", self._on_orbital_view_select)
        tk.Button(schro_frame, text="Show probability distribution", command=self._show_probability_dots).pack(anchor="w", pady=2)

        tk.Label(self.sidebar_frame, text="Periodic table (1–82)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.scrollable = ScrollableSidebar(self.sidebar_frame, width=260)
        self.scrollable.pack(fill="both", expand=True)

    def _build_periodic_sidebar(self):
        inner = self.scrollable.inner
        for z, symbol in enumerate(ELEMENTS_1_82, start=1):
            block = ElementBlock(inner, z, symbol, self._update_3d)
            block.pack(anchor="w", pady=2)
            self.element_blocks.append(block)

    def _update_electron_label(self):
        try:
            n = self.num_electrons_var.get()
        except tk.TclError:
            n = 0
        self.electron_count_label.config(text=str(n))

    def _collect_nuclei(self):
        out = []
        for block in self.element_blocks:
            out.extend(block.get_nuclei())
        return out

    def _sync_nuclei_to_potential(self):
        """Write current nucleus positions to nuclei_state.json for potential_3d and render."""
        nuclei = self._collect_nuclei()
        data = [{"symbol": s, "x": x, "y": y, "z": z} for s, (x, y, z) in nuclei]
        with open(NUCLEI_STATE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        self._update_3d()

    def _reset_atoms(self):
        """Remove all atoms from the sandbox and clear the 3D view."""
        for block in self.element_blocks:
            for row in block.nucleus_rows:
                row.destroy()
            block.nucleus_rows.clear()
        self.schrodinger_energies = []
        self.schrodinger_wavefunctions = []
        self.schrodinger_grid_info = {}
        self.selected_orbital_var.set(-1)
        self.energy_listbox.delete(0, tk.END)
        self.orbital_combo["values"] = ["Nuclei only"]
        self.orbital_combo.set("Nuclei only")
        with open(NUCLEI_STATE_PATH, "w") as f:
            json.dump([], f, indent=2)
        self._update_3d()

    def _compute_schrodinger(self):
        """Solve Schrödinger equation for current nuclei; fill energy levels and orbital list."""
        nuclei = self._collect_nuclei()
        if not nuclei:
            messagebox.showinfo("No nuclei", "Add at least one nucleus (periodic table +) and set coordinates.")
            return
        try:
            n = min(10, 8)
            energies, wavefunctions, grid_info = solve_schrodinger(nuclei, n_grid=26, num_states=n)
        except Exception as e:
            messagebox.showerror("Schrödinger solver", str(e))
            return
        if len(energies) == 0:
            messagebox.showwarning("No states", "Solver returned no eigenstates. Try a smaller grid or check nuclei.")
            return
        self.schrodinger_energies = energies
        self.schrodinger_wavefunctions = wavefunctions
        self.schrodinger_grid_info = grid_info
        self.energy_listbox.delete(0, tk.END)
        for i, E in enumerate(energies):
            self.energy_listbox.insert(tk.END, f"E{i} = {E:.5f} Ha")
        self.orbital_combo["values"] = ["Nuclei only"] + [f"Orbital {i} (E={energies[i]:.4f})" for i in range(len(energies))]
        self.orbital_combo.set("Nuclei only")
        self.selected_orbital_var.set(-1)
        self._update_3d()

    def _on_orbital_select(self, event):
        pass

    def _on_orbital_view_select(self, event):
        """User chose an orbital from the combo; show its probability if not 'Nuclei only'."""
        sel = self.orbital_combo.get()
        if sel == "Nuclei only":
            self.selected_orbital_var.set(-1)
        else:
            try:
                i = int(sel.split()[1])
                self.selected_orbital_var.set(i)
            except (IndexError, ValueError):
                self.selected_orbital_var.set(-1)
        self._update_3d()

    def _show_probability_dots(self):
        """Switch view to selected orbital's position probability (dots); if none selected use first."""
        if not self.schrodinger_wavefunctions:
            messagebox.showinfo("Compute first", "Click 'Compute energy levels' first.")
            return
        cur = self.orbital_combo.get()
        if cur == "Nuclei only":
            self.orbital_combo.set(self.orbital_combo["values"][1])
            self.selected_orbital_var.set(0)
        else:
            try:
                i = int(cur.split()[1])
                self.selected_orbital_var.set(i)
            except (IndexError, ValueError):
                self.selected_orbital_var.set(0)
        self._update_3d()

    def _update_3d(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        nuclei = self._collect_nuclei()
        show_orbital = self.selected_orbital_var.get()
        if show_orbital >= 0 and show_orbital < len(self.schrodinger_wavefunctions) and self.schrodinger_grid_info:
            self.ax.set_title("Position probability (dots) — Orbital " + str(show_orbital))
        else:
            self.ax.set_title("Nuclei positions")

        if not nuclei and show_orbital < 0:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            self.canvas.draw_idle()
            return

        xs = [n[1][0] for n in nuclei] if nuclei else []
        ys = [n[1][1] for n in nuclei] if nuclei else []
        zs = [n[1][2] for n in nuclei] if nuclei else []
        symbols = [n[0] for n in nuclei] if nuclei else []

        # Draw position probability dots for selected orbital (more dots = higher probability)
        if show_orbital >= 0 and show_orbital < len(self.schrodinger_wavefunctions) and self.schrodinger_grid_info:
            np.random.seed(42)
            px, py, pz = sample_positions_from_probability(
                self.schrodinger_wavefunctions[show_orbital],
                self.schrodinger_grid_info,
                num_dots=2500,
                jitter=0.6,
            )
            self.ax.scatter(px, py, pz, c="darkgreen", s=4, alpha=0.5, label="|ψ|²")

        # Nuclei: dots with symbol
        if nuclei:
            self.ax.scatter(xs, ys, zs, c="steelblue", s=120, alpha=0.95, edgecolors="navy", linewidths=0.8)
            for sym, (x, y, z) in zip(symbols, zip(xs, ys, zs)):
                self.ax.text(x, y, z, f" {sym} ", fontsize=7, ha="center", va="center", color="white", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.15", facecolor="steelblue", edgecolor="navy", alpha=0.9))

        if nuclei or (show_orbital >= 0 and self.schrodinger_grid_info):
            all_x = list(xs) + (list(px) if show_orbital >= 0 and show_orbital < len(self.schrodinger_wavefunctions) else [])
            all_y = list(ys) + (list(py) if show_orbital >= 0 and show_orbital < len(self.schrodinger_wavefunctions) else [])
            all_z = list(zs) + (list(pz) if show_orbital >= 0 and show_orbital < len(self.schrodinger_wavefunctions) else [])
            if all_x:
                margin = 0.5
                x_min, x_max = min(all_x) - margin, max(all_x) + margin
                y_min, y_max = min(all_y) - margin, max(all_y) + margin
                z_min, z_max = min(all_z) - margin, max(all_z) + margin
                # Equal aspect so orbitals look spherical, not box-like
                r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                cz = (z_min + z_max) / 2
                self.ax.set_xlim(cx - r, cx + r)
                self.ax.set_ylim(cy - r, cy + r)
                self.ax.set_zlim(cz - r, cz + r)
        self.canvas.draw_idle()


def main():
    App()


if __name__ == "__main__":
    main()
