"""
Quantum Simulator UI - Python
Right sidebar: periodic table (1-82), add atoms with x,y,z. Left: 3D nuclei view.
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from periodic_data import ELEMENTS_1_82


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
        self._setup_sidebar()

        self.element_blocks = []
        self._build_periodic_sidebar()
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
        tk.Button(self.plot_frame, text="Update 3D view", command=self._update_3d).pack(pady=4)

    def _setup_sidebar(self):
        tk.Label(self.sidebar_frame, text="Periodic table (1–82)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.scrollable = ScrollableSidebar(self.sidebar_frame, width=260)
        self.scrollable.pack(fill="both", expand=True)

    def _build_periodic_sidebar(self):
        inner = self.scrollable.inner
        for z, symbol in enumerate(ELEMENTS_1_82, start=1):
            block = ElementBlock(inner, z, symbol, self._update_3d)
            block.pack(anchor="w", pady=2)
            self.element_blocks.append(block)

    def _collect_nuclei(self):
        out = []
        for block in self.element_blocks:
            out.extend(block.get_nuclei())
        return out

    def _update_3d(self):
        self.ax.clear()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_title("Nuclei positions")
        nuclei = self._collect_nuclei()
        if not nuclei:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            self.canvas.draw_idle()
            return
        xs = [n[1][0] for n in nuclei]
        ys = [n[1][1] for n in nuclei]
        zs = [n[1][2] for n in nuclei]
        symbols = [n[0] for n in nuclei]
        # Tiny dots with symbol inside: small scatter + text at same position
        self.ax.scatter(xs, ys, zs, c="steelblue", s=120, alpha=0.95, edgecolors="navy", linewidths=0.8)
        for sym, (x, y, z) in zip(symbols, zip(xs, ys, zs)):
            self.ax.text(x, y, z, f" {sym} ", fontsize=7, ha="center", va="center", color="white", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor="steelblue", edgecolor="navy", alpha=0.9))
        margin = max(0.5, (max(xs) - min(xs) + max(ys) - min(ys) + max(zs) - min(zs)) / 3 * 0.2)
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.ax.set_zlim(min(zs) - margin, max(zs) + margin)
        self.canvas.draw_idle()


def main():
    App()


if __name__ == "__main__":
    main()