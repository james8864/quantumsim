"""
Simulateur quantique — Python / Tkinter

Barre latérale droite : tableau périodique (1–82), ajout/suppression de noyaux x,y,z.
Gauche               : graphique 3-D (noyaux, probabilité orbitale, densité, PES).
Schrödinger          : solveur SCF avec répulsion e⁻–e⁻, exclusion de Pauli,
                       niveaux d'énergie, densité combinée, potentiel électrostatique.
"""
import json
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from periodic_data import ELEMENTS_1_82
from potential_3d import NUCLEI_STATE_PATH
from molecules import MOLECULE_LIBRARY
from schrodinger_3d import (
    solve_schrodinger,
    sample_positions_from_probability,
    get_esp_at_points,
)


# ─────────────────────────  Barre latérale défilante  ─────────
class ScrollableSidebar(tk.Frame):
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


# ─────────────────────────  Ligne de noyau  ──────────────────
class NucleusRow:
    """Un noyau : symbole + champs x,y,z + bouton supprimer."""
    def __init__(self, parent, symbol, on_coord_change, on_remove):
        self.symbol = symbol
        self.frame = tk.Frame(parent)
        self.x_var = tk.StringVar(value="0")
        self.y_var = tk.StringVar(value="0")
        self.z_var = tk.StringVar(value="0")

        tk.Label(self.frame, text="x:", width=2).pack(side="left", padx=(8, 0))
        tk.Entry(self.frame, textvariable=self.x_var, width=6).pack(side="left", padx=2)
        tk.Label(self.frame, text="y:", width=2).pack(side="left", padx=(4, 0))
        tk.Entry(self.frame, textvariable=self.y_var, width=6).pack(side="left", padx=2)
        tk.Label(self.frame, text="z:", width=2).pack(side="left", padx=(4, 0))
        tk.Entry(self.frame, textvariable=self.z_var, width=6).pack(side="left", padx=2)
        tk.Button(self.frame, text="\u2212", width=2, cursor="hand2",
                  command=on_remove).pack(side="left", padx=4)

        for v in (self.x_var, self.y_var, self.z_var):
            v.trace_add("write", lambda *_: on_coord_change())

    def get_xyz(self):
        def _f(v):
            try:
                return float(v.get().strip() or 0)
            except ValueError:
                return 0.0
        return _f(self.x_var), _f(self.y_var), _f(self.z_var)

    def pack(self, **kw):
        self.frame.pack(**kw)

    def destroy(self):
        self.frame.destroy()


# ─────────────────────────  Bloc d'élément  ──────────────────
class ElementBlock:
    """Un élément dans la barre latérale : symbole, bouton [+], lignes de noyaux."""
    def __init__(self, parent, z, symbol, on_add_nucleus):
        self.z = z
        self.symbol = symbol
        self.on_add_nucleus = on_add_nucleus
        self.nucleus_rows = []

        self.frame = tk.Frame(parent)
        row = tk.Frame(self.frame)
        row.pack(anchor="w")
        tk.Label(row, text=f"{z}", width=3, anchor="e").pack(side="left", padx=(0, 4))
        tk.Label(row, text=symbol, width=3, font=("Segoe UI", 10, "bold")).pack(side="left", padx=2)
        tk.Button(row, text="+", width=2, cursor="hand2",
                  command=self._add_nucleus).pack(side="left", padx=2)

        self.coords_frame = tk.Frame(self.frame)
        self.coords_frame.pack(anchor="w", fill="x")

    def _add_nucleus(self):
        def remove_this():
            if nr in self.nucleus_rows:
                self.nucleus_rows.remove(nr)
                nr.destroy()
            self.on_add_nucleus()

        nr = NucleusRow(self.coords_frame, self.symbol,
                        self.on_add_nucleus, remove_this)
        nr.pack(anchor="w", pady=1)
        self.nucleus_rows.append(nr)
        self.on_add_nucleus()

    # ── méthodes programmatiques ──
    def add_nucleus_with_coords(self, x, y, z):
        """Ajouter un noyau avec des coordonnées prédéfinies."""
        def remove_this():
            if nr in self.nucleus_rows:
                self.nucleus_rows.remove(nr)
                nr.destroy()
            self.on_add_nucleus()

        nr = NucleusRow(self.coords_frame, self.symbol,
                        self.on_add_nucleus, remove_this)
        nr.x_var.set(str(round(x, 4)))
        nr.y_var.set(str(round(y, 4)))
        nr.z_var.set(str(round(z, 4)))
        nr.pack(anchor="w", pady=1)
        self.nucleus_rows.append(nr)

    def clear_all_nuclei(self):
        """Supprimer toutes les lignes de noyaux."""
        for nr in list(self.nucleus_rows):
            nr.destroy()
        self.nucleus_rows.clear()

    def get_nuclei(self):
        return [(self.symbol, r.get_xyz()) for r in self.nucleus_rows]

    def pack(self, **kw):
        self.frame.pack(**kw)


# ═════════════════════════  Application  ══════════════════════
VIEW_NUCLEI      = -1
VIEW_DENSITY     = -2
VIEW_ESP         = -3

# Constantes d'interface en français
_NUCLEI_ONLY       = "Noyaux uniquement"
_COMBINED_DENSITY  = "Densit\u00e9 \u00e9lectronique combin\u00e9e"
_ESP_MAP           = "Carte du potentiel \u00e9lectrostatique"
_SELECT_MOL        = "-- S\u00e9lectionner --"


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simulateur quantique \u2014 SCF (1\u201382)")
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)

        self._batch_update = False          # empêche les mises à jour multiples

        self.main = tk.PanedWindow(self.root, orient="horizontal")
        self.main.pack(fill="both", expand=True, padx=4, pady=4)

        # Gauche : graphique 3-D
        self.plot_frame = tk.Frame(self.main, width=700)
        self.plot_frame.pack(side="left", fill="both", expand=True)
        self._setup_3d_plot()

        # Droite : barre latérale
        self.sidebar_frame = tk.Frame(self.main, width=280)
        self.sidebar_frame.pack(side="right", fill="y", padx=(8, 0))
        self.num_electrons_var = tk.StringVar(value="0")
        self._setup_sidebar()

        self.element_blocks = []
        self._build_periodic_sidebar()
        self.element_map = {b.symbol: b for b in self.element_blocks}

        # État Schrödinger
        self._clear_schrodinger()

        self._update_3d()
        self.root.mainloop()

    # ───────────────  graphique 3-D  ─────────────────────────
    def _setup_3d_plot(self):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x (Bohr)")
        self.ax.set_ylabel("y (Bohr)")
        self.ax.set_zlabel("z (Bohr)")
        self.ax.set_title("Positions des noyaux")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tk.Button(self.plot_frame, text="Synchroniser noyaux \u2192 potentiel",
                  command=self._sync_nuclei_to_potential).pack(pady=4)
        self._colorbar = None

    # ───────────────  barre latérale  ────────────────────────
    def _setup_sidebar(self):
        # ── Nombre d'électrons (champ de saisie) ──
        ef = tk.Frame(self.sidebar_frame)
        ef.pack(anchor="w", pady=(0, 8))
        tk.Label(ef, text="Nombre d'\u00e9lectrons",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        entry_frame = tk.Frame(ef)
        entry_frame.pack(anchor="w", pady=(2, 0))
        self.electron_entry = tk.Entry(entry_frame, textvariable=self.num_electrons_var,
                                       width=8, font=("Segoe UI", 10))
        self.electron_entry.pack(side="left", padx=(0, 4))

        # ── Bibliothèque de molécules ──
        mf = tk.LabelFrame(self.sidebar_frame,
                           text="Biblioth\u00e8que de mol\u00e9cules",
                           font=("Segoe UI", 10, "bold"))
        mf.pack(anchor="w", pady=(0, 8), fill="x")
        self.molecule_var = tk.StringVar(value=_SELECT_MOL)
        mol_names = [_SELECT_MOL] + [m["name"] for m in MOLECULE_LIBRARY]
        self.molecule_combo = ttk.Combobox(mf, textvariable=self.molecule_var,
                                           state="readonly", width=30)
        self.molecule_combo["values"] = mol_names
        self.molecule_combo.pack(anchor="w", pady=2, padx=4)
        self.molecule_combo.bind("<<ComboboxSelected>>", self._on_molecule_select)

        # ── Contrôles Schrödinger ──
        sf = tk.LabelFrame(self.sidebar_frame, text="Schr\u00f6dinger (SCF)",
                           font=("Segoe UI", 10, "bold"))
        sf.pack(anchor="w", pady=(0, 8), fill="x")
        tk.Button(sf, text="Calculer les niveaux d'\u00e9nergie",
                  command=self._compute_schrodinger).pack(anchor="w", pady=2)
        self.energy_listbox = tk.Listbox(sf, height=5, width=32,
                                         font=("Consolas", 8))
        self.energy_listbox.pack(anchor="w", pady=2)
        self.energy_listbox.bind("<<ListboxSelect>>", self._on_orbital_select)

        tk.Label(sf, text="Afficher :", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        self.orbital_view_var = tk.StringVar(value=_NUCLEI_ONLY)
        self.orbital_combo = ttk.Combobox(sf, textvariable=self.orbital_view_var,
                                          state="readonly", width=30)
        self.orbital_combo.pack(anchor="w", pady=2)
        self.orbital_combo.bind("<<ComboboxSelected>>", self._on_orbital_view_select)
        tk.Button(sf, text="Afficher la distribution de probabilit\u00e9",
                  command=self._show_probability_dots).pack(anchor="w", pady=2)

        tk.Label(self.sidebar_frame,
                 text="Tableau p\u00e9riodique (1\u201382) \u2014 positions en Bohr",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))
        self.scrollable = ScrollableSidebar(self.sidebar_frame, width=260)
        self.scrollable.pack(fill="both", expand=True)

    def _build_periodic_sidebar(self):
        inner = self.scrollable.inner
        for z, symbol in enumerate(ELEMENTS_1_82, start=1):
            block = ElementBlock(inner, z, symbol, self._on_nuclei_change)
            block.pack(anchor="w", pady=2)
            self.element_blocks.append(block)

    # ───────────────  utilitaires  ───────────────────────────
    def _get_num_electrons(self):
        """Retourne le nombre d'électrons saisi (int)."""
        try:
            return max(0, int(self.num_electrons_var.get().strip() or "0"))
        except ValueError:
            return 0

    def _collect_nuclei(self):
        out = []
        for block in self.element_blocks:
            out.extend(block.get_nuclei())
        return out

    def _clear_schrodinger(self):
        self.schrodinger_energies = []
        self.schrodinger_wavefunctions = []
        self.schrodinger_grid_info = {}
        self.schrodinger_occupancies = []
        self.schrodinger_density = None
        self.schrodinger_esp = None
        self.selected_orbital_var = tk.IntVar(value=VIEW_NUCLEI)
        self.energy_listbox.delete(0, tk.END)
        self.orbital_combo.set(_NUCLEI_ONLY)
        self.orbital_combo["values"] = [_NUCLEI_ONLY]

    def _on_nuclei_change(self):
        if self._batch_update:
            return
        self._clear_schrodinger()
        self._update_3d()

    def _sync_nuclei_to_potential(self):
        nuclei = self._collect_nuclei()
        data = [{"symbol": s, "x": x, "y": y, "z": z} for s, (x, y, z) in nuclei]
        with open(NUCLEI_STATE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        self._update_3d()

    # ───────────────  bibliothèque de molécules  ─────────────
    def _on_molecule_select(self, event):
        name = self.molecule_combo.get()
        if name == _SELECT_MOL:
            return
        for mol in MOLECULE_LIBRARY:
            if mol["name"] == name:
                self._load_molecule(mol)
                break

    def _load_molecule(self, mol):
        """Charger une molécule prédéfinie dans l'interface."""
        self._batch_update = True
        # supprimer tous les noyaux existants
        for block in self.element_blocks:
            block.clear_all_nuclei()
        # ajouter les atomes de la molécule
        for sym, (x, y, z) in mol["atoms"]:
            block = self.element_map.get(sym)
            if block:
                block.add_nucleus_with_coords(x, y, z)
        # définir le nombre d'électrons
        self.num_electrons_var.set(str(mol["electrons"]))
        self._batch_update = False
        self._clear_schrodinger()
        self._update_3d()

    # ───────────────  Schrödinger  ───────────────────────────
    def _compute_schrodinger(self):
        nuclei = self._collect_nuclei()
        if not nuclei:
            messagebox.showinfo("Aucun noyau",
                                "Ajoutez au moins un noyau (+) et d\u00e9finissez les coordonn\u00e9es.")
            return

        num_e = self._get_num_electrons()

        try:
            result = solve_schrodinger(nuclei, num_electrons=num_e,
                                       n_grid=16, num_states=7)
            energies, wfs, gi, occ, density, esp = result
        except Exception as e:
            messagebox.showerror("Solveur Schr\u00f6dinger", str(e))
            return

        if len(energies) == 0:
            messagebox.showwarning("Aucun \u00e9tat",
                                   "Le solveur n\u2019a retourn\u00e9 aucun \u00e9tat propre.")
            return

        self.schrodinger_energies = energies
        self.schrodinger_wavefunctions = wfs
        self.schrodinger_grid_info = gi
        self.schrodinger_occupancies = occ
        self.schrodinger_density = density
        self.schrodinger_esp = esp

        # liste des énergies avec occupation
        self.energy_listbox.delete(0, tk.END)
        for i, E in enumerate(energies):
            ne = occ[i] if i < len(occ) else 0
            tag = f" ({ne}e\u207b)" if num_e > 0 else ""
            self.energy_listbox.insert(tk.END, f"E{i} = {E:.5f} Ha{tag}")

        # options du menu déroulant
        opts = [_NUCLEI_ONLY]
        for i in range(len(energies)):
            ne = occ[i] if i < len(occ) else 0
            opts.append(f"Orbitale {i} (E={energies[i]:.4f}, {ne}e\u207b)")
        if density is not None and num_e > 0:
            opts.append(_COMBINED_DENSITY)
        if esp is not None and num_e > 0:
            opts.append(_ESP_MAP)
        self.orbital_combo["values"] = opts
        self.orbital_combo.set(_NUCLEI_ONLY)
        self.selected_orbital_var.set(VIEW_NUCLEI)
        self._update_3d()

    # ───────────────  sélection combo / listbox  ─────────────
    def _on_orbital_select(self, event):
        pass

    def _on_orbital_view_select(self, event):
        sel = self.orbital_combo.get()
        if sel == _NUCLEI_ONLY:
            self.selected_orbital_var.set(VIEW_NUCLEI)
        elif sel == _COMBINED_DENSITY:
            self.selected_orbital_var.set(VIEW_DENSITY)
        elif sel == _ESP_MAP:
            self.selected_orbital_var.set(VIEW_ESP)
        else:
            try:
                i = int(sel.split()[1])
                self.selected_orbital_var.set(i)
            except (IndexError, ValueError):
                self.selected_orbital_var.set(VIEW_NUCLEI)
        self._update_3d()

    def _show_probability_dots(self):
        if not self.schrodinger_wavefunctions:
            messagebox.showinfo(
                "Calculez d\u2019abord",
                "Cliquez d\u2019abord sur \u00ab Calculer les niveaux d\u2019\u00e9nergie \u00bb.")
            return
        cur = self.orbital_combo.get()
        if cur == _NUCLEI_ONLY:
            self.orbital_combo.set(self.orbital_combo["values"][1])
            self.selected_orbital_var.set(0)
        self._update_3d()

    # ───────────────  rendu 3-D  ─────────────────────────────
    def _update_3d(self):
        self.ax.clear()
        self.ax.set_xlabel("x (Bohr)")
        self.ax.set_ylabel("y (Bohr)")
        self.ax.set_zlabel("z (Bohr)")

        # supprimer l'ancienne barre de couleur
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        nuclei = self._collect_nuclei()
        show = self.selected_orbital_var.get()
        gi = self.schrodinger_grid_info
        wfs = self.schrodinger_wavefunctions

        # ── titre ──
        if show >= 0 and show < len(wfs):
            self.ax.set_title(f"Orbitale {show}  |\u03c8|\u00b2 probabilit\u00e9")
        elif show == VIEW_DENSITY:
            self.ax.set_title("Densit\u00e9 \u00e9lectronique combin\u00e9e")
        elif show == VIEW_ESP:
            self.ax.set_title("Carte du potentiel \u00e9lectrostatique")
        else:
            self.ax.set_title("Positions des noyaux")

        # ── état vide ──
        if not nuclei and show == VIEW_NUCLEI:
            self.ax.set_xlim(-1, 1); self.ax.set_ylim(-1, 1); self.ax.set_zlim(-1, 1)
            self.canvas.draw_idle()
            return

        xs = [n[1][0] for n in nuclei] if nuclei else []
        ys = [n[1][1] for n in nuclei] if nuclei else []
        zs = [n[1][2] for n in nuclei] if nuclei else []
        symbols = [n[0] for n in nuclei] if nuclei else []

        px = py = pz = np.array([])

        # ── orbitale individuelle ──
        if show >= 0 and show < len(wfs) and gi:
            np.random.seed(42)
            prob = np.abs(wfs[show]) ** 2
            px, py, pz = sample_positions_from_probability(
                prob, gi, num_dots=3000)
            self.ax.scatter(px, py, pz, c="darkgreen", s=4, alpha=0.5,
                            label="|\u03c8|\u00b2")

        # ── densité combinée ──
        elif show == VIEW_DENSITY and self.schrodinger_density is not None and gi:
            np.random.seed(42)
            px, py, pz = sample_positions_from_probability(
                self.schrodinger_density, gi, num_dots=4000)
            self.ax.scatter(px, py, pz, c="purple", s=4, alpha=0.45,
                            label="\u03c1(r)")

        # ── carte du potentiel électrostatique ──
        elif show == VIEW_ESP and self.schrodinger_esp is not None and gi:
            np.random.seed(42)
            px, py, pz = sample_positions_from_probability(
                self.schrodinger_density, gi, num_dots=4000)
            phi = get_esp_at_points(px, py, pz,
                                    self.schrodinger_esp, gi)
            # découper au 5e–95e percentile pour une gamme de couleur lisible
            lo, hi = np.percentile(phi, [5, 95])
            if lo >= hi:
                lo, hi = phi.min(), phi.max()
            if lo >= hi:
                lo, hi = -1.0, 1.0
            norm = mcolors.TwoSlopeNorm(vcenter=0.0,
                                        vmin=min(lo, -0.01),
                                        vmax=max(hi, 0.01))
            cmap = mcm.RdBu_r   # rouge = négatif (riche en e⁻)
            self.ax.scatter(px, py, pz, c=cmap(norm(phi)),
                            s=5, alpha=0.6)
            sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            self._colorbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.55,
                                               pad=0.08, label="PES (Ha/e)")

        # ── sphères de noyaux ──
        if nuclei:
            self.ax.scatter(xs, ys, zs, c="steelblue", s=120, alpha=0.95,
                            edgecolors="navy", linewidths=0.8)
            for sym, (x, y, z) in zip(symbols, zip(xs, ys, zs)):
                self.ax.text(
                    x, y, z, f" {sym} ", fontsize=7,
                    ha="center", va="center", color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="steelblue", edgecolor="navy", alpha=0.9),
                )

        # ── limites des axes ──
        all_x = list(xs) + list(px)
        all_y = list(ys) + list(py)
        all_z = list(zs) + list(pz)
        if all_x:
            ext = max(max(all_x) - min(all_x),
                      max(all_y) - min(all_y),
                      max(all_z) - min(all_z), 1.0)
            margin = max(1.0, ext * 0.15)
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            self.ax.set_zlim(min(all_z) - margin, max(all_z) + margin)

        self.canvas.draw_idle()


# ═══════════════════════  Point d'entrée  ═════════════════════
def main():
    App()

if __name__ == "__main__":
    main()
