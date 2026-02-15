"""
PubChem Chatbot — 3D molecular geometry extraction + Schrödinger cloud.

Enter a molecule name. The chatbot queries PubChem PUG-REST, displays
3D geometry, then lets you run the Schrödinger solver for electron density
visualization (potential V(r) and probability heatmaps).

Toolbar: Potential V(r) slices, Schrödinger Cloud (50x50x50), zoom with scroll.

Usage: python pubchem_chatbot.py
"""

import math
import re
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import requests

from potential_3d import (
    SYMBOL_TO_Z,      # symbol → Z mapping (1-118, single source of truth)
    Z_TO_SYMBOL,      # Z → symbol reverse lookup
    save_nuclei,      # write nuclei to shared nuclei_state.json
    potential_3d as build_potential_func,   # V(x,y,z) callable builder
    potential_on_grid,                      # evaluate V on a 3D meshgrid
)
from schrodinger import (
    compute_schrodinger_cloud,
    open_schrodinger_cloud_window,
)

# ─────────────────────────  Constantes  ─────────────────────
ANGSTROM_TO_BOHR = 1.8897259886          # 1 Å = 1.889… a₀
PUBCHEM_3D_URL   = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                     "compound/name/{name}/JSON?record_type=3d")

# CPK colors per element
_ELEMENT_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0", "Br": "#A62929",
    "I": "#940094",  "Au": "#FFD123", "Pt": "#D0D0E0",
}
_DEFAULT_COLOR = "#FF69B4"


# ─────────────────────────  PubChem  ────────────────────────

def fetch_pubchem_3d(molecule_name: str) -> dict:
    """
    Query PubChem for 3D geometry of a molecule.

    Returns a dict:
        cid, name, atoms [(sym,(x,y,z))], elements [sym], bonds [(i,j,order)],
        total_electrons (neutral molecule)
    """
    url = PUBCHEM_3D_URL.format(name=requests.utils.quote(molecule_name))
    resp = requests.get(url, timeout=15)

    if resp.status_code == 404:
        raise ValueError(
            f"Molecule \"{molecule_name}\" not found on PubChem.\n"
            "Try another name (English recommended) or a chemical formula.")
    if resp.status_code != 200:
        raise ConnectionError(
            f"PubChem responded with code {resp.status_code}.\n"
            "Check your internet connection and try again.")

    data = resp.json()
    compounds = data.get("PC_Compounds", [])
    if not compounds:
        raise ValueError(f"No compound returned for \"{molecule_name}\".")

    comp = compounds[0]
    cid = comp.get("id", {}).get("id", {}).get("cid", 0)

    atom_section = comp.get("atoms", {})
    elements_z   = atom_section.get("element", [])
    aids         = atom_section.get("aid", [])

    coords_section = comp.get("coords", [{}])[0]
    conformers     = coords_section.get("conformers", [])
    if not conformers:
        raise ValueError(
            f"No 3D conformer for \"{molecule_name}\" (CID {cid}).")
    conf = conformers[0]
    xs_ang = conf.get("x", [])
    ys_ang = conf.get("y", [])
    zs_ang = conf.get("z", [])

    bond_section = comp.get("bonds", {})
    aid1_list  = bond_section.get("aid1", [])
    aid2_list  = bond_section.get("aid2", [])
    order_list = bond_section.get("order", [])
    aid_to_idx = {a: i for i, a in enumerate(aids)}
    bonds = []
    for a1, a2, order in zip(aid1_list, aid2_list, order_list):
        i1, i2 = aid_to_idx.get(a1), aid_to_idx.get(a2)
        if i1 is not None and i2 is not None:
            bonds.append((i1, i2, order))

    atoms = []
    symbols = []
    total_e = 0
    for z_num, xA, yA, zA in zip(elements_z, xs_ang, ys_ang, zs_ang):
        sym = Z_TO_SYMBOL.get(z_num, "?")  # use shared mapping from potential_3d
        atoms.append((sym, (xA * ANGSTROM_TO_BOHR,
                            yA * ANGSTROM_TO_BOHR,
                            zA * ANGSTROM_TO_BOHR)))
        symbols.append(sym)
        total_e += z_num  # neutral molecule: electrons = sum of Z

    return {
        "cid": cid, "name": molecule_name,
        "atoms": atoms, "elements": symbols, "bonds": bonds,
        "total_electrons": total_e,
    }


# ─────────────────────  Analyse de commande  ────────────────

_TRIGGER_PATTERNS = [
    r"(?:cherche|montre|affiche|charge|trouve|donne|récupère|ouvre)\s+(?:la molécule\s+|la géométrie (?:de |d')?)?(.+)",
    r"(?:géométrie|structure|coordonnées|atomes)\s+(?:de |d'|du |des )?(.+)",
    r"(?:show|display|load|find|get|fetch|search|open|plot|draw|visualize|give)\s+(?:me\s+)?(?:the\s+)?(?:molecule\s+|geometry (?:of |for )?)?(.+)",
    r"(?:geometry|structure|coordinates|atoms)\s+(?:of |for )?(.+)",
    r"(?:what does|how does|what is|qu'est[- ]ce que)\s+(.+?)(?:\s+look like)?$",
]
_STRIP_WORDS = {"please", "s'il te plaît", "stp", "svp", "molecule",
                "molécule", "the", "la", "le", "un", "une", "of", "de", "d'",
                "du", "for", "?", "!", "."}


def _extract_molecule_name(user_input: str) -> str:
    text = user_input.strip()
    if not text:
        return ""
    for pat in _TRIGGER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            tokens = [t for t in name.split() if t.lower() not in _STRIP_WORDS]
            if tokens:
                return " ".join(tokens)
    return text


# ─────────────────────────  Toolbar labels  ─────────────────


# ─────────────────────────  Interface  ──────────────────────

class PubChemChatbot:
    """Tkinter window: toolbar + 3D plot + chat."""

    # Catppuccin Mocha
    _BG        = "#1e1e2e"
    _BG_CHAT   = "#181825"
    _FG        = "#cdd6f4"
    _ACCENT    = "#89b4fa"
    _TB_BG     = "#313244"   # toolbar background
    _ENTRY_BG  = "#313244"
    _ENTRY_FG  = "#cdd6f4"
    _BTN_BG    = "#45475a"
    _BTN_FG    = "#cdd6f4"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PubChem Chatbot — Schrödinger 3D")
        self.root.geometry("1280x760")
        self.root.minsize(960, 580)
        self.root.configure(bg=self._BG)

        self._current_mol = None       # last PubChem result
        self._display_bohr = True
        self._zoom = 10.0              # matplotlib 3D "dist" for zoom
        self._colorbar = None

        # ── layout : top toolbar | left plot | right chat ──
        self._setup_toolbar()

        body = tk.PanedWindow(self.root, orient="horizontal",
                              bg=self._BG, sashwidth=4)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.plot_frame = tk.Frame(body, bg=self._BG)
        body.add(self.plot_frame, width=660)
        self._setup_plot()

        chat_frame = tk.Frame(body, bg=self._BG)
        body.add(chat_frame, width=560)
        self._setup_chat(chat_frame)

        # Welcome message
        self._bot_say(
            "Welcome! I'm the PubChem + Schrödinger chatbot.\n"
            "Type a molecule name and I'll display its 3D geometry.\n\n"
            "Examples:\n"
            "  • water  •  methane  •  aspirin  •  caffeine\n"
            "  • show me benzene  •  ethanol\n\n"
            "Toolbar: Potential V(r) slices, Schrödinger Cloud (electron density).\n"
            "Zoom with mouse wheel.\n\n"
            "Commands: bohr / angstrom  •  info  •  clear  •  help"
        )

        self.root.mainloop()

    # ═══════════════════  TOOLBAR  ═══════════════════════════
    def _setup_toolbar(self):
        tb = tk.Frame(self.root, bg=self._TB_BG, height=42)
        tb.pack(fill="x", padx=6, pady=(6, 0))

        # ── Potential V(r) — slice viewer ──
        self._potential_btn = tk.Button(
            tb, text="Potential V(r)", bg="#f9e2af", fg="#1e1e2e",
            activebackground="#f5c2e7", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self._on_show_potential)
        self._potential_btn.pack(side="left", padx=(10, 8), ipady=2, ipadx=4)

        # ── Schrödinger Cloud — electron density from schrodinger.py ──
        self._cloud_schro_btn = tk.Button(
            tb, text="Schrödinger Cloud", bg="#cba6f7", fg="#1e1e2e",
            activebackground="#f5c2e7", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self._on_show_schrodinger_cloud)
        self._cloud_schro_btn.pack(side="left", padx=(0, 8), ipady=2, ipadx=4)

        # ── Energy Levels — eigenvalues diagram (Pauli: orbitals = ceil(electrons/2)) ──
        self._energy_levels_btn = tk.Button(
            tb, text="Energy Levels", bg="#a6e3a1", fg="#1e1e2e",
            activebackground="#f5c2e7", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self._on_show_energy_levels)
        self._energy_levels_btn.pack(side="left", padx=(0, 8), ipady=2, ipadx=4)

        # ── separator ──
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                  padx=4, pady=6)

        # ── Zoom label ──
        tk.Label(tb, text="Zoom: mouse wheel",
                 bg=self._TB_BG, fg="#6c7086",
                 font=("Segoe UI", 8, "italic")).pack(side="left", padx=8)

    # ═══════════════════  3D PLOT  ══════════════════════════
    def _setup_plot(self):
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=self._BG)
        self.ax = self.fig.add_subplot(111, projection="3d",
                                       facecolor=self._BG_CHAT)
        self._style_axes()
        self.ax.set_title("No molecule loaded",
                          color=self._FG, fontsize=11, pad=12)
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_mpl.get_tk_widget().pack(fill="both", expand=True)

        # ── scroll-to-zoom ──
        widget = self.canvas_mpl.get_tk_widget()
        widget.bind("<MouseWheel>", self._on_scroll_zoom)          # Windows
        widget.bind("<Button-4>",   self._on_scroll_zoom_linux)    # Linux up
        widget.bind("<Button-5>",   self._on_scroll_zoom_linux)    # Linux down

    def _style_axes(self):
        self.ax.set_xlabel("x (Bohr)", color=self._FG, fontsize=9)
        self.ax.set_ylabel("y (Bohr)", color=self._FG, fontsize=9)
        self.ax.set_zlabel("z (Bohr)", color=self._FG, fontsize=9)
        self.ax.tick_params(colors=self._FG, labelsize=7)

    def _on_scroll_zoom(self, event):
        """Windows scroll zoom: delta > 0 = scroll up = zoom in."""
        if event.delta > 0:
            self._zoom = max(2.0, self._zoom - 0.6)
        else:
            self._zoom = min(30.0, self._zoom + 0.6)
        self.ax.dist = self._zoom
        self.canvas_mpl.draw_idle()

    def _on_scroll_zoom_linux(self, event):
        if event.num == 4:
            self._zoom = max(2.0, self._zoom - 0.6)
        else:
            self._zoom = min(30.0, self._zoom + 0.6)
        self.ax.dist = self._zoom
        self.canvas_mpl.draw_idle()

    # ═══════════════════  CHAT  ═════════════════════════════
    def _setup_chat(self, parent):
        header = tk.Label(parent, text="\U0001f4ac  PubChem Chat",
                          font=("Segoe UI", 13, "bold"),
                          bg=self._BG, fg=self._ACCENT, anchor="w")
        header.pack(fill="x", pady=(0, 4))

        self.chat_text = tk.Text(
            parent, wrap="word", state="disabled",
            bg=self._BG_CHAT, fg=self._FG, font=("Consolas", 10),
            relief="flat", padx=10, pady=8,
            insertbackground=self._FG, selectbackground=self._ACCENT)
        self.chat_text.pack(fill="both", expand=True, pady=(0, 4))

        self.chat_text.tag_configure("user", foreground="#a6e3a1",
                                     font=("Consolas", 10, "bold"))
        self.chat_text.tag_configure("bot", foreground=self._FG,
                                     font=("Consolas", 10))
        self.chat_text.tag_configure("error", foreground="#f38ba8",
                                     font=("Consolas", 10, "italic"))
        self.chat_text.tag_configure("info", foreground="#89dceb",
                                     font=("Consolas", 10))

        entry_frame = tk.Frame(parent, bg=self._BG)
        entry_frame.pack(fill="x")

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            entry_frame, textvariable=self.entry_var,
            bg=self._ENTRY_BG, fg=self._ENTRY_FG,
            insertbackground=self._ENTRY_FG,
            font=("Consolas", 11), relief="flat", bd=0)
        self.entry.pack(side="left", fill="x", expand=True,
                        ipady=6, padx=(0, 4))
        self.entry.bind("<Return>", self._on_enter)

        tk.Button(entry_frame, text="Send",
                  bg=self._ACCENT, fg="#1e1e2e",
                  activebackground="#74c7ec",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", cursor="hand2",
                  command=self._on_enter).pack(side="right", ipady=4, ipadx=8)
        self.entry.focus_set()

    # ─────────────  Messages  ─────────────────────────────────
    def _append_text(self, text: str, tag: str = "bot"):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text + "\n\n", tag)
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _user_say(self, t):   self._append_text(f"You:  {t}", "user")
    def _bot_say(self, t):    self._append_text(f"Bot:  {t}", "bot")
    def _bot_error(self, t):  self._append_text(f"Bot :  \u26a0 {t}", "error")
    def _bot_info(self, t):   self._append_text(f"Bot :  \u2139 {t}", "info")

    # ─────────────  Chat input  ───────────────────────────────
    def _on_enter(self, event=None):
        text = self.entry_var.get().strip()
        if not text:
            return
        self.entry_var.set("")
        self._user_say(text)
        self._process_command(text)

    def _process_command(self, text: str):
        low = text.lower().strip()

        if low in ("aide", "help", "?"):
            self._bot_say(
                "Type a molecule name to display its geometry.\n"
                "  • water, methane, aspirin, caffeine …\n"
                "  • show me benzene / ethanol\n\n"
                "Toolbar: Potential V(r), Schrödinger Cloud (electron density).\n"
                "Zoom with mouse wheel.\n\n"
                "Commands: bohr / angstrom  •  info  •  clear  •  help")
            return

        if low in ("bohr", "bohrs"):
            self._display_bohr = True
            self._bot_info("Unit → Bohr (a\u2080).")
            if self._current_mol:
                self._redraw()
            return

        if low in ("angstrom", "angstroms", "\u00e5", "ang"):
            self._display_bohr = False
            self._bot_info("Unit → \u00c5ngstr\u00f6m (\u00c5).")
            if self._current_mol:
                self._redraw()
            return

        if low in ("info", "infos"):
            if self._current_mol:
                self._show_molecule_info(self._current_mol)
            else:
                self._bot_info("No molecule loaded.")
            return

        if low in ("clear", "effacer", "reset"):
            self._current_mol = None
            save_nuclei([])  # clear shared state file
            self.ax.clear()
            self._style_axes()
            self.ax.set_title("No molecule loaded",
                              color=self._FG, fontsize=11, pad=12)
            self.canvas_mpl.draw_idle()
            self._bot_info("Display cleared.")
            return

        # Molecule search
        name = _extract_molecule_name(text)
        if not name:
            self._bot_error("I didn't understand. Type \"help\".")
            return
        self._bot_info(f"Searching for \"{name}\" on PubChem...")
        threading.Thread(target=self._fetch_and_display,
                         args=(name,), daemon=True).start()

    # ─────────────  PubChem fetch  ────────────────────────────
    def _fetch_and_display(self, name: str):
        try:
            mol = fetch_pubchem_3d(name)
        except (ValueError, ConnectionError) as e:
            self.root.after(0, self._bot_error, str(e))
            return
        except Exception as e:
            self.root.after(0, self._bot_error, f"Error: {e}")
            return
        self.root.after(0, self._on_molecule_received, mol)

    def _on_molecule_received(self, mol: dict):
        self._current_mol = mol

        # Sync PubChem geometry → shared nuclei_state.json so that
        # potential_3d, schrodinger_3d, and render_energy_3d all see it.
        save_nuclei(mol["atoms"])

        self._show_molecule_info(mol)
        self._redraw()

    def _show_molecule_info(self, mol: dict):
        atoms = mol["atoms"]
        counts = {}
        for sym, _ in atoms:
            counts[sym] = counts.get(sym, 0) + 1
        formula = "".join(f"{s}{c if c > 1 else ''}"
                          for s, c in sorted(counts.items()))
        unit = "Bohr" if self._display_bohr else "\u00c5"
        factor = 1.0 if self._display_bohr else (1.0 / ANGSTROM_TO_BOHR)
        lines = [
            f"Molecule: {mol['name']}  (CID {mol['cid']})",
            f"Formula : {formula}",
            f"Atoms   : {len(atoms)}     Bonds: {len(mol['bonds'])}",
            f"Electrons (neutral): {mol['total_electrons']}",
            "", f"Coordinates ({unit}):"]
        for i, (sym, (x, y, z)) in enumerate(atoms):
            lines.append(f"  {i+1:3d}  {sym:2s}  "
                         f"({x*factor:8.4f}, {y*factor:8.4f}, {z*factor:8.4f})")
        self._bot_say("\n".join(lines))

    def _on_show_schrodinger_cloud(self):
        """Run schrodinger.py pipeline and open cloud visualization (V, |psi_i|^2, combined)."""
        if not self._current_mol:
            self._bot_error("Load a molecule first.")
            return

        nuclei = self._current_mol["atoms"]
        mol_name = self._current_mol["name"]
        save_nuclei(nuclei)  # sync PubChem geometry to nuclei_state.json

        self._cloud_schro_btn.config(state="disabled", text="Computing…")
        self._bot_info(
            f"Running Schrödinger solver (50x50x50) for {mol_name}…\n"
            "This may take 1–2 minutes.")

        def run():
            try:
                data = compute_schrodinger_cloud(nuclei)
                self.root.after(0, self._on_schrodinger_cloud_done, data, mol_name)
            except Exception as e:
                self.root.after(0, self._on_schrodinger_cloud_error, str(e))

        threading.Thread(target=run, daemon=True).start()

    def _on_schrodinger_cloud_done(self, data, mol_name):
        self._cloud_schro_btn.config(state="normal", text="Schrödinger Cloud")
        if data is None:
            self._bot_error("Solver returned nothing.")
            return
        open_schrodinger_cloud_window(data, parent=self.root, mol_name=mol_name)
        self._bot_info(f"Schrödinger cloud window opened for {mol_name}.")

    def _on_schrodinger_cloud_error(self, msg):
        self._cloud_schro_btn.config(state="normal", text="Schrödinger Cloud")
        self._bot_error(f"Schrödinger error: {msg}")

    def _on_show_energy_levels(self):
        """Compute eigenvalues (Pauli: num_orbitals = ceil(electrons/2)) and show energy diagram."""
        if not self._current_mol:
            self._bot_error("Load a molecule first.")
            return

        nuclei = self._current_mol["atoms"]
        mol_name = self._current_mol["name"]
        total_electrons = self._current_mol["total_electrons"]
        save_nuclei(nuclei)

        # Pauli exclusion: 2 electrons per orbital → orbitals = ceil(electrons/2)
        num_orbitals = max(1, math.ceil(total_electrons / 2))

        self._energy_levels_btn.config(state="disabled", text="Computing…")
        self._bot_info(
            f"Computing {num_orbitals} eigenvalues for {mol_name} "
            f"({total_electrons} electrons, {num_orbitals} orbitals)…\n"
            "This may take 1–2 minutes.")

        def run():
            try:
                data = compute_schrodinger_cloud(nuclei, num_states=num_orbitals)
                self.root.after(0, self._on_energy_levels_done,
                                data, mol_name, total_electrons)
            except Exception as e:
                self.root.after(0, self._on_energy_levels_error, str(e))

        threading.Thread(target=run, daemon=True).start()

    def _on_energy_levels_done(self, data, mol_name, total_electrons):
        self._energy_levels_btn.config(state="normal", text="Energy Levels")
        if data is None:
            self._bot_error("Solver returned nothing.")
            return
        self._open_energy_levels_window(
            data["eigenvalues"], mol_name, total_electrons)
        self._bot_info(f"Energy levels window opened for {mol_name}.")

    def _on_energy_levels_error(self, msg):
        self._energy_levels_btn.config(state="normal", text="Energy Levels")
        self._bot_error(f"Energy levels error: {msg}")

    def _open_energy_levels_window(self, eigenvalues, mol_name, total_electrons):
        """Open a new Toplevel with energy level diagram (eigenvalues as horizontal bars).
        Displays exactly ceil(electrons/2) orbitals — all filled (Pauli exclusion).
        """
        win = tk.Toplevel(self.root)
        win.title(f"Energy Levels — {mol_name}")
        win.geometry("680x520")
        win.configure(bg=self._BG)

        n_orb = len(eigenvalues)
        indices = np.arange(n_orb)

        fig = Figure(figsize=(7, 5), dpi=100, facecolor=self._BG)
        ax = fig.add_subplot(111, facecolor="#181825")
        ax.barh(indices, eigenvalues, color="#a6e3a1", edgecolor="#45475a",
                linewidth=0.8, height=0.6)
        ax.set_xlabel("Energy (Hartree)", color=self._FG, fontsize=10)
        ax.set_ylabel("Orbital index", color=self._FG, fontsize=10)
        ax.set_title(
            f"{mol_name} — {total_electrons} electrons, "
            f"{n_orb} orbital{'s' if n_orb != 1 else ''} (Pauli: 2e⁻/orbital)",
            color=self._FG, fontsize=11)
        ax.set_yticks(indices)
        ax.tick_params(colors=self._FG, labelsize=9)
        ax.set_facecolor("#181825")
        for spine in ax.spines.values():
            spine.set_color(self._FG)
            spine.set_alpha(0.5)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    # ═══════════════════  POTENTIAL SLICE VIEWER  ═══════════
    def _on_show_potential(self):
        """Open a new window with 2-D slice heatmaps of V(r) for the
        current PubChem molecule (uses potential_on_grid directly)."""
        if not self._current_mol:
            self._bot_error("Load a molecule first.")
            return

        nuclei = self._current_mol["atoms"]
        name = self._current_mol["name"]

        self._bot_info(f"Computing potential V(r) for {name}…")

        # ── grid ──
        xs = [p[1][0] for p in nuclei]
        ys = [p[1][1] for p in nuclei]
        zs = [p[1][2] for p in nuclei]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys),
                   max(zs) - min(zs), 0.0)
        half = span / 2.0 + 6.0
        n_pts = 120
        x_1d = np.linspace(cx - half, cx + half, n_pts)
        y_1d = np.linspace(cy - half, cy + half, n_pts)
        z_1d = np.linspace(cz - half, cz + half, n_pts)

        X, Y, Z, V_grid = potential_on_grid(nuclei, x_1d, y_1d, z_1d)

        # color range (skip singularity / tail)
        flat = V_grid.ravel()
        v_lo = float(np.percentile(flat, 1))
        v_hi = float(min(np.percentile(flat, 99), -0.01))
        if v_lo >= v_hi:
            v_lo, v_hi = float(flat.min()), -0.01

        n_contours = 30
        cmap_name = "RdBu_r"

        # ── new Toplevel window ──
        win = tk.Toplevel(self.root)
        win.title(f"Potential V(r) — {name}")
        win.geometry("1300x720")
        win.configure(bg=self._BG)

        import matplotlib.pyplot as plt
        from matplotlib import cm as mcm_local

        fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(
            1, 3, figsize=(14, 4.3), dpi=100, facecolor=self._BG)
        fig.subplots_adjust(left=0.05, right=0.92, bottom=0.12, top=0.88,
                            wspace=0.35)

        canvas_v = FigureCanvasTkAgg(fig, master=win)
        canvas_v.get_tk_widget().pack(fill="both", expand=True)

        # mutable slice positions
        sz_val = [cz]
        sy_val = [cy]
        sx_val = [cx]

        def nearest(arr, val):
            return int(np.argmin(np.abs(arr - val)))

        def draw():
            sz, sy, sx = sz_val[0], sy_val[0], sx_val[0]
            iz = nearest(z_1d, sz)
            iy = nearest(y_1d, sy)
            ix = nearest(x_1d, sx)

            for a in (ax_xy, ax_xz, ax_yz):
                a.clear()

            # xy slice (z=sz)
            Vxy = V_grid[:, :, iz]
            ax_xy.pcolormesh(x_1d, y_1d, Vxy.T, cmap=cmap_name,
                             vmin=v_lo, vmax=v_hi, shading="auto")
            ax_xy.contour(x_1d, y_1d, Vxy.T, levels=n_contours,
                          colors="k", linewidths=0.3, alpha=0.5)
            ax_xy.set_xlabel("x (Bohr)"); ax_xy.set_ylabel("y (Bohr)")
            ax_xy.set_title(f"xy  (z={sz:.2f})", color=self._FG, fontsize=10)
            ax_xy.set_aspect("equal"); ax_xy.set_facecolor(self._BG_CHAT)
            ax_xy.tick_params(colors=self._FG, labelsize=7)

            # xz slice (y=sy)
            Vxz = V_grid[:, iy, :]
            ax_xz.pcolormesh(x_1d, z_1d, Vxz.T, cmap=cmap_name,
                             vmin=v_lo, vmax=v_hi, shading="auto")
            ax_xz.contour(x_1d, z_1d, Vxz.T, levels=n_contours,
                          colors="k", linewidths=0.3, alpha=0.5)
            ax_xz.set_xlabel("x (Bohr)"); ax_xz.set_ylabel("z (Bohr)")
            ax_xz.set_title(f"xz  (y={sy:.2f})", color=self._FG, fontsize=10)
            ax_xz.set_aspect("equal"); ax_xz.set_facecolor(self._BG_CHAT)
            ax_xz.tick_params(colors=self._FG, labelsize=7)

            # yz slice (x=sx)
            Vyz = V_grid[ix, :, :]
            ax_yz.pcolormesh(y_1d, z_1d, Vyz.T, cmap=cmap_name,
                             vmin=v_lo, vmax=v_hi, shading="auto")
            ax_yz.contour(y_1d, z_1d, Vyz.T, levels=n_contours,
                          colors="k", linewidths=0.3, alpha=0.5)
            ax_yz.set_xlabel("y (Bohr)"); ax_yz.set_ylabel("z (Bohr)")
            ax_yz.set_title(f"yz  (x={sx:.2f})", color=self._FG, fontsize=10)
            ax_yz.set_aspect("equal"); ax_yz.set_facecolor(self._BG_CHAT)
            ax_yz.tick_params(colors=self._FG, labelsize=7)

            # nuclei markers
            tol = (x_1d[-1] - x_1d[0]) / n_pts * 2.5
            for sym, (nx, ny, nz) in nuclei:
                if abs(nz - sz) < tol:
                    ax_xy.plot(nx, ny, "o", color="red", ms=6,
                               mec="darkred", mew=0.8)
                    ax_xy.text(nx, ny, f" {sym}", fontsize=7,
                               color="white", fontweight="bold")
                if abs(ny - sy) < tol:
                    ax_xz.plot(nx, nz, "o", color="red", ms=6,
                               mec="darkred", mew=0.8)
                    ax_xz.text(nx, nz, f" {sym}", fontsize=7,
                               color="white", fontweight="bold")
                if abs(nx - sx) < tol:
                    ax_yz.plot(ny, nz, "o", color="red", ms=6,
                               mec="darkred", mew=0.8)
                    ax_yz.text(ny, nz, f" {sym}", fontsize=7,
                               color="white", fontweight="bold")

            canvas_v.draw_idle()

        # colorbar
        sm = mcm_local.ScalarMappable(
            cmap=cmap_name, norm=plt.Normalize(vmin=v_lo, vmax=v_hi))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_xy, ax_xz, ax_yz],
                            shrink=0.85, pad=0.03, aspect=30)
        cbar.set_label("V (Ha)", color=self._FG, fontsize=9)
        cbar.ax.tick_params(colors=self._FG, labelsize=7)

        # sliders
        ctrl = tk.Frame(win, bg=self._TB_BG)
        ctrl.pack(fill="x", padx=6, pady=(0, 6))

        def make_slider(parent, label, lo, hi, init, cb):
            f = tk.Frame(parent, bg=self._TB_BG)
            f.pack(side="left", padx=12, pady=4)
            tk.Label(f, text=label, bg=self._TB_BG, fg=self._FG,
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Scale(f, from_=lo, to=hi, resolution=0.05,
                     orient="horizontal", length=200,
                     command=cb,
                     bg=self._TB_BG, fg=self._FG,
                     troughcolor="#45475a",
                     highlightthickness=0).set(init)
            return f

        make_slider(ctrl, "z (xy) :",
                    float(z_1d[0]), float(z_1d[-1]), cz,
                    lambda v: (sz_val.__setitem__(0, float(v)), draw()))
        make_slider(ctrl, "y (xz) :",
                    float(y_1d[0]), float(y_1d[-1]), cy,
                    lambda v: (sy_val.__setitem__(0, float(v)), draw()))
        make_slider(ctrl, "x (yz) :",
                    float(x_1d[0]), float(x_1d[-1]), cx,
                    lambda v: (sx_val.__setitem__(0, float(v)), draw()))

        draw()
        self._bot_info(
            f"Potential window opened for {name}.\n"
            "Move the sliders to slice at different positions.")

    # ═══════════════════  3D RENDERING  ═════════════════════
    def _redraw(self):
        """Main drawing: nuclei + bonds. Use Schrödinger Cloud for electron density."""
        self.ax.clear()
        self._style_axes()

        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        mol = self._current_mol
        if not mol:
            self.ax.set_title("No molecule loaded",
                              color=self._FG, fontsize=11, pad=12)
            self.canvas_mpl.draw_idle()
            return

        atoms = mol["atoms"]
        bonds = mol["bonds"]
        unit = "Bohr" if self._display_bohr else "\u00c5"
        factor = 1.0 if self._display_bohr else (1.0 / ANGSTROM_TO_BOHR)
        name = mol["name"]

        self.ax.set_title(f"{name}  (CID {mol['cid']})",
                          color=self._FG, fontsize=11, pad=12)

        xs = np.array([x * factor for _, (x, y, z) in atoms])
        ys = np.array([y * factor for _, (x, y, z) in atoms])
        zs = np.array([z * factor for _, (x, y, z) in atoms])
        labels = [s for s, _ in atoms]
        colors = [_ELEMENT_COLORS.get(s, _DEFAULT_COLOR) for s in labels]

        for i1, i2, order in bonds:
            lw = 1.0 + 0.8 * (order - 1)
            self.ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]],
                         [zs[i1], zs[i2]],
                         color="#585b70", linewidth=lw, alpha=0.6)

        for i, (sym, c) in enumerate(zip(labels, colors)):
            self.ax.scatter(xs[i], ys[i], zs[i], c=c, s=160, alpha=0.95,
                            edgecolors="#45475a", linewidths=0.8, zorder=5)
            self.ax.text(xs[i], ys[i], zs[i], f"  {sym}", fontsize=8,
                         color=self._FG, fontweight="bold", zorder=6)

        if len(xs):
            cx = (xs.min() + xs.max()) / 2
            cy = (ys.min() + ys.max()) / 2
            cz = (zs.min() + zs.max()) / 2
            ext = max(xs.max() - xs.min(),
                      ys.max() - ys.min(),
                      zs.max() - zs.min(), 0.5)
            half = ext / 2 + 1.0
            self.ax.set_xlim(cx - half, cx + half)
            self.ax.set_ylim(cy - half, cy + half)
            self.ax.set_zlim(cz - half, cz + half)

        self.ax.set_xlabel(f"x ({unit})", color=self._FG, fontsize=9)
        self.ax.set_ylabel(f"y ({unit})", color=self._FG, fontsize=9)
        self.ax.set_zlabel(f"z ({unit})", color=self._FG, fontsize=9)
        self.ax.dist = self._zoom
        self.canvas_mpl.draw_idle()


# ─────────────────────────  Entry point  ─────────────────
def main():
    PubChemChatbot()

if __name__ == "__main__":
    main()
