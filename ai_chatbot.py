"""
Chatbot PubChem — Extraction de géométries moléculaires 3D + Schrödinger SCF.

L'utilisateur entre une commande contenant le nom d'une molécule.
Le chatbot interroge PubChem PUG-REST, affiche la géométrie 3-D,
puis permet de lancer le solveur Schrödinger pour calculer les niveaux
d'énergie et afficher le nuage électronique (probabilité) par points.

Barre d'outils en haut : électrons, calcul SCF, sélection du niveau,
mode d'affichage.  Zoom à la molette.

Utilisation :
    python pubchem_chatbot.py
"""

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
from schrodinger_3d import (
    solve_schrodinger,
    sample_positions_from_probability,
    get_esp_at_points,
)

# ─────────────────────────  Constantes  ─────────────────────
ANGSTROM_TO_BOHR = 1.8897259886          # 1 Å = 1.889… a₀
PUBCHEM_3D_URL   = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                     "compound/name/{name}/JSON?record_type=3d")

# Couleurs CPK par élément
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
    Interroge PubChem pour la géométrie 3D d'une molécule.

    Retourne un dict :
        cid, name, atoms [(sym,(x,y,z))], elements [sym], bonds [(i,j,order)],
        total_electrons (neutral molecule)
    """
    url = PUBCHEM_3D_URL.format(name=requests.utils.quote(molecule_name))
    resp = requests.get(url, timeout=15)

    if resp.status_code == 404:
        raise ValueError(
            f"Molécule « {molecule_name} » introuvable sur PubChem.\n"
            "Essayez un autre nom (anglais recommandé) ou une formule brute.")
    if resp.status_code != 200:
        raise ConnectionError(
            f"PubChem a répondu avec le code {resp.status_code}.\n"
            "Vérifiez votre connexion Internet et réessayez.")

    data = resp.json()
    compounds = data.get("PC_Compounds", [])
    if not compounds:
        raise ValueError(f"Aucun composé retourné pour « {molecule_name} ».")

    comp = compounds[0]
    cid = comp.get("id", {}).get("id", {}).get("cid", 0)

    atom_section = comp.get("atoms", {})
    elements_z   = atom_section.get("element", [])
    aids         = atom_section.get("aid", [])

    coords_section = comp.get("coords", [{}])[0]
    conformers     = coords_section.get("conformers", [])
    if not conformers:
        raise ValueError(
            f"Pas de conformère 3D pour « {molecule_name} » (CID {cid}).")
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
_VIEW_NUCLEI   = "Noyaux uniquement"
_VIEW_DENSITY  = "Densité électronique"
_VIEW_ESP      = "Carte potentiel (ESP)"


# ─────────────────────────  Interface  ──────────────────────

class PubChemChatbot:
    """Fenêtre Tkinter : barre d'outils + graphique 3D + chat."""

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

        self._current_mol = None       # dernier résultat PubChem
        self._display_bohr = True
        self._zoom = 10.0              # matplotlib 3D "dist" for zoom
        self._computing = False        # SCF running flag

        # Schrödinger state
        self._scf_energies = []
        self._scf_wavefunctions = []
        self._scf_grid_info = {}
        self._scf_occupancies = []
        self._scf_density = None
        self._scf_esp = None
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
            "Bienvenue ! Je suis le chatbot PubChem + Schrödinger.\n"
            "Tapez le nom d'une molécule et j'afficherai sa géométrie 3D.\n\n"
            "Exemples :\n"
            "  • water  •  methane  •  aspirin  •  caffeine\n"
            "  • show me benzene  •  cherche ethanol\n\n"
            "Barre d'outils (en haut) :\n"
            "  1. Le nombre d'électrons est rempli automatiquement\n"
            "  2. « Calculer SCF » lance le solveur Schrödinger\n"
            "  3. Sélectionnez un niveau d'énergie dans le menu\n"
            "  4. Zoomez avec la molette de la souris\n\n"
            "Commandes spéciales :\n"
            "  • bohr / angstrom — unité d'affichage\n"
            "  • aide / help — ce message"
        )

        self.root.mainloop()

    # ═══════════════════  TOOLBAR  ═══════════════════════════
    def _setup_toolbar(self):
        tb = tk.Frame(self.root, bg=self._TB_BG, height=42)
        tb.pack(fill="x", padx=6, pady=(6, 0))

        # ── Électrons ──
        tk.Label(tb, text="Électrons :", bg=self._TB_BG, fg=self._FG,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        self._electrons_var = tk.StringVar(value="0")
        self._electrons_entry = tk.Entry(
            tb, textvariable=self._electrons_var, width=5,
            bg=self._ENTRY_BG, fg=self._ENTRY_FG,
            insertbackground=self._ENTRY_FG,
            font=("Consolas", 10), relief="flat", bd=0)
        self._electrons_entry.pack(side="left", padx=(0, 8), ipady=2)

        # ── Calculer SCF ──
        self._compute_btn = tk.Button(
            tb, text="Calculer SCF", bg=self._ACCENT, fg="#1e1e2e",
            activebackground="#74c7ec", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self._on_compute_scf)
        self._compute_btn.pack(side="left", padx=(0, 12), ipady=2, ipadx=6)

        # ── separator ──
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                  padx=4, pady=6)

        # ── Niveau d'énergie ──
        tk.Label(tb, text="Niveau :", bg=self._TB_BG, fg=self._FG,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        self._level_var = tk.StringVar(value=_VIEW_NUCLEI)
        self._level_combo = ttk.Combobox(
            tb, textvariable=self._level_var, state="readonly", width=34,
            font=("Consolas", 9))
        self._level_combo["values"] = [_VIEW_NUCLEI]
        self._level_combo.pack(side="left", padx=(0, 12))
        self._level_combo.bind("<<ComboboxSelected>>", self._on_level_select)

        # ── Afficher nuage ──
        self._cloud_btn = tk.Button(
            tb, text="Afficher nuage", bg=self._BTN_BG, fg=self._BTN_FG,
            activebackground="#585b70", font=("Segoe UI", 9),
            relief="flat", cursor="hand2", command=self._on_show_cloud)
        self._cloud_btn.pack(side="left", padx=(0, 8), ipady=2, ipadx=4)

        # ── Potentiel V(r) — slice viewer ──
        self._potential_btn = tk.Button(
            tb, text="Potentiel V(r)", bg="#f9e2af", fg="#1e1e2e",
            activebackground="#f5c2e7", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self._on_show_potential)
        self._potential_btn.pack(side="left", padx=(0, 8), ipady=2, ipadx=4)

        # ── separator ──
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y",
                                                  padx=4, pady=6)

        # ── Zoom label ──
        tk.Label(tb, text="Zoom : molette",
                 bg=self._TB_BG, fg="#6c7086",
                 font=("Segoe UI", 8, "italic")).pack(side="left", padx=8)

    # ═══════════════════  3D PLOT  ══════════════════════════
    def _setup_plot(self):
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=self._BG)
        self.ax = self.fig.add_subplot(111, projection="3d",
                                       facecolor=self._BG_CHAT)
        self._style_axes()
        self.ax.set_title("Aucune molécule chargée",
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

        tk.Button(entry_frame, text="Envoyer",
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

    def _user_say(self, t):   self._append_text(f"Vous :  {t}", "user")
    def _bot_say(self, t):    self._append_text(f"Bot :  {t}", "bot")
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
                "Tapez le nom d'une molécule pour afficher sa géométrie.\n"
                "  • water, methane, aspirin, caffeine …\n"
                "  • show me benzene / cherche ethanol\n\n"
                "Barre d'outils :\n"
                "  • Modifiez le nombre d'électrons si besoin\n"
                "  • « Calculer SCF » résout l'équation de Schrödinger\n"
                "  • Choisissez un niveau dans le menu déroulant\n"
                "  • « Afficher nuage » dessine les probabilités\n"
                "  • Zoomez avec la molette\n\n"
                "Commandes :\n"
                "  • bohr / angstrom  •  info  •  clear  •  aide")
            return

        if low in ("bohr", "bohrs"):
            self._display_bohr = True
            self._bot_info("Unité → Bohr (a\u2080).")
            if self._current_mol:
                self._redraw()
            return

        if low in ("angstrom", "angstroms", "\u00e5", "ang"):
            self._display_bohr = False
            self._bot_info("Unité → \u00c5ngstr\u00f6m (\u00c5).")
            if self._current_mol:
                self._redraw()
            return

        if low in ("info", "infos"):
            if self._current_mol:
                self._show_molecule_info(self._current_mol)
            else:
                self._bot_info("Aucune molécule chargée.")
            return

        if low in ("clear", "effacer", "reset"):
            self._current_mol = None
            self._clear_scf()
            save_nuclei([])  # clear shared state file
            self.ax.clear()
            self._style_axes()
            self.ax.set_title("Aucune molécule chargée",
                              color=self._FG, fontsize=11, pad=12)
            self.canvas_mpl.draw_idle()
            self._bot_info("Graphique effacé.")
            return

        # Molecule search
        name = _extract_molecule_name(text)
        if not name:
            self._bot_error("Je n'ai pas compris. Tapez « aide ».")
            return
        self._bot_info(f"Recherche de « {name} » sur PubChem …")
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
            self.root.after(0, self._bot_error, f"Erreur : {e}")
            return
        self.root.after(0, self._on_molecule_received, mol)

    def _on_molecule_received(self, mol: dict):
        self._current_mol = mol
        self._clear_scf()
        self._electrons_var.set(str(mol["total_electrons"]))

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
            f"Molécule : {mol['name']}  (CID {mol['cid']})",
            f"Formule  : {formula}",
            f"Atomes   : {len(atoms)}     Liaisons : {len(mol['bonds'])}",
            f"Électrons (neutre) : {mol['total_electrons']}",
            "", f"Coordonnées ({unit}) :"]
        for i, (sym, (x, y, z)) in enumerate(atoms):
            lines.append(f"  {i+1:3d}  {sym:2s}  "
                         f"({x*factor:8.4f}, {y*factor:8.4f}, {z*factor:8.4f})")
        self._bot_say("\n".join(lines))

    # ═══════════════════  SCF  ══════════════════════════════
    def _clear_scf(self):
        self._scf_energies = []
        self._scf_wavefunctions = []
        self._scf_grid_info = {}
        self._scf_occupancies = []
        self._scf_density = None
        self._scf_esp = None
        self._level_var.set(_VIEW_NUCLEI)
        self._level_combo["values"] = [_VIEW_NUCLEI]

    def _get_num_electrons(self):
        try:
            return max(0, int(self._electrons_var.get().strip() or "0"))
        except ValueError:
            return 0

    def _on_compute_scf(self):
        if self._computing:
            return
        if not self._current_mol:
            self._bot_error("Chargez d'abord une molécule (tapez son nom).")
            return

        num_e = self._get_num_electrons()
        nuclei = self._current_mol["atoms"]

        self._computing = True
        self._compute_btn.config(text="Calcul en cours…", state="disabled")
        self._bot_info(
            f"Lancement SCF : {len(nuclei)} noyaux, {num_e} électrons…\n"
            "Cela peut prendre quelques secondes.")

        threading.Thread(target=self._run_scf,
                         args=(nuclei, num_e), daemon=True).start()

    def _run_scf(self, nuclei, num_e):
        try:
            result = solve_schrodinger(nuclei, num_electrons=num_e,
                                       n_grid=16, num_states=7)
            self.root.after(0, self._on_scf_done, result, num_e)
        except Exception as e:
            self.root.after(0, self._on_scf_error, str(e))

    def _on_scf_error(self, msg):
        self._computing = False
        self._compute_btn.config(text="Calculer SCF", state="normal")
        self._bot_error(f"Erreur SCF : {msg}")

    def _on_scf_done(self, result, num_e):
        self._computing = False
        self._compute_btn.config(text="Calculer SCF", state="normal")

        energies, wfs, gi, occ, density, esp = result
        if len(energies) == 0:
            self._bot_error("Le solveur n'a retourné aucun état propre.")
            return

        self._scf_energies = energies
        self._scf_wavefunctions = wfs
        self._scf_grid_info = gi
        self._scf_occupancies = occ
        self._scf_density = density
        self._scf_esp = esp

        # Fill energy listbox in chat
        lines = ["Niveaux d'énergie calculés (Hartree) :"]
        for i, E in enumerate(energies):
            ne = occ[i] if i < len(occ) else 0
            lines.append(f"  E{i} = {E:+.5f} Ha   ({ne} e\u207b)")
        self._bot_say("\n".join(lines))

        # Fill toolbar level combo
        opts = [_VIEW_NUCLEI]
        for i, E in enumerate(energies):
            ne = occ[i] if i < len(occ) else 0
            opts.append(f"E{i} = {E:+.5f} Ha  ({ne}e\u207b)")
        if density is not None and num_e > 0:
            opts.append(_VIEW_DENSITY)
        if esp is not None and num_e > 0:
            opts.append(_VIEW_ESP)
        self._level_combo["values"] = opts
        self._level_combo.set(_VIEW_NUCLEI)

        self._bot_info(
            "Calcul terminé ! Sélectionnez un niveau dans le menu\n"
            "puis cliquez « Afficher nuage » pour voir les probabilités.")

    # ── toolbar: level select ──
    def _on_level_select(self, event=None):
        self._redraw()

    def _on_show_cloud(self):
        if not self._scf_wavefunctions:
            self._bot_error(
                "Calculez d'abord (« Calculer SCF ») avant d'afficher.")
            return
        sel = self._level_var.get()
        if sel == _VIEW_NUCLEI:
            # Auto-select first orbital
            vals = self._level_combo["values"]
            if len(vals) > 1:
                self._level_combo.set(vals[1])
        self._redraw()

    # ═══════════════════  POTENTIAL SLICE VIEWER  ═══════════
    def _on_show_potential(self):
        """Open a new window with 2-D slice heatmaps of V(r) for the
        current PubChem molecule (uses potential_on_grid directly)."""
        if not self._current_mol:
            self._bot_error("Chargez d'abord une molécule.")
            return

        nuclei = self._current_mol["atoms"]
        name = self._current_mol["name"]

        self._bot_info(f"Calcul du potentiel V(r) pour {name}…")

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
        win.title(f"Potentiel V(r) — {name}")
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
            f"Fenêtre du potentiel ouverte pour {name}.\n"
            "Déplacez les curseurs pour couper à différentes positions.")

    # ═══════════════════  3D RENDERING  ═════════════════════
    def _redraw(self):
        """Main drawing routine: nuclei + optional electron cloud."""
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
            self.ax.set_title("Aucune molécule chargée",
                              color=self._FG, fontsize=11, pad=12)
            self.canvas_mpl.draw_idle()
            return

        atoms = mol["atoms"]
        bonds = mol["bonds"]
        unit = "Bohr" if self._display_bohr else "\u00c5"
        factor = 1.0 if self._display_bohr else (1.0 / ANGSTROM_TO_BOHR)

        # ── determine what to show ──
        sel = self._level_var.get()
        show_orbital = -1   # -1=nuclei, >=0 orbital index
        show_density = False
        show_esp = False

        if sel == _VIEW_DENSITY:
            show_density = True
        elif sel == _VIEW_ESP:
            show_esp = True
        elif sel != _VIEW_NUCLEI and sel:
            # parse "E2 = ..."  → orbital index 2
            try:
                show_orbital = int(sel.split("=")[0].strip()[1:])
            except (ValueError, IndexError):
                pass

        # ── title ──
        name = mol["name"]
        if show_density:
            self.ax.set_title(f"{name} — Densité électronique",
                              color=self._FG, fontsize=11, pad=12)
        elif show_esp:
            self.ax.set_title(f"{name} — Potentiel électrostatique",
                              color=self._FG, fontsize=11, pad=12)
        elif show_orbital >= 0:
            E = (self._scf_energies[show_orbital]
                 if show_orbital < len(self._scf_energies) else 0)
            self.ax.set_title(f"{name} — Orbitale {show_orbital}  "
                              f"(E={E:+.4f} Ha)",
                              color=self._FG, fontsize=11, pad=12)
        else:
            self.ax.set_title(f"{name}  (CID {mol['cid']})",
                              color=self._FG, fontsize=11, pad=12)

        # ── atom coordinates ──
        xs = np.array([x * factor for _, (x, y, z) in atoms])
        ys = np.array([y * factor for _, (x, y, z) in atoms])
        zs = np.array([z * factor for _, (x, y, z) in atoms])
        labels = [s for s, _ in atoms]
        colors = [_ELEMENT_COLORS.get(s, _DEFAULT_COLOR) for s in labels]

        # ── electron cloud dots ──
        px = py = pz = np.array([])
        gi = self._scf_grid_info
        wfs = self._scf_wavefunctions

        if show_orbital >= 0 and show_orbital < len(wfs) and gi:
            np.random.seed(42)
            prob = np.abs(wfs[show_orbital]) ** 2
            px, py, pz = sample_positions_from_probability(
                prob, gi, num_dots=3000)
            if not self._display_bohr:
                px /= ANGSTROM_TO_BOHR
                py /= ANGSTROM_TO_BOHR
                pz /= ANGSTROM_TO_BOHR
            self.ax.scatter(px, py, pz, c="#a6e3a1", s=4, alpha=0.45,
                            label=f"|\u03c8{show_orbital}|\u00b2")

        elif show_density and self._scf_density is not None and gi:
            np.random.seed(42)
            px, py, pz = sample_positions_from_probability(
                self._scf_density, gi, num_dots=4000)
            if not self._display_bohr:
                px /= ANGSTROM_TO_BOHR
                py /= ANGSTROM_TO_BOHR
                pz /= ANGSTROM_TO_BOHR
            self.ax.scatter(px, py, pz, c="#cba6f7", s=4, alpha=0.4,
                            label="\u03c1(r)")

        elif show_esp and self._scf_esp is not None and gi:
            np.random.seed(42)
            px, py, pz = sample_positions_from_probability(
                self._scf_density, gi, num_dots=4000)
            phi = get_esp_at_points(px, py, pz, self._scf_esp, gi)
            if not self._display_bohr:
                px /= ANGSTROM_TO_BOHR
                py /= ANGSTROM_TO_BOHR
                pz /= ANGSTROM_TO_BOHR
            lo, hi = np.percentile(phi, [5, 95])
            if lo >= hi:
                lo, hi = phi.min(), phi.max()
            if lo >= hi:
                lo, hi = -1.0, 1.0
            norm = mcolors.TwoSlopeNorm(
                vcenter=0.0, vmin=min(lo, -0.01), vmax=max(hi, 0.01))
            cmap = mcm.RdBu_r
            self.ax.scatter(px, py, pz, c=cmap(norm(phi)),
                            s=5, alpha=0.55)
            sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            self._colorbar = self.fig.colorbar(
                sm, ax=self.ax, shrink=0.55, pad=0.08,
                label="ESP (Ha/e)")

        # ── bonds ──
        for i1, i2, order in bonds:
            lw = 1.0 + 0.8 * (order - 1)
            self.ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]],
                         [zs[i1], zs[i2]],
                         color="#585b70", linewidth=lw, alpha=0.6)

        # ── nuclei ──
        for i, (sym, c) in enumerate(zip(labels, colors)):
            self.ax.scatter(xs[i], ys[i], zs[i], c=c, s=160, alpha=0.95,
                            edgecolors="#45475a", linewidths=0.8, zorder=5)
            self.ax.text(xs[i], ys[i], zs[i], f"  {sym}", fontsize=8,
                         color=self._FG, fontweight="bold", zorder=6)

        # ── axis limits ──
        all_x = np.concatenate([xs, px]) if len(px) else xs
        all_y = np.concatenate([ys, py]) if len(py) else ys
        all_z = np.concatenate([zs, pz]) if len(pz) else zs
        if len(all_x):
            cx = (all_x.min() + all_x.max()) / 2
            cy = (all_y.min() + all_y.max()) / 2
            cz = (all_z.min() + all_z.max()) / 2
            ext = max(all_x.max() - all_x.min(),
                      all_y.max() - all_y.min(),
                      all_z.max() - all_z.min(), 0.5)
            half = ext / 2 + 1.0
            self.ax.set_xlim(cx - half, cx + half)
            self.ax.set_ylim(cy - half, cy + half)
            self.ax.set_zlim(cz - half, cz + half)

        self.ax.set_xlabel(f"x ({unit})", color=self._FG, fontsize=9)
        self.ax.set_ylabel(f"y ({unit})", color=self._FG, fontsize=9)
        self.ax.set_zlabel(f"z ({unit})", color=self._FG, fontsize=9)

        self.ax.dist = self._zoom
        self.canvas_mpl.draw_idle()


# ─────────────────────────  Point d'entrée  ─────────────────
def main():
    PubChemChatbot()

if __name__ == "__main__":
    main()
