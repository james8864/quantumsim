"""
Chatbot PubChem — Extraction de géométries moléculaires 3D.

L'utilisateur entre une commande (en français ou anglais) contenant
le nom d'une molécule.  Le chatbot interroge l'API PubChem PUG-REST
pour récupérer les coordonnées 3D des atomes, puis les affiche sur un
plan cartésien matplotlib 3-D (positions des noyaux en Bohr).

Utilisation :
    python pubchem_chatbot.py
"""

import re
import threading
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import requests

# ─────────────────────────  Constantes  ─────────────────────
ANGSTROM_TO_BOHR = 1.8897259886          # 1 Å = 1.889… a₀
PUBCHEM_3D_URL   = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                     "compound/name/{name}/JSON?record_type=3d")

# Numéro atomique → symbole (1–118)
_ELEMENT_SYMBOLS = [
    "",                                                                      # 0 (unused)
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",           # 1-10
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",                        # 11-18
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",     # 19-29
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",                              # 30-36
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",    # 37-47
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",                              # 48-54
    "Cs", "Ba",                                                              # 55-56
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",           # 57-66
    "Ho", "Er", "Tm", "Yb", "Lu",                                           # 67-71
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",                 # 72-80
    "Tl", "Pb", "Bi", "Po", "At", "Rn",                                     # 81-86
    "Fr", "Ra",                                                              # 87-88
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",           # 89-98
    "Es", "Fm", "Md", "No", "Lr",                                           # 99-103
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",                 # 104-112
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",                                     # 113-118
]

# Couleurs par élément (CPK) pour les noyaux
_ELEMENT_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0", "Br": "#A62929",
    "I": "#940094",  "Au": "#FFD123", "Pt": "#D0D0E0",
}
_DEFAULT_COLOR = "#FF69B4"  # rose par défaut


# ─────────────────────────  PubChem  ────────────────────────

def fetch_pubchem_3d(molecule_name: str) -> dict:
    """
    Interroge PubChem pour la géométrie 3D d'une molécule.

    Retourne un dict :
        {
            "cid":      int,
            "name":     str,
            "atoms":    [(symbol, (x_bohr, y_bohr, z_bohr)), ...],
            "elements": [symbol, ...],
            "bonds":    [(i, j, order), ...],       # indices 0-based
        }
    Lève une Exception avec message lisible si la molécule
    n'est pas trouvée ou si PubChem ne fournit pas de conformère 3D.
    """
    url = PUBCHEM_3D_URL.format(name=requests.utils.quote(molecule_name))
    resp = requests.get(url, timeout=15)

    # — gestion des erreurs HTTP / JSON —
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
        raise ValueError(
            f"Aucun composé retourné pour « {molecule_name} ».")

    comp = compounds[0]
    cid = comp.get("id", {}).get("id", {}).get("cid", 0)

    # — atomes —
    atom_section = comp.get("atoms", {})
    elements_z   = atom_section.get("element", [])    # atomic numbers
    aids         = atom_section.get("aid", [])

    # — coordonnées 3D —
    coords_section = comp.get("coords", [{}])[0]
    conformers     = coords_section.get("conformers", [])
    if not conformers:
        raise ValueError(
            f"Pas de conformère 3D disponible pour « {molecule_name} » (CID {cid}).")
    conf = conformers[0]
    xs_ang = conf.get("x", [])
    ys_ang = conf.get("y", [])
    zs_ang = conf.get("z", [])

    # — liaisons —
    bond_section = comp.get("bonds", {})
    aid1_list = bond_section.get("aid1", [])
    aid2_list = bond_section.get("aid2", [])
    order_list = bond_section.get("order", [])

    # Construire un mapping aid → index 0-based
    aid_to_idx = {a: i for i, a in enumerate(aids)}

    bonds = []
    for a1, a2, order in zip(aid1_list, aid2_list, order_list):
        i1 = aid_to_idx.get(a1)
        i2 = aid_to_idx.get(a2)
        if i1 is not None and i2 is not None:
            bonds.append((i1, i2, order))

    # — assembler —
    atoms = []
    symbols = []
    for z_num, xA, yA, zA in zip(elements_z, xs_ang, ys_ang, zs_ang):
        sym = _ELEMENT_SYMBOLS[z_num] if z_num < len(_ELEMENT_SYMBOLS) else "?"
        xB = xA * ANGSTROM_TO_BOHR
        yB = yA * ANGSTROM_TO_BOHR
        zB = zA * ANGSTROM_TO_BOHR
        atoms.append((sym, (xB, yB, zB)))
        symbols.append(sym)

    return {
        "cid":      cid,
        "name":     molecule_name,
        "atoms":    atoms,
        "elements": symbols,
        "bonds":    bonds,
    }


# ─────────────────────  Analyse de commande  ────────────────

# Mots-clés que l'utilisateur peut écrire avant le nom de la molécule
_TRIGGER_PATTERNS = [
    # Français
    r"(?:cherche|montre|affiche|charge|trouve|donne|récupère|ouvre)\s+(?:la molécule\s+|la géométrie (?:de |d')?)?(.+)",
    r"(?:géométrie|structure|coordonnées|atomes)\s+(?:de |d'|du |des )?(.+)",
    # Anglais
    r"(?:show|display|load|find|get|fetch|search|open|plot|draw|visualize|give)\s+(?:me\s+)?(?:the\s+)?(?:molecule\s+|geometry (?:of |for )?)?(.+)",
    r"(?:geometry|structure|coordinates|atoms)\s+(?:of |for )?(.+)",
    # Questions
    r"(?:what does|how does|what is|qu'est[- ]ce que)\s+(.+?)(?:\s+look like)?$",
]

# Mots à ignorer s'ils se retrouvent dans le nom extrait
_STRIP_WORDS = {"please", "s'il te plaît", "stp", "svp", "molecule",
                "molécule", "the", "la", "le", "un", "une", "of", "de", "d'",
                "du", "for", "?", "!", "."}


def _extract_molecule_name(user_input: str) -> str:
    """
    Tente d'extraire le nom de la molécule depuis l'entrée utilisateur.
    Retourne le nom nettoyé, ou la ligne complète si aucun pattern ne matche
    (on laisse PubChem décider).
    """
    text = user_input.strip()
    if not text:
        return ""

    # Essayer chaque pattern
    for pat in _TRIGGER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Retirer les mots parasites
            tokens = name.split()
            tokens = [t for t in tokens if t.lower() not in _STRIP_WORDS]
            if tokens:
                return " ".join(tokens)

    # Aucun pattern → on envoie tel quel (le texte brut est peut-être un nom)
    return text


# ─────────────────────────  Interface  ──────────────────────

class PubChemChatbot:
    """Fenêtre Tkinter : chat + graphique 3D."""

    # Couleurs du thème
    _BG        = "#1e1e2e"
    _BG_CHAT   = "#181825"
    _FG        = "#cdd6f4"
    _ACCENT    = "#89b4fa"
    _USER_BG   = "#313244"
    _BOT_BG    = "#11111b"
    _ENTRY_BG  = "#313244"
    _ENTRY_FG  = "#cdd6f4"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PubChem Chatbot — Géométrie moléculaire 3D")
        self.root.geometry("1200x720")
        self.root.minsize(900, 550)
        self.root.configure(bg=self._BG)

        self._current_mol = None  # dernier résultat PubChem

        # ── disposition : gauche = plot, droite = chat ──
        main_pw = tk.PanedWindow(self.root, orient="horizontal",
                                 bg=self._BG, sashwidth=4)
        main_pw.pack(fill="both", expand=True, padx=6, pady=6)

        # === Gauche : graphique 3D ===
        self.plot_frame = tk.Frame(main_pw, bg=self._BG)
        main_pw.add(self.plot_frame, width=620)
        self._setup_plot()

        # === Droite : chat ===
        chat_frame = tk.Frame(main_pw, bg=self._BG)
        main_pw.add(chat_frame, width=540)
        self._setup_chat(chat_frame)

        # Message d'accueil
        self._bot_say(
            "Bienvenue ! Je suis le chatbot PubChem.\n"
            "Tapez le nom d'une molécule (en anglais de préférence) "
            "et j'afficherai sa géométrie 3D.\n\n"
            "Exemples :\n"
            "  • water\n"
            "  • methane\n"
            "  • aspirin\n"
            "  • caffeine\n"
            "  • show me benzene\n"
            "  • cherche ethanol\n"
            "  • geometry of ammonia\n\n"
            "Commandes spéciales :\n"
            "  • bohr / angstrom — changer l'unité d'affichage\n"
            "  • aide / help — afficher ce message"
        )

        self.root.mainloop()

    # ─────────────  Graphique 3D  ─────────────────────────────
    def _setup_plot(self):
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor=self._BG)
        self.ax = self.fig.add_subplot(111, projection="3d",
                                       facecolor=self._BG_CHAT)
        self.ax.set_xlabel("x (Bohr)", color=self._FG, fontsize=9)
        self.ax.set_ylabel("y (Bohr)", color=self._FG, fontsize=9)
        self.ax.set_zlabel("z (Bohr)", color=self._FG, fontsize=9)
        self.ax.set_title("Aucune molécule chargée",
                          color=self._FG, fontsize=11, pad=12)
        self.ax.tick_params(colors=self._FG, labelsize=7)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_mpl.get_tk_widget().pack(fill="both", expand=True)

        # Unité d'affichage (Bohr par défaut)
        self._display_bohr = True

    # ─────────────  Chat  ─────────────────────────────────────
    def _setup_chat(self, parent):
        # Titre
        header = tk.Label(parent, text="💬  PubChem Chat",
                          font=("Segoe UI", 13, "bold"),
                          bg=self._BG, fg=self._ACCENT, anchor="w")
        header.pack(fill="x", pady=(0, 4))

        # Zone de messages
        self.chat_text = tk.Text(parent, wrap="word",
                                 state="disabled",
                                 bg=self._BG_CHAT, fg=self._FG,
                                 font=("Consolas", 10),
                                 relief="flat", padx=10, pady=8,
                                 insertbackground=self._FG,
                                 selectbackground=self._ACCENT)
        self.chat_text.pack(fill="both", expand=True, pady=(0, 4))

        # Tags pour colorer les messages
        self.chat_text.tag_configure("user",
                                     foreground="#a6e3a1",
                                     font=("Consolas", 10, "bold"))
        self.chat_text.tag_configure("bot",
                                     foreground=self._FG,
                                     font=("Consolas", 10))
        self.chat_text.tag_configure("error",
                                     foreground="#f38ba8",
                                     font=("Consolas", 10, "italic"))
        self.chat_text.tag_configure("info",
                                     foreground="#89dceb",
                                     font=("Consolas", 10))

        # Barre de saisie
        entry_frame = tk.Frame(parent, bg=self._BG)
        entry_frame.pack(fill="x")

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(entry_frame,
                              textvariable=self.entry_var,
                              bg=self._ENTRY_BG, fg=self._ENTRY_FG,
                              insertbackground=self._ENTRY_FG,
                              font=("Consolas", 11),
                              relief="flat", bd=0)
        self.entry.pack(side="left", fill="x", expand=True,
                        ipady=6, padx=(0, 4))
        self.entry.bind("<Return>", self._on_enter)

        send_btn = tk.Button(entry_frame, text="Envoyer",
                             bg=self._ACCENT, fg="#1e1e2e",
                             activebackground="#74c7ec",
                             font=("Segoe UI", 10, "bold"),
                             relief="flat", cursor="hand2",
                             command=self._on_enter)
        send_btn.pack(side="right", ipady=4, ipadx=8)

        self.entry.focus_set()

    # ─────────────  Messages  ─────────────────────────────────
    def _append_text(self, text: str, tag: str = "bot"):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text + "\n\n", tag)
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _user_say(self, text: str):
        self._append_text(f"Vous :  {text}", "user")

    def _bot_say(self, text: str):
        self._append_text(f"Bot :  {text}", "bot")

    def _bot_error(self, text: str):
        self._append_text(f"Bot :  ⚠ {text}", "error")

    def _bot_info(self, text: str):
        self._append_text(f"Bot :  ℹ {text}", "info")

    # ─────────────  Gestion des entrées  ──────────────────────
    def _on_enter(self, event=None):
        text = self.entry_var.get().strip()
        if not text:
            return
        self.entry_var.set("")
        self._user_say(text)
        self._process_command(text)

    def _process_command(self, text: str):
        low = text.lower().strip()

        # — Commandes spéciales —
        if low in ("aide", "help", "?"):
            self._bot_say(
                "Tapez le nom d'une molécule pour afficher sa géométrie 3D.\n"
                "  • water, methane, aspirin, caffeine …\n"
                "  • show me benzene / cherche ethanol\n"
                "  • geometry of ammonia\n\n"
                "Commandes :\n"
                "  • bohr     — afficher les coordonnées en Bohr (défaut)\n"
                "  • angstrom — afficher les coordonnées en Ångströms\n"
                "  • info     — ré-afficher les infos de la molécule courante\n"
                "  • clear    — effacer le graphique\n"
                "  • aide     — ce message"
            )
            return

        if low in ("bohr", "bohrs"):
            self._display_bohr = True
            self._bot_info("Unité d'affichage → Bohr (a₀).")
            if self._current_mol:
                self._plot_molecule(self._current_mol)
            return

        if low in ("angstrom", "angstroms", "å", "ang"):
            self._display_bohr = False
            self._bot_info("Unité d'affichage → Ångström (Å).")
            if self._current_mol:
                self._plot_molecule(self._current_mol)
            return

        if low in ("info", "infos"):
            if self._current_mol:
                self._show_molecule_info(self._current_mol)
            else:
                self._bot_info("Aucune molécule chargée. "
                               "Tapez un nom de molécule.")
            return

        if low in ("clear", "effacer", "reset"):
            self._current_mol = None
            self.ax.clear()
            self.ax.set_title("Aucune molécule chargée",
                              color=self._FG, fontsize=11, pad=12)
            self.canvas_mpl.draw_idle()
            self._bot_info("Graphique effacé.")
            return

        # — Recherche d'une molécule —
        name = _extract_molecule_name(text)
        if not name:
            self._bot_error("Je n'ai pas compris. Tapez « aide » pour l'aide.")
            return

        self._bot_info(f"Recherche de « {name} » sur PubChem …")

        # Lancer la requête dans un thread pour ne pas geler l'interface
        threading.Thread(target=self._fetch_and_display,
                         args=(name,), daemon=True).start()

    def _fetch_and_display(self, name: str):
        """Thread secondaire : requête PubChem puis mise à jour UI."""
        try:
            mol = fetch_pubchem_3d(name)
        except (ValueError, ConnectionError) as e:
            self.root.after(0, self._bot_error, str(e))
            return
        except Exception as e:
            self.root.after(0, self._bot_error,
                            f"Erreur inattendue : {e}")
            return

        # Retour au thread principal pour mettre à jour l'interface
        self.root.after(0, self._on_molecule_received, mol)

    def _on_molecule_received(self, mol: dict):
        self._current_mol = mol
        self._show_molecule_info(mol)
        self._plot_molecule(mol)

    # ─────────────  Infos molécule  ───────────────────────────
    def _show_molecule_info(self, mol: dict):
        atoms = mol["atoms"]
        n_atoms = len(atoms)

        # Compter par élément
        counts = {}
        for sym, _ in atoms:
            counts[sym] = counts.get(sym, 0) + 1
        formula = "".join(f"{s}{(c if c > 1 else '')}"
                          for s, c in sorted(counts.items()))

        unit = "Bohr" if self._display_bohr else "Å"
        factor = 1.0 if self._display_bohr else (1.0 / ANGSTROM_TO_BOHR)

        lines = [
            f"Molécule : {mol['name']}  (CID {mol['cid']})",
            f"Formule  : {formula}",
            f"Atomes   : {n_atoms}",
            f"Liaisons : {len(mol['bonds'])}",
            f"",
            f"Coordonnées ({unit}) :"
        ]
        for i, (sym, (x, y, z)) in enumerate(atoms):
            lines.append(
                f"  {i+1:3d}  {sym:2s}  "
                f"({x*factor:8.4f}, {y*factor:8.4f}, {z*factor:8.4f})"
            )

        self._bot_say("\n".join(lines))

    # ─────────────  Rendu 3D  ─────────────────────────────────
    def _plot_molecule(self, mol: dict):
        self.ax.clear()

        atoms = mol["atoms"]
        bonds = mol["bonds"]

        unit = "Bohr" if self._display_bohr else "Å"
        factor = 1.0 if self._display_bohr else (1.0 / ANGSTROM_TO_BOHR)

        # Coordonnées et couleurs
        xs, ys, zs = [], [], []
        colors = []
        labels = []
        for sym, (x, y, z) in atoms:
            xs.append(x * factor)
            ys.append(y * factor)
            zs.append(z * factor)
            colors.append(_ELEMENT_COLORS.get(sym, _DEFAULT_COLOR))
            labels.append(sym)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        # Dessiner les liaisons
        for i1, i2, order in bonds:
            lw = 1.0 + 0.8 * (order - 1)   # simple=1, double=1.8, triple=2.6
            self.ax.plot([xs[i1], xs[i2]],
                         [ys[i1], ys[i2]],
                         [zs[i1], zs[i2]],
                         color="#585b70", linewidth=lw, alpha=0.7)

        # Dessiner les noyaux
        # Taille proportionnelle au numéro atomique (un peu)
        for i, (sym, c) in enumerate(zip(labels, colors)):
            self.ax.scatter(xs[i], ys[i], zs[i],
                            c=c, s=180, alpha=0.95,
                            edgecolors="#45475a", linewidths=0.8,
                            zorder=5)
            self.ax.text(xs[i], ys[i], zs[i],
                         f"  {sym}", fontsize=8, color=self._FG,
                         fontweight="bold", zorder=6)

        # Axes et titre
        self.ax.set_xlabel(f"x ({unit})", color=self._FG, fontsize=9)
        self.ax.set_ylabel(f"y ({unit})", color=self._FG, fontsize=9)
        self.ax.set_zlabel(f"z ({unit})", color=self._FG, fontsize=9)
        self.ax.set_title(f"{mol['name']}  (CID {mol['cid']})",
                          color=self._FG, fontsize=11, pad=12)
        self.ax.tick_params(colors=self._FG, labelsize=7)

        # Limites symétriques
        if len(xs) > 0:
            cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
            ext = max(xs.max() - xs.min(),
                      ys.max() - ys.min(),
                      zs.max() - zs.min(), 0.5)
            half = ext / 2 + 1.0
            self.ax.set_xlim(cx - half, cx + half)
            self.ax.set_ylim(cy - half, cy + half)
            self.ax.set_zlim(cz - half, cz + half)

        self.canvas_mpl.draw_idle()


# ─────────────────────────  Point d'entrée  ─────────────────
def main():
    PubChemChatbot()

if __name__ == "__main__":
    main()
