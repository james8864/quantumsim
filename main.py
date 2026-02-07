"""
Variables:

Independent variables:
- V(psi) - potential energy graph
- mass of particle
- time
- initial wavefunction (coefficients (only real))

Dependent variables:
- Hamiltonians
- final wavefunction (coefficients only)
- probability of each eigenstate

Control variables:
- five energy levels

Constants:
- h bar = 1
- dx (smallest measure of position)

-------------------------------------------------------------
Code:

Input: (use function input mechanism from previous hackathon)
1- Potential energy function
2- Initial wavefunction

Process:
1- Construction of hamiltonian matrix
    1.1- 3

2- Compute eigenvalues and eigenvectors of the Hamiltonian

3- Compute Schrodinger's equation to get the resulting
    wavefunction in position eigenbasis

4- Compute probability of being in each eigenstate

Output:
1- Render Cartesian plane
2- Render single state wavefunctions for each of the 5 energy levels
3- Display probability of each eigenstate

"""

import pygame
import numpy as np
import sys
from scipy.linalg import eigh
from scipy.sparse import diags
import numexpr as ne

# Initialize Pygame
pygame.init()


def safe_formula(expr, x):
    allowed = {
        "x": x,
        "pi": np.pi,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs
    }

    try:
        return ne.evaluate(expr, local_dict=allowed)
    except Exception:
        return None


class Button:
    def __init__(self, rect, text, font, color, hover_color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, self.hover_color if is_hover else self.color, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)

        text_surf = self.font.render(self.text, True, self.text_color)
        screen.blit(
            text_surf,
            text_surf.get_rect(center=self.rect.center)
        )

    def clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


class QuantumSystemVisualizer:
    def __init__(self, width=1200, height=800):
        # Screen dimensions
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Quantum System Simulation")
        self.formula_text = ""
        self.formula_active = False
        self.mode = "menu"  # "menu" or "simulation"
        self.selected_potential = None
        self.selected_wavefunction = None

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 50, 50)
        self.BLUE = (50, 100, 255)
        self.GREEN = (50, 200, 100)
        self.PURPLE = (180, 70, 220)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.GRAY = (100, 100, 100)

        # Fonts
        self.font = pygame.font.SysFont('Arial', 18)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Quantum system parameters
        self.x_min = -10
        self.x_max = 10
        self.n_points = 500
        self.hbar = 1.0
        self.mass = 1.0

        # Create position grid
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        self.dx = self.x[1] - self.x[0]

        # System state
        self.V = None
        self.psi0 = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.probabilities = None
        self.n_states = 5

        # Visualization parameters
        self.wavefunction_colors = [self.RED, self.BLUE, self.GREEN, self.PURPLE, self.YELLOW]

        # Layout parameters
        self.graph_padding = 50
        self.probability_panel_width = 300

        # Time evolution
        self.time = 0.0
        self.time_step = 0.05

        # Default potential and wavefunction
        # self.set_default_system()

        # Menu Buttons
        self.menu_buttons = {
            "harmonic": Button((100, 150, 250, 50), "Harmonic Potential", self.font, (60, 60, 80), (100, 100, 150)),
            "square_well": Button((100, 220, 250, 50), "Infinite Square Well", self.font, (60, 60, 80),
                                  (100, 100, 150)),
            "barrier": Button((100, 290, 250, 50), "Finite Barrier", self.font, (60, 60, 80), (100, 100, 150)),
            "free": Button((100, 360, 250, 50), "Free Particle", self.font, (60, 60, 80), (100, 100, 150)),
            "formula": Button((100, 430, 250, 50), "Type V(x) Formula", self.font, (80, 60, 60), (150, 100, 100))}

        self.back_button = Button(
            rect=(20, 40, 160, 40),
            text="← Back to Menu",
            font=self.font,
            color=(80, 60, 60),
            hover_color=(140, 90, 90))

    def draw_menu(self):
        self.screen.fill((20, 20, 30))

        title = self.title_font.render("Choose Potential Energy Function", True, (255, 255, 255))
        self.screen.blit(title, (100, 80))

        for button in self.menu_buttons.values():
            button.draw(self.screen)

        pygame.display.flip()

    def set_default_system(self):
        """Set up default quantum system (harmonic oscillator)"""
        # Default potential: Harmonic oscillator
        k = 0.5  # Spring constant
        self.V = 0.5 * k * self.x ** 2

        # Default initial wavefunction: Gaussian wavepacket
        x0 = 2.0  # Initial position
        sigma = 1.0  # Width
        self.psi0 = np.exp(-((self.x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * 2 * self.x)

        # Normalize
        norm = np.sqrt(np.trapz(np.abs(self.psi0) ** 2, self.x))
        self.psi0 = self.psi0 / norm

        # Compute eigenstates
        self.compute_eigenstates()

    def construct_hamiltonian(self):
        """Construct the Hamiltonian matrix"""
        # Kinetic energy operator
        main_diag = np.ones(self.n_points) * (self.hbar ** 2) / (self.mass * self.dx ** 2)
        off_diag = -0.5 * (self.hbar ** 2) / (self.mass * self.dx ** 2) * np.ones(self.n_points - 1)

        # Construct matrices
        T = diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(self.n_points, self.n_points))
        V_matrix = diags(self.V, 0, shape=(self.n_points, self.n_points))

        return T + V_matrix

    def compute_eigenstates(self):
        """Compute eigenvalues and eigenvectors"""
        H = self.construct_hamiltonian()
        H_dense = H.toarray()

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = eigh(H_dense, subset_by_index=[0, self.n_states - 1])

        # Normalize eigenvectors
        for i in range(self.n_states):
            norm = np.sqrt(np.trapz(np.abs(self.eigenvectors[:, i]) ** 2, self.x))
            self.eigenvectors[:, i] = self.eigenvectors[:, i] / norm

        # Compute probabilities if initial wavefunction exists
        if self.psi0 is not None:
            self.compute_probabilities()

    def compute_probabilities(self):
        """Compute probability of being in each eigenstate"""
        self.probabilities = np.zeros(self.n_states)

        for i in range(self.n_states):
            # Project initial wavefunction onto eigenstate
            projection = np.trapz(np.conj(self.eigenvectors[:, i]) * self.psi0, self.x)
            self.probabilities[i] = np.abs(projection) ** 2

    def compute_time_evolution(self, t):
        """Compute wavefunction at time t"""
        psi_t = np.zeros(self.n_points, dtype=complex)

        for i in range(self.n_states):
            # Add contribution from each eigenstate with time evolution factor
            psi_t += (np.trapz(np.conj(self.eigenvectors[:, i]) * self.psi0, self.x) *
                      np.exp(-1j * self.eigenvalues[i] * t) *
                      self.eigenvectors[:, i])

        return psi_t

    def draw_cartesian_plane(self, surface, x, y, width, height):
        """Draw Cartesian coordinate system"""
        # Clear background
        pygame.draw.rect(surface, self.BLACK, (x, y, width, height))

        # Calculate origin position
        x_origin = x + width // 2
        y_origin = y + height // 2

        # Draw grid
        grid_color = (40, 40, 40)
        grid_spacing = 50

        # Vertical grid lines
        for i in range(0, width, grid_spacing):
            pygame.draw.line(surface, grid_color, (x + i, y), (x + i, y + height), 1)

        # Horizontal grid lines
        for i in range(0, height, grid_spacing):
            pygame.draw.line(surface, grid_color, (x, y + i), (x + width, y + i), 1)

        # Draw axes
        pygame.draw.line(surface, self.WHITE, (x, y_origin), (x + width, y_origin), 2)  # X-axis
        pygame.draw.line(surface, self.WHITE, (x_origin, y), (x_origin, y + height), 2)  # Y-axis

        # Axis labels
        x_label = self.font.render("Position (x)", True, self.WHITE)
        y_label = self.font.render("Energy / Amplitude", True, self.WHITE)
        surface.blit(x_label, (x + width // 2 - 40, y + height - 20))

        # Rotate Y label
        y_label_rotated = pygame.transform.rotate(y_label, 90)
        surface.blit(y_label_rotated, (x + 10, y + height // 2 - 60))

        return x_origin, y_origin, width, height

    def draw_potential(self, surface, x_origin, y_origin, graph_width, graph_height):
        """Draw potential energy function"""
        if self.V is None:
            return

        # Scale potential to fit in graph
        V_max = np.max(np.abs(self.V))
        if V_max == 0:
            V_max = 1

        scale_y = graph_height / (4 * V_max)
        offset_y = y_origin

        points = []
        for i in range(self.n_points):
            x_pos = x_origin + (self.x[i] - self.x_min) / (self.x_max - self.x_min) * graph_width - graph_width / 2
            y_pos = offset_y - self.V[i] * scale_y
            points.append((x_pos, y_pos))

        # Draw potential curve
        if len(points) > 1:
            pygame.draw.lines(surface, self.CYAN, False, points, 2)

        # Label
        potential_label = self.font.render("Potential V(x)", True, self.CYAN)
        surface.blit(potential_label, (x_origin + graph_width // 2 - 60, y_origin - graph_height // 2 + 10))

    def draw_wavefunctions(self, surface, x_origin, y_origin, graph_width, graph_height, show_time_evolution=True):
        """Draw wavefunctions for each energy level"""
        if self.eigenvectors is None:
            return

        # Scale wavefunctions to fit in graph
        max_amplitude = np.max(np.abs(self.eigenvectors))
        if max_amplitude == 0:
            max_amplitude = 1

        scale_y = graph_height / (6 * max_amplitude)
        energy_scale = graph_height / (2 * np.max(np.abs(self.eigenvalues))) if np.max(
            np.abs(self.eigenvalues)) > 0 else 1

        # Draw eigenstate wavefunctions
        for state in range(self.n_states):
            color = self.wavefunction_colors[state % len(self.wavefunction_colors)]

            # Calculate vertical offset based on energy level
            energy_offset = -self.eigenvalues[state] * energy_scale

            points = []
            for i in range(self.n_points):
                x_pos = x_origin + (self.x[i] - self.x_min) / (self.x_max - self.x_min) * graph_width - graph_width / 2
                y_pos = y_origin + energy_offset - self.eigenvectors[i, state].real * scale_y
                points.append((x_pos, y_pos))

            # Draw wavefunction curve
            if len(points) > 1:
                pygame.draw.lines(surface, color, False, points, 2)

            # Draw energy level line
            y_level = y_origin + energy_offset
            pygame.draw.line(surface, self.GRAY,
                             (x_origin - graph_width // 2, y_level),
                             (x_origin + graph_width // 2, y_level), 1)

            # Label energy level
            energy_text = self.font.render(f"E{state} = {self.eigenvalues[state]:.3f}", True, color)
            surface.blit(energy_text, (x_origin + graph_width // 2 + 10, y_level - 10))

        # Draw time-evolved wavefunction if requested
        if show_time_evolution and self.psi0 is not None:
            psi_t = self.compute_time_evolution(self.time)

            points = []
            for i in range(self.n_points):
                x_pos = x_origin + (self.x[i] - self.x_min) / (self.x_max - self.x_min) * graph_width - graph_width / 2
                y_pos = y_origin - psi_t[i].real * scale_y * 2  # Scale up for visibility
                points.append((x_pos, y_pos))

            # Draw time-evolved wavefunction
            if len(points) > 1:
                pygame.draw.lines(surface, self.WHITE, False, points, 3)

    def draw_probability_panel(self, surface, x, y, width, height):
        """Draw panel showing eigenstate probabilities"""
        # Background
        pygame.draw.rect(surface, (30, 30, 40), (x, y, width, height))
        pygame.draw.rect(surface, self.WHITE, (x, y, width, height), 2)

        # Title
        title = self.title_font.render("Eigenstate Probabilities", True, self.WHITE)
        surface.blit(title, (x + width // 2 - title.get_width() // 2, y + 20))

        # Time display
        time_text = self.font.render(f"Time: {self.time:.2f}", True, self.WHITE)
        surface.blit(time_text, (x + 20, y + 60))

        # Draw probabilities
        y_offset = y + 100
        if self.probabilities is not None:
            total_prob = 0

            for i in range(self.n_states):
                prob = self.probabilities[i]
                total_prob += prob

                # Probability bar
                bar_width = int((width - 100) * prob)
                color = self.wavefunction_colors[i % len(self.wavefunction_colors)]

                # Bar background
                pygame.draw.rect(surface, (50, 50, 60), (x + 20, y_offset, width - 40, 30))
                # Bar fill
                pygame.draw.rect(surface, color, (x + 20, y_offset, bar_width, 30))
                # Bar border
                pygame.draw.rect(surface, self.WHITE, (x + 20, y_offset, width - 40, 30), 1)

                # Probability text
                prob_text = self.font.render(f"State {i}: {prob:.4f}", True, self.WHITE)
                surface.blit(prob_text, (x + 30, y_offset + 5))

                y_offset += 40

            # Total probability (should be ~1)
            total_text = self.font.render(f"Total: {total_prob:.6f}", True,
                                          self.GREEN if abs(total_prob - 1.0) < 0.01 else self.RED)
            surface.blit(total_text, (x + 20, y_offset + 10))

            # Instructions
            y_offset += 50
            instructions = [
                "CONTROLS:",
                "SPACE: Pause/Resume time",
                "R: Reset time to 0",
                "T: Toggle time evolution",
                "UP/DOWN: Adjust time speed",
                "ESC: Exit"
            ]

            for i, instruction in enumerate(instructions):
                inst_text = self.font.render(instruction, True, (200, 200, 200))
                surface.blit(inst_text, (x + 20, y_offset + i * 25))

    def draw_legend(self, surface, x, y):
        """Draw color legend"""
        legend_y = y
        for i in range(self.n_states):
            color = self.wavefunction_colors[i % len(self.wavefunction_colors)]
            pygame.draw.rect(surface, color, (x, legend_y, 20, 20))
            state_text = self.font.render(f"ψ{i} (E{i})", True, self.WHITE)
            surface.blit(state_text, (x + 30, legend_y))
            legend_y += 30

        # Time evolved wavefunction
        pygame.draw.line(surface, self.WHITE, (x, legend_y + 10), (x + 20, legend_y + 10), 3)
        time_text = self.font.render("Time-evolved ψ(t)", True, self.WHITE)
        surface.blit(time_text, (x + 30, legend_y))
        legend_y += 30

        # Potential
        pygame.draw.line(surface, self.CYAN, (x, legend_y + 10), (x + 20, legend_y + 10), 2)
        pot_text = self.font.render("Potential V(x)", True, self.CYAN)
        surface.blit(pot_text, (x + 30, legend_y))

    def draw(self):
        """Draw everything on screen"""
        self.screen.fill(self.BLACK)

        # Main graph area (Cartesian plane)
        graph_x = self.graph_padding
        graph_y = self.graph_padding
        graph_width = self.width - self.graph_padding * 2 - self.probability_panel_width
        graph_height = self.height - self.graph_padding * 2

        # Draw Cartesian plane
        x_origin, y_origin, graph_width, graph_height = self.draw_cartesian_plane(
            self.screen, graph_x, graph_y, graph_width, graph_height
        )

        # Draw potential
        self.draw_potential(self.screen, x_origin, y_origin, graph_width, graph_height)

        # Draw wavefunctions
        self.draw_wavefunctions(self.screen, x_origin, y_origin, graph_width, graph_height)

        # Draw probability panel
        panel_x = self.width - self.probability_panel_width
        panel_y = self.graph_padding
        panel_width = self.probability_panel_width - self.graph_padding
        panel_height = graph_height

        self.draw_probability_panel(self.screen, panel_x, panel_y, panel_width, panel_height)

        # Draw legend
        self.draw_legend(self.screen, graph_x + 20, graph_y + 20)

        # Draw title
        title = self.title_font.render("Quantum System Simulation - Schrödinger Equation Solver", True, self.WHITE)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))
        self.back_button.draw(self.screen)
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            # Back to menu button (simulation mode)
            if self.mode == "simulation" and event.type == pygame.MOUSEBUTTONDOWN:
                if self.back_button.clicked(event):
                    self.mode = "menu"
                    self.time = 0.0
                    self.V = None
                    self.psi0 = None
                    self.eigenvalues = None
                    self.eigenvectors = None
                    self.probabilities = None
            # --------------------
            # GLOBAL KEYS
            # --------------------
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.time_step = 0 if self.time_step != 0 else 0.05
                elif event.key == pygame.K_r:
                    self.time = 0.0
                elif event.key == pygame.K_UP and self.time_step != 0:
                    self.time_step *= 1.5
                elif event.key == pygame.K_DOWN and self.time_step != 0:
                    self.time_step /= 1.5

            # --------------------
            # MENU MODE (MOUSE)
            # --------------------
            if self.mode == "menu" and event.type == pygame.MOUSEBUTTONDOWN:

                if self.menu_buttons["harmonic"].clicked(event):
                    self.V = 0.5 * self.x ** 2
                    self.finish_setup()

                elif self.menu_buttons["square_well"].clicked(event):
                    self.V = np.where(np.abs(self.x) > 4, 1e6, 0)
                    self.finish_setup()

                elif self.menu_buttons["barrier"].clicked(event):
                    self.V = np.where(np.abs(self.x) < 1, 20, 0)
                    self.finish_setup()

                elif self.menu_buttons["free"].clicked(event):
                    self.V = np.zeros_like(self.x)
                    self.finish_setup()

                elif self.menu_buttons["formula"].clicked(event):
                    self.formula_text = ""
                    self.mode = "formula_input"

            # --------------------
            # FORMULA INPUT MODE
            # --------------------
            elif self.mode == "formula_input" and event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RETURN:
                    V_try = safe_formula(self.formula_text, self.x)
                    if V_try is not None:
                        self.V = V_try
                        self.finish_setup()
                    else:
                        self.formula_text = ""

                elif event.key == pygame.K_BACKSPACE:
                    self.formula_text = self.formula_text[:-1]

                else:
                    self.formula_text += event.unicode

        return True

    def finish_setup(self):
        self.psi0 = gaussian_wavepacket(self.x)
        self.psi0 /= np.sqrt(np.trapz(np.abs(self.psi0) ** 2, self.x))
        self.compute_eigenstates()
        self.mode = "simulation"

    def update(self, dt):
        """Update simulation state"""
        if self.time_step != 0:
            self.time += self.time_step * dt
            # Keep time from getting too large
            if self.time > 100:
                self.time = 0

    def draw_formula_input(self):
        self.screen.fill((20, 20, 30))

        prompt = self.font.render("Enter V(x) formula (e.g. 0.5*x**2 + 5*sin(x))", True, (255, 255, 255))
        self.screen.blit(prompt, (100, 150))

        box = pygame.Rect(100, 200, 600, 50)
        pygame.draw.rect(self.screen, (40, 40, 60), box)
        pygame.draw.rect(self.screen, (255, 255, 255), box, 2)

        text = self.font.render(self.formula_text, True, (0, 255, 255))
        self.screen.blit(text, (box.x + 10, box.y + 10))

        pygame.display.flip()

    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(60) / 1000.0  # Delta time in seconds

            # Handle events
            running = self.handle_events()

            # Update simulation
            self.update(dt)

            # Draw everything
            if self.mode == "menu":
                self.draw_menu()
            elif self.mode == "formula_input":
                self.draw_formula_input()
            else:
                self.draw()

        pygame.quit()
        sys.exit()


# Example potential and wavefunction functions for user input
def harmonic_potential(x, k=0.5):
    return 0.5 * k * x ** 2


def square_well_potential(x, width=5, depth=10):
    V = np.zeros_like(x)
    V[np.abs(x) > width / 2] = depth
    return V


def gaussian_wavepacket(x, x0=0.0, sigma=1.0, k0=2.0):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * k0 * x)


# Create and run the visualizer
if __name__ == "__main__":
    visualizer = QuantumSystemVisualizer()
    visualizer.run()