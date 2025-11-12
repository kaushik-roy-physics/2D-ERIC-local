# Code for generating .mp4 movies showing phase evolution in the (b,\Lambda) space at constant K for the 2D ERIC model with excitability (2DE+ex) as described in the paper: "Foci, waves, excitability: self-organization of phase waves in a model of asymmetrically coupled embryonic oscillators".The ERIC_2D_exe model represents excitable phase oscillators on a 2D lattice with the dynamics:
#      dθ/dt = ω - b*sin(θ) + K * [Σ_neighbors (sin(θ_j - θ_i) + L*sin²(θ_j - θ_i))] where:
#        - θ: phase of oscillator
#        - ω: natural frequency
#        - b: excitability parameter 
#        - K: coupling strength (K = a * (ω_max - ω_min))
#        - L: asymmetry parameter

#Each movie is generated for a certain K=a\Delta_{\omega}. For each movie, each frame is a 10x10 grid showing the phasemap in the (b,\Lambda) parameter space at a certain timepoint. The parameter ranges can be changed by the user depending on what they want to study. 
#
#
#
# Author: Kaushik Roy
# Email: kr70@rice.edu
#
#
#
#
#---------Requires JAX for implementation----------
#-----Please install using: 'pip install -U "jax[cuda12]' .It is advisable to ensure that CUDA toolkit is installed properly before installing JAX. Follow: https://docs.jax.dev/en/latest/installation.html for more details
#
#
#


import os
import numpy as np

# Set legacy seed for reproducibility
np.random.seed(12345)

os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import jax
import jax.numpy as jnp
from jax import jit

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import time
import argparse
import json

from typing import List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from functools import partial

# Configure JAX to use GPU
jax.config.update('jax_platform_name', 'gpu')


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    N: int = 50
    dt: float = 0.01
    w_min: float = 2 * np.pi / 180
    w_max: float = 2 * np.pi / 150
    times: List[int] = None
    
    # Parameter ranges for ERIC_2D_exe
    b_range: tuple = (0.0, 0.06, 10)  # (start, stop, num)
    L_range: tuple = (1.6, 2.0, 10)   # (start, stop, num)
    a_values: List[float] = None       # Fixed a values
    
    def __post_init__(self):
        if self.times is None:
            self.times = [100, 500, 1000, 1200, 1500, 2000, 2500, 
                         3000, 4000, 5000, 7000, 10000, 12000, 15000]
        if self.a_values is None:
            self.a_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class PhaseConditionConfig:
    """Configuration for a phase condition"""
    phase_func: str
    phase_desc: str


class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_default_config(config_path: str = 'simulation_config_bL.json'):
        """Save default configuration to JSON file"""
        default_config = {
            "simulation": {
                "N": 50,
                "dt": 0.01,
                "w_min": 0.03490658503988659,  # 2*pi/180
                "w_max": 0.04188790204786391,  # 2*pi/150
                "times": [100, 500, 1000, 1200, 1500, 2000, 2500, 
                         3000, 4000, 5000, 7000, 10000, 12000, 15000],
                "b_range": [0.0, 0.06, 10],
                "L_range": [1.6, 2.0, 10],
                "a_values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            },
            "phase_conditions": {
                "randomp": {
                    "phase_func": "randomp_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi, \pi)$"
                },
                "narrowp1": {
                    "phase_func": "narrowp1_phase",
                    "phase_desc": r"$\theta_{in}=U(0, \pi/2)$"
                },
               "widerp3": {
                    "phase_func": "widerp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/4, \pi)$",
                },
                "narrowp3": {
                    "phase_func": "narrowp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/2, \pi/2)$",
                },
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Default configuration saved to {config_path}")
        return default_config


class ERICExcitableSimulator:
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize ERIC excitable simulator.
        
        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation configuration. If None, uses defaults.
        """
        self.config = config if config is not None else SimulationConfig()
        
        self.N = self.config.N
        self.dt = self.config.dt
        self.w_min = self.config.w_min
        self.w_max = self.config.w_max
        self.times = self.config.times
        
        # Phase condition configurations
        self.phase_conditions: dict[str, PhaseConditionConfig] = {}
        
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
    
    def load_phase_conditions_from_config(self, config_dict: dict):
        """Load phase conditions from configuration dictionary"""
        for name, params in config_dict.items():
            self.phase_conditions[name] = PhaseConditionConfig(**params)
    
    # ==================== Phase Initialization Functions ====================
    
    def narrowp1_phase(self, N: int) -> np.ndarray:
        """Generate phase distribution: U(0, π/2)"""
        return np.random.uniform(0, np.pi/2, (N, N))
    
    def widerp3_phase(self, N: int) -> np.ndarray:
        """Generate phase distribution: U(-π/4, π)"""
        return np.random.uniform(-np.pi/4, np.pi, (N, N))
    
    def narrowp3_phase(self, N: int) -> np.ndarray:
        """Generate phase distribution: U(-π/2, π/2)"""
        return np.random.uniform(-np.pi/2, np.pi/2, (N, N))
    
    def randomp_phase(self, N: int) -> np.ndarray:
        """Generate phase distribution: U(-π, π)"""
        return np.random.uniform(-np.pi, np.pi, (N, N))
    
    def get_phase_function(self, func_name: str) -> Callable:
        """Get phase initialization function by name"""
        return getattr(self, func_name)
    
    def omega_uniform(self) -> np.ndarray:
        """Generate uniform omega distribution"""
        return np.random.uniform(self.w_min, self.w_max, size=(self.N, self.N))
    
    # ==================== JAX Static Methods (JIT-compiled) ====================
    
    @staticmethod
    @jit
    def _get_neighbors_jax(phi: jnp.ndarray) -> tuple:
        """
        Get neighboring phases with open boundary conditions.
        
        Works for both single (N, N) and batched (B, N, N) arrays using
        periodic boundary conditions with corrections at edges.
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (N, N) or (B, N, N)
        
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (phi_up, phi_down, phi_left, phi_right) - neighboring phase values
        """
        # Roll operations work on the last two dimensions
        phi_up = jnp.roll(phi, -1, axis=-2)
        phi_down = jnp.roll(phi, 1, axis=-2)
        phi_left = jnp.roll(phi, -1, axis=-1)
        phi_right = jnp.roll(phi, 1, axis=-1)
        
        # Apply open boundary conditions
        # The ellipsis (...) handles both (N, N) and (B, N, N) cases
        phi_up = phi_up.at[..., -1, :].set(phi[..., -1, :])
        phi_down = phi_down.at[..., 0, :].set(phi[..., 0, :])
        phi_left = phi_left.at[..., :, -1].set(phi[..., :, -1])
        phi_right = phi_right.at[..., :, 0].set(phi[..., :, 0])
        
        return phi_up, phi_down, phi_left, phi_right
    
    @staticmethod
    @jit
    def _ERIC_2D_excitable_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                               K: jnp.ndarray, L: jnp.ndarray, 
                               b: jnp.ndarray) -> jnp.ndarray:
        """
        2D ERIC model with excitability dynamics (JAX version).

        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N) where B is batch size
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        L : jnp.ndarray
            Nonlinearity parameters of shape (B,) or (B, 1, 1)
        b : jnp.ndarray
            Excitability parameters of shape (B,) or (B, 1, 1)
        
        Returns
        -------
        jnp.ndarray
            Time derivative dθ/dt of phases, shape (B, N, N)
        """
        phi_up, phi_down, phi_left, phi_right = \
            ERICExcitableSimulator._get_neighbors_jax(phi)
        
        # Ensure proper broadcasting shapes (B, 1, 1)
        if K.ndim == 1:
            K = K[:, None, None]
        if L.ndim == 1:
            L = L[:, None, None]
        if b.ndim == 1:
            b = b[:, None, None]
        
        # Initialize coupling term
        coupling = jnp.zeros_like(phi)
        
        # Vertical neighbors (up and down)
        coupling += (jnp.sin(phi_up - phi) + jnp.sin(phi_down - phi) + 
                    L * (jnp.sin(phi_up - phi)**2) + L * (jnp.sin(phi_down - phi)**2))
        
        # Horizontal neighbors (left and right)
        coupling += (jnp.sin(phi_left - phi) + jnp.sin(phi_right - phi) + 
                    L * (jnp.sin(phi_left - phi)**2) + L * (jnp.sin(phi_right - phi)**2))
        
        # Complete dynamics: natural frequency - excitability + coupling
        dphi_dt = omega - b * jnp.sin(phi) + K * coupling
        
        return dphi_dt
    
    @staticmethod
    @partial(jit, static_argnames=['dynamics_func'])
    def _rk4_step_jax(phi: jnp.ndarray, dynamics_func: Callable, 
                      dt: float, *args) -> jnp.ndarray:
        """
        4th order Runge-Kutta integration step (JAX version).
        
        Implements the classical RK4 method for numerical integration:
        k1 = f(y)
        k2 = f(y + dt/2 * k1)
        k3 = f(y + dt/2 * k2)
        k4 = f(y + dt * k3)
        y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        Parameters
        ----------
        phi : jnp.ndarray
            Current phase array of shape (B, N, N)
        dynamics_func : Callable
            Dynamics function (must be JAX-compatible and static)
        dt : float
            Time step size
        *args : tuple
            Additional arguments for dynamics_func (omega, K, L, b)
        
        Returns
        -------
        jnp.ndarray
            Updated phase array, normalized to [-π, π]
        """
        k1 = dynamics_func(phi, *args)
        k2 = dynamics_func(phi + 0.5 * dt * k1, *args)
        k3 = dynamics_func(phi + 0.5 * dt * k2, *args)
        k4 = dynamics_func(phi + dt * k3, *args)
        
        phi_new = phi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Normalize phases to [-π, π] using complex exponential
        phi_new = jnp.angle(jnp.exp(1j * phi_new))
        
        return phi_new
    
    # ==================== Simulation Methods ====================
    
    def simulate_batched_bL(self, phi_initial_batch: np.ndarray, 
                           omega_batch: np.ndarray, 
                           a_value: float,
                           b_array: np.ndarray, 
                           L_array: np.ndarray) -> List[np.ndarray]:
        """
        Simulate all (b, L) parameter combinations simultaneously on GPU for fixed a.
        
        This method exploits GPU parallelism to simulate an entire (b, L) parameter
        grid in a single pass, providing massive speedup over sequential approaches.
        Each element of the batch represents one (b, L) combination.
        
        Parameters
        ----------
        phi_initial_batch : np.ndarray
            Initial phases of shape (B, N, N) where B = len(b_array) * len(L_array)
        omega_batch : np.ndarray
            Natural frequencies of shape (B, N, N)
        a_value : float
            Fixed coupling strength parameter (K = a * (ω_max - ω_min))
        b_array : np.ndarray
            Excitability parameters of shape (B,)
        L_array : np.ndarray
            Nonlinearity parameters of shape (B,)
        
        Returns
        -------
        List[np.ndarray]
            List of frames (each of shape (B, N, N)) at designated timepoints
        """
        print(f"  Starting batched GPU simulation for a={a_value:.2f}")
        print(f"  Parameter combinations: {phi_initial_batch.shape[0]}")
        start_gpu = time.time()
        
        # Convert to JAX arrays and move to GPU
        phi = jnp.array(phi_initial_batch)
        omega = jnp.array(omega_batch)
        b = jnp.array(b_array)
        L = jnp.array(L_array)
        
        # Compute coupling strength K from a
        K = jnp.full_like(b, a_value * (self.w_max - self.w_min))
        
        frames = []
        max_time = max(self.times)
        num_steps = int(max_time / self.dt)
        
        time_idx = 0
        next_capture_time = self.times[time_idx]
        
        # Main simulation loop - all (b, L) combinations evolve together
        for step in range(num_steps):
            # Single RK4 step for ALL grids simultaneously
            phi = self._rk4_step_jax(phi, self._ERIC_2D_excitable_jax, 
                                    self.dt, omega, K, L, b)
            
            current_time = (step + 1) * self.dt
            
            # Capture frame if we've reached a designated timepoint
            if current_time >= next_capture_time:
                frames.append(np.array(phi))
                time_idx += 1
                if time_idx < len(self.times):
                    next_capture_time = self.times[time_idx]
                
                # Progress indicator
                if time_idx % 5 == 0 or time_idx == len(self.times):
                    elapsed = time.time() - start_gpu
                    print(f"    Captured frame {time_idx}/{len(self.times)} "
                          f"at t={current_time:.1f}, elapsed: {elapsed:.1f}s")
        
        end_gpu = time.time()
        print(f"  GPU simulation completed in {end_gpu - start_gpu:.2f} seconds")
        
        return frames
    
    def get_frames_for_a_value(self, a_value: float, 
                              phase_condition: str) -> List[List[np.ndarray]]:
        """
        Get all frames for a specific a value across (b, L) parameter space.
        
        This method generates the complete spatiotemporal dynamics for all (b, L)
        combinations at a fixed coupling strength a, using batched GPU computation.
        
        Parameters
        ----------
        a_value : float
            Fixed coupling strength parameter
        phase_condition : str
            Name of phase condition to use for initial conditions
        
        Returns
        -------
        List[List[np.ndarray]]
            Nested list structure: [param_combo][timepoint] -> phase grid (N, N)
        """
        if phase_condition not in self.phase_conditions:
            raise ValueError(f"Unknown phase condition: {phase_condition}. "
                           f"Available: {list(self.phase_conditions.keys())}")
        
        condition_data = self.phase_conditions[phase_condition]
        
        # Generate parameter values
        b_values = np.linspace(*self.config.b_range)
        L_values = np.linspace(*self.config.L_range)
        
        # CRITICAL: Reset seed right before generating initial conditions
        np.random.seed(12345)
        
        # Generate initial conditions (same for all parameter combinations)
        omega_initial = self.omega_uniform()
        phase_func = self.get_phase_function(condition_data.phase_func)
        phi_initial = phase_func(self.N)
        
        # Create batched arrays for all (b, L) parameter combinations
        n_params = len(b_values) * len(L_values)
        phi_batch = np.tile(phi_initial, (n_params, 1, 1))  # (B, N, N)
        omega_batch = np.tile(omega_initial, (n_params, 1, 1))  # (B, N, N)
        
        # Build parameter arrays
        b_array = []
        L_array = []
        
        for b in b_values:
            for L in L_values:
                b_array.append(b)
                L_array.append(L)
        
        b_array = np.array(b_array)  # (B,)
        L_array = np.array(L_array)  # (B,)
        
        print(f"  Batch size: {n_params} parameter combinations")
        print(f"  Grid size: {self.N}x{self.N}")
        print(f"  Total cells: {n_params * self.N * self.N:,}")
        
        # Run single batched simulation on GPU
        frames = self.simulate_batched_bL(phi_batch, omega_batch, 
                                         a_value, b_array, L_array)
        
        # Reshape frames for visualization
        # frames: [frame0, frame1, ...] where each frame is (B, N, N)
        # Convert to: [[param0_frames], [param1_frames], ...]
        frames_reshaped = []
        for i in range(n_params):
            param_frames = [frame[i] for frame in frames]
            frames_reshaped.append(param_frames)
        
        return frames_reshaped
    
    def create_animation_for_a(self, a_value: float, phase_condition: str,
                              save_video: bool = True, 
                              output_dir: str = 'output') -> animation.FuncAnimation:
        """
        Create animation for specific a value showing (b, L) parameter space.
        
        Generates a grid animation where:
        - Rows represent different b values (excitability)
        - Columns represent different L values (nonlinearity)
        - Each subplot shows the phase dynamics for one (b, L) combination
        
        Parameters
        ----------
        a_value : float
            Fixed coupling strength parameter
        phase_condition : str
            Name of phase condition
        save_video : bool
            Whether to save as MP4 video
        output_dir : str
            Directory to save output
        
        Returns
        -------
        animation.FuncAnimation
            The created animation object
        """
        print(f"\n{'='*70}")
        print(f"Processing a={a_value:.2f} with {phase_condition}")
        print(f"{'='*70}")
        start_time = time.time()
        
        condition_data = self.phase_conditions[phase_condition]
        frames_data = self.get_frames_for_a_value(a_value, phase_condition)
        
        # Generate parameter values for labels
        b_values = np.linspace(*self.config.b_range)
        L_values = np.linspace(*self.config.L_range)
        
        n_rows = len(b_values)
        n_cols = len(L_values)
        
        # Setup figure with appropriate size
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(n_rows + 1, n_cols + 1, figure=fig, 
                              wspace=0.05, hspace=0.05,
                              #top=0.94, bottom=0.04, left=0.08, right=0.98
                              )
        
        # Create grid of subplots
        plots = []
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i, j])
                plot = ax.imshow(np.zeros((self.N, self.N)), 
                               cmap='twilight', vmin=-np.pi, vmax=np.pi)
                ax.axis('off')
                plots.append(plot)
        
        # Add parameter labels
        # Labels for 'b' on the left
        for i, b_val in enumerate(b_values):
            ax_b = fig.add_subplot(gs[i, 0], frameon=False)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            ax_b.text(-0.6, 0.5, f'b={b_val:.3f}', va='center', ha='center', 
                     fontsize=10, transform=ax_b.transAxes, rotation=0)
        
        # Labels for 'L' on the bottom
        for j, L_val in enumerate(L_values):
            ax_L = fig.add_subplot(gs[n_rows, j], frameon=False)
            ax_L.set_xticks([])
            ax_L.set_yticks([])
            ax_L.text(0.5, 0.6, fr'$\Lambda$={L_val:.2f}', va='center', ha='center', 
                     fontsize=10, transform=ax_L.transAxes)
        
        def update(frame_idx):
            """Update function for animation"""
            for i, plot in enumerate(plots):
                plot.set_data(frames_data[i][frame_idx])
            
            # Update title
            phase_desc = condition_data.phase_desc
            omega_desc = (fr'$\omega = U\left(\frac{{2\pi}}{{180}}, '
                         fr'\frac{{2\pi}}{{150}}\right)$ min$^{{-1}}$')
            
            fig.suptitle(f'ERIC_2D_exe: Phases at t={self.times[frame_idx]} min\n'
                        f'a={a_value:.2f}, {phase_desc}, {omega_desc}',
                        fontsize=14, y=0.98)
            
            return plots
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.times), 
                                    interval=1000, blit=False)
        
        if save_video:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, 
                                   f'ERIC_exe_bL_a{a_value:.2f}_{phase_condition}.mp4')
            print(f"Saving animation to {filename}...")
            ani.save(filename, writer='ffmpeg', fps=1, dpi=150)
            print(f"Animation saved as {filename}")
        
        end_time = time.time()
        print(f'Total time for a={a_value:.2f}: {end_time - start_time:.2f} seconds')
        
        plt.close()
        return ani
    
    def run_all_a_values(self, phase_condition: str = 'randomp',
                        save_videos: bool = True, output_dir: str = 'output'):
        """
        Run simulations for all a values.
        
        Generates a separate animation file for each coupling strength value,
        showing the full (b, L) parameter space.
        
        Parameters
        ----------
        phase_condition : str
            Name of phase condition to use
        save_videos : bool
            Whether to save animations as MP4 videos
        output_dir : str
            Directory to save outputs
        """
        print(f"\n{'#'*70}")
        print(f"Running simulations for all a values")
        print(f"Phase condition: {phase_condition}")
        print(f"a values: {self.config.a_values}")
        print(f"{'#'*70}\n")
        
        for a_value in self.config.a_values:
            self.create_animation_for_a(a_value, phase_condition, 
                                       save_videos, output_dir)
    
    def list_available_conditions(self):
        """Print available phase conditions"""
        print("\nAvailable Phase Conditions:")
        print("-" * 70)
        for name, config in self.phase_conditions.items():
            print(f"{name:20s} - {config.phase_desc}")
        print("-" * 70)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ERIC Excitable Model Simulator (JAX GPU-Accelerated)\n'
                    'Simulates ERIC_2D_exe model in (b, L) parameter space for fixed a values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default config file
  python phasegrids_gpu_abl.py --generate-config
  
  # Run specific a value with randomp condition
  python phasegrids_gpu_abl.py --a-value 0.8 --condition randomp
  
  # Run all a values
  python phasegrids_gpu_abl.py --all-a --condition randomp
  
  # List available conditions
  python phasegrids_gpu_abl.py --list-conditions
        """
    )
    
    parser.add_argument('--config', type=str, default='simulation_config_bL.json',
                       help='Path to configuration file (default: simulation_config_bL.json)')
    
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate default configuration file and exit')
    
    parser.add_argument('--a-value', type=float,
                       help='Specific a value to simulate')
    
    parser.add_argument('--all-a', action='store_true',
                       help='Run all a values from config')
    
    parser.add_argument('--condition', type=str, default='randomp',
                       help='Phase condition to use (default: randomp)')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save animation as video')
    
    parser.add_argument('--output-dir', type=str, default='bL_outputs',
                       help='Directory to save output files (default: bL_outputs)')
    
    parser.add_argument('--list-conditions', action='store_true',
                       help='List available phase conditions and exit')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Generate config if requested
    if args.generate_config:
        ConfigManager.save_default_config(args.config)
        return
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found.")
        print("Generating default configuration...")
        config_dict = ConfigManager.save_default_config(args.config)
    else:
        config_dict = ConfigManager.load_config(args.config)
    
    # Create simulation config
    sim_config = SimulationConfig(**config_dict['simulation'])
    
    # Initialize simulator
    simulator = ERICExcitableSimulator(config=sim_config)
    
    # Load phase conditions
    simulator.load_phase_conditions_from_config(config_dict['phase_conditions'])
    
    # List conditions if requested
    if args.list_conditions:
        simulator.list_available_conditions()
        return
    
    # Validate phase condition
    if args.condition not in simulator.phase_conditions:
        print(f"Error: Unknown phase condition '{args.condition}'")
        print(f"Available conditions: {list(simulator.phase_conditions.keys())}")
        return
    
    # Run simulations
    save_videos = not args.no_save
    
    if args.all_a:
        print(f"\nRunning all a values with condition: {args.condition}")
        simulator.run_all_a_values(args.condition, save_videos, args.output_dir)
    
    elif args.a_value is not None:
        print(f"\nRunning a={args.a_value} with condition: {args.condition}")
        simulator.create_animation_for_a(args.a_value, args.condition, 
                                        save_videos, args.output_dir)
    
    else:
        print("Error: Must specify either --a-value or --all-a")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()