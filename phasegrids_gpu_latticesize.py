# Code for generating .mp4 movies showing phase evolution for different lattice sizes for different phase oscillator model described in the paper: "Foci, waves, excitability: self-organization of phase waves in a model of asymmetrically coupled embryonic oscillators". In the paper, we use only the 2D ERIC model to generate the plots. Each frame in the movie is a 10x10 grid showing the phasemap in the (K,\Lambda) parameter space at a certain timepoint. The parameter ranges can be changed by the user depending on what they want to study. The code is highly modular and additional phase models can be easily incorporated.
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

# Set legacy seed for reproducibility (NumPy only, JAX uses different approach)
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

from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from functools import partial

# Configure JAX to use GPU
jax.config.update('jax_platform_name', 'gpu')

@dataclass
class ModelMetadata:
    """Metadata describing model parameter requirements"""
    name: str
    requires_L: bool
    requires_b: bool
    description: str


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    N: int = 50  # Default single lattice size
    N_values: Optional[List[int]] = None  # For lattice size studies
    dt: float = 0.01
    w_min: float = 2 * np.pi / 180
    w_max: float = 2 * np.pi / 150
    b: float = 0.01
    times: List[int] = None
    
    def __post_init__(self):
        if self.times is None:
            self.times = [100, 500, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 12000, 15000]
        
        # If N_values not specified but N is, use single N
        if self.N_values is None:
            self.N_values = [self.N]
    
    def get_active_n_values(self) -> List[int]:
        """Get the list of N values to simulate"""
        return self.N_values if self.N_values else [self.N]


@dataclass
class PhaseConditionConfig:
    """Configuration for a phase condition"""
    phase_func: str
    phase_desc: str
    a_range: tuple = (0.5, 2.0, 10)  # (start, stop, num)
    L_range: tuple = (1.2, 1.6, 10)  # (start, stop, num)


class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_default_config(config_path: str = 'simulation_config_latticesize.json'):
        """Save default configuration to JSON file"""
        default_config = {
            "simulation": {
                "N": 50,
                "N_values": None,  # Set to [20, 50, 100] for lattice size studies
                "dt": 0.01,
                "w_min": 0.03490658503988659,  # 2*pi/180
                "w_max": 0.04188790204786391,  # 2*pi/150
                "b": 0.01,
                "times": [100, 500, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 12000, 15000]
            },
            "phase_conditions": {
                "narrowp1": {
                    "phase_func": "narrowp1_phase",
                    "phase_desc": r"$\theta_{in}=U(0, \pi/2)$",
                    "a_range": [0.4, 1.2, 10],                   
                    "L_range": [1.0, 2.0, 10],
                    #"a_range": [0.0, 2.0, 10],
                    #"L_range": [0.0, 2.5, 10],                    
                },
                "widerp3": {
                    "phase_func": "widerp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/4, \pi)$",
                    "a_range": [0.4, 1.2, 10],                   
                    "L_range": [1.0, 2.0, 10],
                    #"a_range": [0.0, 2.0, 10],
                    #"L_range": [0.0, 2.5, 10]
                },
                "narrowp3": {
                    "phase_func": "narrowp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/2, \pi/2)$",
                    "a_range": [0.8, 1.2, 10],                    
                    "L_range": [1.6, 2.0, 10],
                    #"a_range": [0.0, 2.0, 10],
                    #"L_range": [0.0, 2.5, 10]
                },
                "randomp": {
                    "phase_func": "randomp_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi, \pi)$",
                    "a_range": [0.0, 2.0, 10],
                    "L_range": [0.0, 2.5, 10]
                }
            },
            "models": {
                "ERIC_2D": {
                    "requires_L": True,
                    "requires_b": False,
                    "description": "2D ERIC model with Lambda parameter"
                },
                "ERIC_2D_exe": {
                    "requires_L": True,
                    "requires_b": True,
                    "description": "2D ERIC model with excitability"
                },
                "kuramoto_2D": {
                    "requires_L": False,
                    "requires_b": False,
                    "description": "Standard Kuramoto model"
                },
                "ReKU_2D": {
                    "requires_L": False,
                    "requires_b": False,
                    "description": "Rectified Kuramoto model"
                },
                "QIF_2D": {
                    "requires_L": True,
                    "requires_b": False,
                    "description": "Quadratic Integrate and Fire model"
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Default configuration saved to {config_path}")
        return default_config


class JAXPhaseDynamicsSimulator:
    """
    GPU-optimized simulator using JAX batched operations with lattice size variation support.
    
    This simulator can run studies across different lattice sizes (N) in coupled phase oscillator systems.
    
    Key Features
    ------------
    - Batched GPU simulation of all parameter combinations simultaneously
    - Support for multiple lattice sizes (N) in a single run
    - Multiple phase oscillator models (ERIC, Kuramoto, ReKU, QIF)
    - Configurable via JSON files and command-line arguments
    - Automatic video generation for each configuration
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize JAX-based simulator.
        
        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation configuration. If None, uses defaults.
        """
        self.config = config if config is not None else SimulationConfig()
        
        # Store configuration parameters
        self.N = self.config.N  # Default N, can be overridden
        self.dt = self.config.dt
        self.w_min = self.config.w_min
        self.w_max = self.config.w_max
        self.b = self.config.b
        self.times = self.config.times
        
        # Model metadata registry
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self._register_default_models()
        
        # Phase condition configurations
        self.phase_conditions: Dict[str, PhaseConditionConfig] = {}
        
        # Dictionary to store model dynamics functions (JAX versions)
        self.models: Dict[str, Callable] = {
            'ERIC_2D': self._ERIC_2D_dynamics_jax,
            'ERIC_2D_exe': self._ERIC_2D_excitable_jax,
            'kuramoto_2D': self._kuramoto_model_jax,
            'ReKU_2D': self._ReKU_model_jax,
            'QIF_2D': self._QIF_model_jax,
        }
        
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
    
    def _register_default_models(self):
        """Register metadata for default models"""
        self.model_metadata['ERIC_2D'] = ModelMetadata(
            name='ERIC_2D',
            requires_L=True,
            requires_b=False,
            description='2D ERIC model with Lambda parameter'
        )
        self.model_metadata['ERIC_2D_exe'] = ModelMetadata(
            name='ERIC_2D_exe',
            requires_L=True,
            requires_b=True,
            description='2D ERIC model with excitability'
        )
        self.model_metadata['kuramoto_2D'] = ModelMetadata(
            name='kuramoto_2D',
            requires_L=False,
            requires_b=False,
            description='Standard Kuramoto model'
        )
        self.model_metadata['ReKU_2D'] = ModelMetadata(
            name='ReKU_2D',
            requires_L=False,
            requires_b=False,
            description='Rectified Kuramoto model'
        )
        self.model_metadata['QIF_2D'] = ModelMetadata(
            name='QIF_2D',
            requires_L=True,
            requires_b=False,
            description='Quadratic Integrate and Fire model'
        )
    
    def load_phase_conditions_from_config(self, config_dict: Dict[str, Any]):
        """Load phase conditions from configuration dictionary"""
        for name, params in config_dict.items():
            # Convert lists to tuples for a_range and L_range
            if 'a_range' in params and isinstance(params['a_range'], list):
                params['a_range'] = tuple(params['a_range'])
            if 'L_range' in params and isinstance(params['L_range'], list):
                params['L_range'] = tuple(params['L_range'])
            self.phase_conditions[name] = PhaseConditionConfig(**params)
    
    # Phase initialization functions (NumPy - used for initial conditions)
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
    
    def omega_uniform(self, N: int) -> np.ndarray:
        """Generate uniform omega distribution for lattice size N"""
        return np.random.uniform(self.w_min, self.w_max, size=(N, N))
    
    # ========== JAX Static Methods (JIT-compiled) ==========
    
    @staticmethod
    @jit
    def _get_neighbors_jax(phi: jnp.ndarray) -> tuple:
        """
        Get neighboring phases with open boundary conditions.
        Works for both single (N, N) and batched (B, N, N) arrays.
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (N, N) or (B, N, N)
        
        Returns
        -------
        tuple
            (phi_up, phi_down, phi_left, phi_right)
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
    def _ERIC_2D_dynamics_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                              K: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
        """
        2D ERIC model dynamics (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N)
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        L : jnp.ndarray
            Lambda parameters of shape (B,) or (B, 1, 1)
        
        Returns
        -------
        jnp.ndarray
            Time derivative of phases
        """
        phi_up, phi_down, phi_left, phi_right = \
            JAXPhaseDynamicsSimulator._get_neighbors_jax(phi)
        
        # Ensure K and L have shape (B, 1, 1) for broadcasting
        if K.ndim == 1:
            K = K[:, None, None]
        if L.ndim == 1:
            L = L[:, None, None]
        
        dphi_dt = jnp.zeros_like(phi)
        
        # Vertical neighbors
        dphi_dt += (jnp.sin(phi_up - phi) + jnp.sin(phi_down - phi) + 
                    L * (jnp.sin(phi_up - phi)**2) + L * (jnp.sin(phi_down - phi)**2))
        
        # Horizontal neighbors
        dphi_dt += (jnp.sin(phi_left - phi) + jnp.sin(phi_right - phi) + 
                    L * (jnp.sin(phi_left - phi)**2) + L * (jnp.sin(phi_right - phi)**2))
        
        dphi_dt = omega + K * dphi_dt
        
        return dphi_dt
    
    @staticmethod
    @jit
    def _ERIC_2D_excitable_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                               K: jnp.ndarray, L: jnp.ndarray, 
                               b: jnp.ndarray) -> jnp.ndarray:
        """
        2D ERIC model with excitability (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N)
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        L : jnp.ndarray
            Lambda parameters of shape (B,) or (B, 1, 1)
        b : jnp.ndarray
            Excitability parameters of shape (B,) or (B, 1, 1) or scalar
        
        Returns
        -------
        jnp.ndarray
            Time derivative of phases
        """
        phi_up, phi_down, phi_left, phi_right = \
            JAXPhaseDynamicsSimulator._get_neighbors_jax(phi)
        
        # Ensure proper broadcasting shapes
        if K.ndim == 1:
            K = K[:, None, None]
        if L.ndim == 1:
            L = L[:, None, None]
        if b.ndim == 0:  # scalar
            pass  # Broadcasting works automatically
        elif b.ndim == 1:
            b = b[:, None, None]
        
        dphi_dt = jnp.zeros_like(phi)
        
        # Vertical neighbors
        dphi_dt += (jnp.sin(phi_up - phi) + jnp.sin(phi_down - phi) + 
                    L * (jnp.sin(phi_up - phi)**2) + L * (jnp.sin(phi_down - phi)**2))
        
        # Horizontal neighbors
        dphi_dt += (jnp.sin(phi_left - phi) + jnp.sin(phi_right - phi) + 
                    L * (jnp.sin(phi_left - phi)**2) + L * (jnp.sin(phi_right - phi)**2))
        
        dphi_dt = omega - b * jnp.sin(phi) + K * dphi_dt
        
        return dphi_dt
    
    @staticmethod
    @jit
    def _kuramoto_model_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                            K: jnp.ndarray) -> jnp.ndarray:
        """
        Kuramoto model dynamics (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N)
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        
        Returns
        -------
        jnp.ndarray
            Time derivative of phases
        """
        phi_up, phi_down, phi_left, phi_right = \
            JAXPhaseDynamicsSimulator._get_neighbors_jax(phi)
        
        if K.ndim == 1:
            K = K[:, None, None]
        
        dphi_dt = jnp.zeros_like(phi)
        dphi_dt += jnp.sin(phi_up - phi) + jnp.sin(phi_down - phi)
        dphi_dt += jnp.sin(phi_left - phi) + jnp.sin(phi_right - phi)
        dphi_dt = omega + K * dphi_dt
        
        return dphi_dt
    
    @staticmethod
    @jit
    def _ReKU_model_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                        K: jnp.ndarray) -> jnp.ndarray:
        """
        Rectified Kuramoto (ReKU) model dynamics (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N)
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        
        Returns
        -------
        jnp.ndarray
            Time derivative of phases
        """
        phi_up, phi_down, phi_left, phi_right = \
            JAXPhaseDynamicsSimulator._get_neighbors_jax(phi)
        
        if K.ndim == 1:
            K = K[:, None, None]
        
        # Rectified sine: max(sin(x), 0)
        dphi_dt = jnp.zeros_like(phi)
        dphi_dt += (jnp.maximum(jnp.sin(phi_up - phi), 0) + 
                    jnp.maximum(jnp.sin(phi_down - phi), 0))
        dphi_dt += (jnp.maximum(jnp.sin(phi_left - phi), 0) + 
                    jnp.maximum(jnp.sin(phi_right - phi), 0))
        dphi_dt = omega + K * dphi_dt
        
        return dphi_dt
    
    @staticmethod
    @jit
    def _QIF_model_jax(phi: jnp.ndarray, omega: jnp.ndarray, 
                       K: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
        """
        Quadratic Integrate and Fire (QIF) model dynamics (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Phase array of shape (B, N, N)
        omega : jnp.ndarray
            Natural frequencies of shape (B, N, N)
        K : jnp.ndarray
            Coupling strengths of shape (B,) or (B, 1, 1)
        L : jnp.ndarray
            Lambda parameters of shape (B,) or (B, 1, 1)
        
        Returns
        -------
        jnp.ndarray
            Time derivative of phases
        """
        phi_up, phi_down, phi_left, phi_right = \
            JAXPhaseDynamicsSimulator._get_neighbors_jax(phi)
        
        if K.ndim == 1:
            K = K[:, None, None]
        if L.ndim == 1:
            L = L[:, None, None]
        
        dphi_dt = jnp.zeros_like(phi)
        
        # QIF uses sin((phi_j - phi_i)/2)^2 instead of sin(phi_j - phi_i)^2
        dphi_dt += (jnp.sin(phi_up - phi) + jnp.sin(phi_down - phi) + 
                    2 * L * (jnp.sin((phi_up - phi) / 2)**2) + 
                    2 * L * (jnp.sin((phi_down - phi) / 2)**2))
        dphi_dt += (jnp.sin(phi_left - phi) + jnp.sin(phi_right - phi) + 
                    2 * L * (jnp.sin((phi_left - phi) / 2)**2) + 
                    2 * L * (jnp.sin((phi_right - phi) / 2)**2))
        dphi_dt = omega + K * dphi_dt
        
        return dphi_dt

    @staticmethod
    @partial(jit, static_argnames=['dynamics_func'])
    def _rk4_step_jax(phi: jnp.ndarray, dynamics_func: Callable, 
                      dt: float, *args) -> jnp.ndarray:
        """
        4th order Runge-Kutta integration step (JAX version).
        
        Parameters
        ----------
        phi : jnp.ndarray
            Current phase array of shape (B, N, N)
        dynamics_func : Callable
            Dynamics function (must be JAX-compatible) - STATIC
        dt : float
            Time step
        *args : 
            Additional arguments for dynamics_func (omega, K, L, b, etc.)
        
        Returns
        -------
        jnp.ndarray
            Updated phase array
        """
        k1 = dynamics_func(phi, *args)
        k2 = dynamics_func(phi + 0.5 * dt * k1, *args)
        k3 = dynamics_func(phi + 0.5 * dt * k2, *args)
        k4 = dynamics_func(phi + dt * k3, *args)
        
        phi_new = phi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Normalize phases to [-π, π]
        phi_new = jnp.angle(jnp.exp(1j * phi_new))
        
        return phi_new

    
    def _build_dynamics_args(self, model_name: str, omega: jnp.ndarray, 
                            K: jnp.ndarray, L: jnp.ndarray) -> tuple:
        """
        Build arguments for dynamics function based on model requirements.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        omega : jnp.ndarray
            Natural frequencies
        K : jnp.ndarray
            Coupling strengths
        L : jnp.ndarray
            Lambda parameters
        
        Returns
        -------
        tuple
            Arguments to pass to dynamics function
        """
        metadata = self.model_metadata.get(model_name)
        if metadata is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name == 'ERIC_2D_exe':
            # Requires omega, K, L, b
            b_jax = jnp.array(self.b)
            return (omega, K, L, b_jax)
        elif metadata.requires_L:
            # Requires omega, K, L
            return (omega, K, L)
        else:
            # Requires only omega, K
            return (omega, K)
    
    def simulate_batched(self, phi_initial_batch: np.ndarray, 
                        omega_batch: np.ndarray, 
                        K_array: np.ndarray, 
                        L_array: np.ndarray,
                        model_name: str = 'ERIC_2D',
                        N: Optional[int] = None) -> List[np.ndarray]:
        """
        Simulate all parameter combinations simultaneously on GPU.
        
        This is the core batched simulation that runs ALL parameter combinations
        in parallel, providing massive speedup over sequential/multiprocessing approaches.
        
        Parameters
        ----------
        phi_initial_batch : np.ndarray
            Initial phases of shape (B, N, N) where B = number of parameter combinations
        omega_batch : np.ndarray
            Natural frequencies of shape (B, N, N)
        K_array : np.ndarray
            Coupling strengths of shape (B,)
        L_array : np.ndarray
            Lambda values of shape (B,)
        model_name : str
            Name of the model to simulate
        N : int, optional
            Lattice size (for display purposes)
        
        Returns
        -------
        list of np.ndarray
            List of frames (each of shape (B, N, N)) at designated timepoints
        """
        N_display = N if N is not None else phi_initial_batch.shape[1]
        print(f"  Starting batched GPU simulation with {phi_initial_batch.shape[0]} parameter combinations (N={N_display})...")
        start_gpu = time.time()
        
        # Convert to JAX arrays and move to GPU
        phi = jnp.array(phi_initial_batch)
        omega = jnp.array(omega_batch)
        K = jnp.array(K_array)
        L = jnp.array(L_array)
        
        # Get the dynamics function
        dynamics_func = self.models[model_name]
        
        # Build arguments based on model requirements
        dynamics_args = self._build_dynamics_args(model_name, omega, K, L)
        
        frames = []
        max_time = max(self.times)
        num_steps = int(max_time / self.dt)
        
        time_idx = 0
        next_capture_time = self.times[time_idx]
        
        # Main simulation loop - all parameter combinations evolve together
        for step in range(num_steps):
            # Single RK4 step for ALL grids simultaneously
            phi = self._rk4_step_jax(phi, dynamics_func, self.dt, *dynamics_args)
            
            current_time = (step + 1) * self.dt
            
            # Capture frame if we've reached a designated timepoint
            if current_time >= next_capture_time:
                # Convert to numpy and store
                frames.append(np.array(phi))
                time_idx += 1
                if time_idx < len(self.times):
                    next_capture_time = self.times[time_idx]
                
                # Progress indicator
                if time_idx % 5 == 0 or time_idx == len(self.times):
                    elapsed = time.time() - start_gpu
                    print(f"    Captured frame {time_idx}/{len(self.times)} at t={current_time:.1f}, elapsed: {elapsed:.1f}s")
        
        end_gpu = time.time()
        print(f"  GPU simulation completed in {end_gpu - start_gpu:.2f} seconds")
        
        return frames
    
    def get_frames_for_condition(self, phase_condition: str, 
                                 model_name: str = 'ERIC_2D',
                                 N: Optional[int] = None) -> List[List[np.ndarray]]:
        """
        Get all frames for a specific phase condition using batched GPU simulation.
        
        Parameters
        ----------
        phase_condition : str
            Name of the phase condition
        model_name : str
            Name of the model to use
        N : int, optional
            Lattice size. If None, uses self.N
        
        Returns
        -------
        List[List[np.ndarray]]
            List of parameter combinations, each containing list of frames
        """
        if phase_condition not in self.phase_conditions:
            raise ValueError(f"Unknown phase condition: {phase_condition}. "
                           f"Available: {list(self.phase_conditions.keys())}")
        
        # Use provided N or default
        lattice_size = N if N is not None else self.N
        
        condition_data = self.phase_conditions[phase_condition]
        metadata = self.model_metadata.get(model_name)
        
        # Determine parameter grid based on model requirements
        if metadata and not metadata.requires_L:
            # For models without L, use 100 a_values in 10x10 grid
            a_values = np.linspace(0, 4, 100)
            L_values = [0.0]  # Dummy value, won't be used
        else:
            # Generate parameter values using np.linspace
            a_values = np.linspace(*condition_data.a_range)  # Unpack tuple: (start, stop, num)
            L_values = np.linspace(*condition_data.L_range)

            
        # CRITICAL: Reset seed right before generating initial conditions
        np.random.seed(12345)
            
        # Generate initial conditions (same for all parameter combinations)
        omega_initial = self.omega_uniform(lattice_size)
        phase_func = self.get_phase_function(condition_data.phase_func)
        phi_initial = phase_func(lattice_size)
        
        # Create batched arrays for all parameter combinations
        n_params = len(a_values) * len(L_values)
        phi_batch = np.tile(phi_initial, (n_params, 1, 1))  # (B, N, N)
        omega_batch = np.tile(omega_initial, (n_params, 1, 1))  # (B, N, N)
        
        # Build parameter arrays
        K_array = []
        L_array_full = []
        
        for a in a_values:
            for L in L_values:
                K = a * (self.w_max - self.w_min)
                K_array.append(K)
                L_array_full.append(L)
        
        K_array = np.array(K_array)  # (B,)
        L_array_full = np.array(L_array_full)  # (B,)
        
        print(f"  Batch size: {n_params} parameter combinations")
        print(f"  Grid size: {lattice_size}x{lattice_size}")
        print(f"  Total cells: {n_params * lattice_size * lattice_size:,}")
        
        # Run single batched simulation on GPU
        frames = self.simulate_batched(phi_batch, omega_batch, K_array, 
                                       L_array_full, model_name, N=lattice_size)
        
        # Reshape frames for compatibility with visualization code
        # frames is currently: [frame0, frame1, ...] where each frame is (B, N, N)
        # Need to convert to: [[param0_frames], [param1_frames], ...]
        frames_reshaped = []
        for i in range(n_params):
            param_frames = [frame[i] for frame in frames]  # Extract all timepoints for this parameter
            frames_reshaped.append(param_frames)
        
        return frames_reshaped
    
    def create_animation(self, phase_condition: str, model_name: str = 'ERIC_2D', 
                        save_video: bool = True, output_dir: str = 'output',
                        N: Optional[int] = None) -> animation.FuncAnimation:
        """
        Create animation for a specific phase condition and model.
        
        Parameters
        ----------
        phase_condition : str
            Name of phase condition
        model_name : str
            Name of model to use
        save_video : bool
            Whether to save as video
        output_dir : str
            Directory to save output
        N : int, optional
            Lattice size. If None, uses self.N
        
        Returns
        -------
        animation.FuncAnimation
            The created animation
        """
        lattice_size = N if N is not None else self.N
        
        print(f"\nProcessing {phase_condition} with {model_name} (N={lattice_size})...")
        start_time = time.time()
        
        condition_data = self.phase_conditions[phase_condition]
        frames_data = self.get_frames_for_condition(phase_condition, model_name, N=lattice_size)
        
        metadata = self.model_metadata.get(model_name)
        requires_L = metadata.requires_L if metadata else True

        # Determine grid layout
        if not requires_L:
            a_values = np.linspace(0, 4, 100)
            n_rows = 10
            n_cols = 10

        else:
            a_values = np.linspace(*condition_data.a_range)
            L_values = np.linspace(*condition_data.L_range)
            n_rows = len(a_values)
            n_cols = len(L_values)
        
        # Setup figure
        if not requires_L:
            fig = plt.figure(figsize=(12, 16))
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, 
                                  wspace=0.05, hspace=0.10,
                                  top=0.94, bottom=0.02, left=0.02, right=0.98)
        else:
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(n_rows + 1, n_cols + 1, figure=fig, 
                                  wspace=0.05, hspace=0.05)
        
        # Set up the grid of subplots
        plots = []
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i, j])
                plot = ax.imshow(np.zeros((lattice_size, lattice_size)), 
                               cmap='twilight', vmin=-np.pi, vmax=np.pi)
                ax.axis('off')
                
                # For models without L, add a_value label above each subplot
                if not requires_L:
                    a_values_grid = a_values.reshape(10, 10)
                    ax.text(0.5, 1.05, f'a={a_values_grid[i, j]:.2f}', 
                           transform=ax.transAxes,
                           ha='center', va='bottom', fontsize=7)
                
                plots.append(plot)
        
        # Add parameter labels for L-dependent models
        if requires_L:
            L_values = np.linspace(*condition_data.L_range)
            # Labels for 'a' on the left
            for i, a_val in enumerate(a_values):
                ax_a = fig.add_subplot(gs[i, 0], frameon=False)
                ax_a.set_xticks([])
                ax_a.set_yticks([])
                ax_a.text(-0.6, 0.5, f'a={a_val:.2f}', va='center', ha='center', 
                         fontsize=10, transform=ax_a.transAxes)
            
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
            
            fig.suptitle(f'Phases at t={self.times[frame_idx]} min (N={lattice_size})\n'
                        f'{phase_desc}, {omega_desc}, Model: {model_name}',
                        fontsize=14, y=0.98)
            
            return plots
            
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.times), 
                                    interval=1000, blit=False)
        
        if save_video:
            os.makedirs(output_dir, exist_ok=True)
            L_suffix = 'with_L' if requires_L else 'grid10x10'
            filename = os.path.join(output_dir, 
                                   f'phasegrids_{phase_condition}_{model_name}_{L_suffix}_N{lattice_size}_jax_gpu.mp4')
            print(f"Saving animation to {filename}...")
            ani.save(filename, writer='ffmpeg', fps=1, dpi=150)
            print(f"Animation saved as {filename}")

        
        end_time = time.time()
        print(f'Total time for {phase_condition} (N={lattice_size}): {end_time - start_time:.2f} seconds')
        
        plt.close()
        return ani
    
    def run_all_conditions(self, model_name: str = 'ERIC_2D', 
                          save_videos: bool = True, output_dir: str = 'output_latticesize',
                          N: Optional[int] = None):
        """
        Run simulations for all phase conditions.
        
        Parameters
        ----------
        model_name : str
            Name of model to use
        save_videos : bool
            Whether to save animations as videos
        output_dir : str
            Directory to save outputs
        N : int, optional
            Lattice size. If None, uses self.N
        """
        for condition in self.phase_conditions.keys():
            self.create_animation(condition, model_name, save_videos, output_dir, N=N)
    
    def run_lattice_size_study(self, phase_condition: str, model_name: str = 'ERIC_2D',
                              N_values: Optional[List[int]] = None,
                              save_videos: bool = True, output_dir: str = 'output_latticesize'):
        """
        Run simulations across multiple lattice sizes for finite-size scaling analysis.
        
        
        Parameters
        ----------
        phase_condition : str
            Name of phase condition to study
        model_name : str
            Name of model to use
        N_values : List[int], optional
            List of lattice sizes to simulate. If None, uses config.N_values
        save_videos : bool
            Whether to save animations as videos
        output_dir : str
            Directory to save outputs
        
        Notes
        -----
        Each lattice size generates a separate video file with N in the filename
        for easy comparison and analysis.
        """
        # Determine which N values to use
        if N_values is None:
            N_values = self.config.get_active_n_values()
        
        print(f"\n{'='*80}")
        print(f"LATTICE SIZE STUDY")
        print(f"{'='*80}")
        print(f"Phase Condition: {phase_condition}")
        print(f"Model: {model_name}")
        print(f"Lattice Sizes (N): {N_values}")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*80}\n")
        
        # Run simulation for each lattice size
        for idx, N in enumerate(N_values, 1):
            print(f"\n{'-'*80}")
            print(f"Lattice Size {idx}/{len(N_values)}: N = {N}")
            print(f"{'-'*80}")
            
            self.create_animation(
                phase_condition=phase_condition,
                model_name=model_name,
                save_video=save_videos,
                output_dir=output_dir,
                N=N
            )
        
        print(f"\n{'='*80}")
        print(f"LATTICE SIZE STUDY COMPLETE")
        print(f"{'='*80}")
        print(f"Generated {len(N_values)} videos in: {output_dir}")
        print(f"Lattice sizes: {N_values}")
        print(f"{'='*80}\n")
    
    def run_comprehensive_study(self, model_names: Optional[List[str]] = None,
                               conditions: Optional[List[str]] = None,
                               N_values: Optional[List[int]] = None,
                               save_videos: bool = True, output_dir: str = 'output_latticesize'):
        """
        Run comprehensive study across models, conditions, and lattice sizes.
        
        This high-level method orchestrates large-scale parameter studies combining:
        - Multiple models (ERIC, Kuramoto, etc.)
        - Multiple initial conditions
        - Multiple lattice sizes
        
        Useful for systematic comparison
        
        Parameters
        ----------
        model_names : List[str], optional
            List of models to study. If None, uses ['ERIC_2D']
        conditions : List[str], optional
            List of phase conditions. If None, uses all available
        N_values : List[int], optional
            List of lattice sizes. If None, uses config.N_values
        save_videos : bool
            Whether to save animations
        output_dir : str
            Directory to save outputs
        """
        if model_names is None:
            model_names = ['ERIC_2D']
        
        if conditions is None:
            conditions = list(self.phase_conditions.keys())
        
        if N_values is None:
            N_values = self.config.get_active_n_values()
        
        total_simulations = len(model_names) * len(conditions) * len(N_values)
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE STUDY")
        print(f"{'='*80}")
        print(f"Models: {model_names}")
        print(f"Conditions: {conditions}")
        print(f"Lattice Sizes: {N_values}")
        print(f"Total Simulations: {total_simulations}")
        print(f"{'='*80}\n")
        
        sim_count = 0
        for model in model_names:
            for condition in conditions:
                for N in N_values:
                    sim_count += 1
                    print(f"\n{'-'*80}")
                    print(f"Simulation {sim_count}/{total_simulations}")
                    print(f"Model: {model} | Condition: {condition} | N: {N}")
                    print(f"{'-'*80}")
                    
                    self.create_animation(
                        phase_condition=condition,
                        model_name=model,
                        save_video=save_videos,
                        output_dir=output_dir,
                        N=N
                    )
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE STUDY COMPLETE")
        print(f"{'='*80}")
        print(f"Total simulations: {sim_count}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")
    
    def list_available_models(self):
        """Print available models and their descriptions"""
        print("\nAvailable Models:")
        print("-" * 70)
        for name, metadata in self.model_metadata.items():
            print(f"{name:20s} - {metadata.description}")
            print(f"{'':20s}   Requires L: {metadata.requires_L}, Requires b: {metadata.requires_b}")
        print("-" * 70)
    
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
        description='Phase Dynamics Simulator with Lattice Size Studies (JAX GPU-Accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default config file
  python phasegrids_gpu_latticesize.py --generate-config
  
  # Run lattice size study for specific condition and model
  python phasegrids_gpu_latticesize.py --condition randomp --model ERIC_2D --n-values 20 50 100
  
  # Run lattice size study using config file
  python phasegrids_gpu_latticesize.py --config simulation_config_latticesize.json --condition randomp --model ERIC_2D
  
  # Run comprehensive study (multiple models, conditions, and N values)
  python phasegrids_gpu_latticesize.py --model ERIC_2D kuramoto_2D --condition randomp narrowp1 --n-values 20 50 100
  
  # List available models and conditions
  python phasegrids_gpu_latticesize.py --config simulation_config_latticesize.json --list-models --list-conditions
        """
    )
    
    parser.add_argument('--config', type=str, default='simulation_config_latticesize.json',
                       help='Path to configuration file (default: simulation_config_latticesize.json)')
    
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate default configuration file and exit')

    parser.add_argument('--condition', type=str, nargs='*', 
                       help='Phase condition(s) to simulate (e.g., randomp narrowp1)')

    parser.add_argument('--model', type=str, nargs='*', default=['ERIC_2D'], 
                       help='Model(s) to use for simulation (default: ERIC_2D)')
    
    parser.add_argument('--n-values', type=int, nargs='*',
                       help='Lattice sizes to simulate (e.g., 20 50 100). Overrides config file.')
    
    parser.add_argument('--all-conditions', action='store_true',
                       help='Run all available phase conditions')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save animation as video')
    
    parser.add_argument('--output-dir', type=str, default='gpu_outputs',
                       help='Directory to save output files (default: gpu_outputs)')
    
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    parser.add_argument('--list-conditions', action='store_true',
                       help='List available phase conditions and exit')
    
    return parser.parse_args()


def main():
    """Main execution function with lattice size study support"""
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
    
    # Override N_values from command line if provided
    if args.n_values is not None:
        config_dict['simulation']['N_values'] = args.n_values
        print(f"Using N values from command line: {args.n_values}")
    elif config_dict['simulation'].get('N_values') is None:
        # Default to [20, 50, 100] for lattice size studies
        config_dict['simulation']['N_values'] = [20, 50, 100]
        print(f"Using default N values: {config_dict['simulation']['N_values']}")
    
    # Create simulation config
    sim_config = SimulationConfig(**config_dict['simulation'])
    
    # Initialize simulator
    simulator = JAXPhaseDynamicsSimulator(config=sim_config)
    
    # Load phase conditions
    simulator.load_phase_conditions_from_config(config_dict['phase_conditions'])
    
    # Load model metadata from config if available
    if 'models' in config_dict:
        for model_name, model_info in config_dict['models'].items():
            if model_name in simulator.model_metadata:
                simulator.model_metadata[model_name] = ModelMetadata(
                    name=model_name,
                    requires_L=model_info['requires_L'],
                    requires_b=model_info['requires_b'],
                    description=model_info['description']
                )
    
    # List models if requested
    if args.list_models:
        simulator.list_available_models()
        if not args.list_conditions:
            return
    
    # List conditions if requested
    if args.list_conditions:
        simulator.list_available_conditions()
        if not args.list_models:
            return
        return
    
    # Convert models to list if not already
    models_to_run = args.model if isinstance(args.model, list) else [args.model]
    
    # Validate all models
    for model in models_to_run:
        if model not in simulator.models:
            print(f"Error: Unknown model '{model}'")
            print(f"Available models: {list(simulator.models.keys())}")
            return
    
    # Run simulations
    save_videos = not args.no_save
    N_values = sim_config.get_active_n_values()
    
    # Determine conditions to run
    if args.all_conditions:
        conditions_to_run = list(simulator.phase_conditions.keys())
    elif args.condition:
        conditions_to_run = args.condition if isinstance(args.condition, list) else [args.condition]
        # Validate conditions
        for condition in conditions_to_run:
            if condition not in simulator.phase_conditions:
                print(f"Error: Unknown phase condition '{condition}'")
                print(f"Available conditions: {list(simulator.phase_conditions.keys())}")
                return
    else:
        print("Error: Must specify either --condition or --all-conditions")
        print("Use --help for usage information")
        return
    
    # Run comprehensive study
    print(f"\n{'='*80}")
    print(f"STARTING LATTICE SIZE STUDY")
    print(f"{'='*80}")
    print(f"Models: {models_to_run}")
    print(f"Conditions: {conditions_to_run}")
    print(f"Lattice Sizes (N): {N_values}")
    print(f"Total Simulations: {len(models_to_run) * len(conditions_to_run) * len(N_values)}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    simulator.run_comprehensive_study(
        model_names=models_to_run,
        conditions=conditions_to_run,
        N_values=N_values,
        save_videos=save_videos,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()