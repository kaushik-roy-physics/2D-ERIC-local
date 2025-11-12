# Code for generating .mp4 movies showing phase evolution for the different phase oscillator models described in the paper: "Foci, waves, excitability: self-organization of phase waves in a model of asymmetrically coupled embryonic oscillators". Each frame is a 10x10 grid showing the phasemap in the (K,\Lambda) parameter space at a certain timepoint. Instead of sampling the natural frequencies of the oscillators from a uniform distribution, we tested truncated normal distributions to see if the patterns change. The parameter ranges can be changed by the user depending on what they want to study. The code is highly modular and additional phase models can be easily incorporated.
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
from scipy.stats import truncnorm

from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
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
    N: int = 50
    dt: float = 0.01
    w_min: float = 2 * np.pi / 180
    w_max: float = 2 * np.pi / 150
    mean: float = 2 * np.pi / 170               # Can be adjusted
    scale: float = 2 * np.pi * (1/160 - 1/170)  # Can be adjusted
    b: float = 0.01
    times: List[int] = None
    
    def __post_init__(self):
        if self.times is None:
            self.times = [100, 500, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 12000, 15000]


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
    def save_default_config(config_path: str = 'simulation_config_truncnorm.json'):
        """Save default configuration to JSON file"""
        default_config = {
            "simulation": {
                "N": 50,
                "dt": 0.01,
                "w_min": 0.03490658503988659,  # 2*pi/180
                "w_max": 0.04188790204786391,  # 2*pi/150
                "mean": 0.03695991357164463,  # 2*pi/170
                "scale": 0.00230999459822779,  # 2*pi*(1/160 -1/170)
                "b": 0.01,
                "times": [100, 500, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 12000, 15000]
            },
            "phase_conditions": {
                "narrowp1": {
                    "phase_func": "narrowp1_phase",
                    "phase_desc": r"$\theta_{in}=U(0, \pi/2)$",
                    "a_range": [0.4, 1.2, 10],
                    "L_range": [1.0, 2.0, 10],
                    #"a_range": [0.5, 2.0, 10],                    
                    #"L_range": [1.2, 1.6, 10],
                    #"a_range": [0.0, 2.0, 10],
                    #"L_range": [0.0, 2.5, 10],                    
                },
                "widerp3": {
                    "phase_func": "widerp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/4, \pi)$",
                    "a_range": [0.4, 1.2, 10],                   
                    "L_range": [1.0, 2.0, 10],
                    #"a_range": [0.5, 2.0, 10],
                    #"L_range": [1.2, 1.6, 10],
                    #"a_range": [0.0, 2.0, 10],
                    #"L_range": [0.0, 2.5, 10]
                },
                "narrowp3": {
                    "phase_func": "narrowp3_phase",
                    "phase_desc": r"$\theta_{in}=U(-\pi/2, \pi/2)$",
                    "a_range": [0.8, 1.2, 10],                    
                    "L_range": [1.6, 2.0, 10],
                    #"a_range": [0.5, 2.0, 10],
                    #"L_range": [1.2, 1.6, 10],
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
    GPU-optimized simulator using JAX batched operations.
    
    Simulates all parameter combinations simultaneously on GPU for maximum performance.
    Supports multiple phase oscillator models with proper batching.
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
        
        self.N = self.config.N
        self.dt = self.config.dt
        self.w_min = self.config.w_min
        self.w_max = self.config.w_max

        self.mean = self.config.mean
        self.scale = self.config.scale
        
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
    
    def omega_uniform(self) -> np.ndarray:
        """Generate uniform omega distribution"""
        return np.random.uniform(self.w_min, self.w_max, size=(self.N, self.N))

    def omega_trunc_normal(self) -> np.ndarray:  
        """Generate truncated normal omega distribution"""
    
        # Define the parameters of the truncated normal distribution
        
        a = (self.w_min - self.mean) / self.scale      # Lower bound
        b = (self.w_max - self.mean) / self.scale      # Upper bound
    
    
        # Generate random samples from the truncated Gaussian distribution
        
        omega = truncnorm.rvs(a, b , loc = mean , scale = scale , size = (N,N))
        
        return omega

    
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
                        model_name: str = 'ERIC_2D') -> List[np.ndarray]:
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
        
        Returns
        -------
        list of np.ndarray
            List of frames (each of shape (B, N, N)) at designated timepoints
        """
        print(f"  Starting batched GPU simulation with {phi_initial_batch.shape[0]} parameter combinations...")
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
                                 model_name: str = 'ERIC_2D') -> List[List[np.ndarray]]:
        """
        Get all frames for a specific phase condition using batched GPU simulation.
        """
        if phase_condition not in self.phase_conditions:
            raise ValueError(f"Unknown phase condition: {phase_condition}. "
                           f"Available: {list(self.phase_conditions.keys())}")
        
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
        omega_initial = self.omega_uniform()
        phase_func = self.get_phase_function(condition_data.phase_func)
        phi_initial = phase_func(self.N)
        
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
        print(f"  Grid size: {self.N}x{self.N}")
        print(f"  Total cells: {n_params * self.N * self.N:,}")
        
        # Run single batched simulation on GPU
        frames = self.simulate_batched(phi_batch, omega_batch, K_array, 
                                       L_array_full, model_name)
        
        # Reshape frames for compatibility with visualization code
        # frames is currently: [frame0, frame1, ...] where each frame is (B, N, N)
        # Need to convert to: [[param0_frames], [param1_frames], ...]
        frames_reshaped = []
        for i in range(n_params):
            param_frames = [frame[i] for frame in frames]  # Extract all timepoints for this parameter
            frames_reshaped.append(param_frames)
        
        return frames_reshaped
    
    def create_animation(self, phase_condition: str, model_name: str = 'ERIC_2D', 
                        save_gif: bool = True, output_dir: str = 'output') -> animation.FuncAnimation:
        """
        Create animation for a specific phase condition and model.
        
        Parameters
        ----------
        phase_condition : str
            Name of phase condition
        model_name : str
            Name of model to use
        save_gif : bool
            Whether to save as GIF
        output_dir : str
            Directory to save output
        
        Returns
        -------
        animation.FuncAnimation
            The created animation
        """
        print(f"\nProcessing {phase_condition} with {model_name}...")
        start_time = time.time()
        
        condition_data = self.phase_conditions[phase_condition]
        frames_data = self.get_frames_for_condition(phase_condition, model_name)
        
        metadata = self.model_metadata.get(model_name)
        requires_L = metadata.requires_L if metadata else True

        # Determine grid layout
        if not requires_L:
            a_values = np.linspace(0, 4, 100)
            n_rows = 10
            n_cols = 10

        else:
            a_values =  np.linspace(*condition_data.a_range)
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
                plot = ax.imshow(np.zeros((self.N, self.N)), 
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
            omega_desc = (fr'$\omega = TN \left(\frac{{2\pi}}{{170}}, '
                         fr'\frac{{2\pi}}{{160}}- \frac{{2\pi}}{{170}} \right)$ min$^{{-1}}$')
            
            fig.suptitle(f'Phases at t={self.times[frame_idx]} min\n'
                        f'{phase_desc}, {omega_desc}, Model: {model_name}',
                        fontsize=14, y=0.98)
            
            return plots
            
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.times), 
                                    interval=1000, blit=False)  # Changed from 500 to 1000
        
        if save_gif:
            os.makedirs(output_dir, exist_ok=True)
            L_suffix = 'with_L' if requires_L else 'grid10x10'
            filename = os.path.join(output_dir, 
                                   f'phasegrids_{phase_condition}_{model_name}_{L_suffix}_truncnorm.mp4')  # Changed .gif to .mp4
            print(f"Saving animation to {filename}...")
            ani.save(filename, writer='ffmpeg', fps=1, dpi=150)  # Changed writer to 'ffmpeg', added fps=1
            print(f"Animation saved as {filename}")

        
        end_time = time.time()
        print(f'Total time for {phase_condition}: {end_time - start_time:.2f} seconds')
        
        plt.close()
        return ani
    
    def run_all_conditions(self, model_name: str = 'ERIC_2D', 
                          save_gifs: bool = True, output_dir: str = 'output'):
        """
        Run simulations for all phase conditions.
        
        Parameters
        ----------
        model_name : str
            Name of model to use
        save_gifs : bool
            Whether to save animations as GIFs
        output_dir : str
            Directory to save outputs
        """
        for condition in self.phase_conditions.keys():
            self.create_animation(condition, model_name, save_gifs, output_dir)
    
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
        description='Phase Dynamics Simulator (JAX GPU-Accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default config file
  python phasegrids_gpu_truncnorm.py --generate-config
  
  # Run specific condition and model
  python phasegrids_gpu_truncnorm.py --config simulation_config_truncnorm.json --condition randomp --model ERIC_2D
  
  # Run all conditions for Kuramoto model
  python phasegrids_gpu_truncnorm.py --config simulation_config_truncnorm.json --model kuramoto_2D --all-conditions
  
  # List available models and conditions
  python phasegrids_gpu_truncnorm.py --config simulation_config_truncnorm.json --list-models --list-conditions
        """
    )
    
    parser.add_argument('--config', type=str, default='simulation_config_truncnorm.json',
                       help='Path to configuration file (default: simulation_config_truncnorm.json)')
    
    parser.add_argument('--generate-config', action='store_true',
                       help='Generate default configuration file and exit')

    parser.add_argument('--condition', type=str, nargs='*', 
                   help='Phase condition(s) to simulate (e.g., randomp narrowp1)')

    parser.add_argument('--model', type=str, nargs='*', default=['ERIC_2D'], 
                   help='Model(s) to use for simulation (default: ERIC_2D)')
    
#    parser.add_argument('--condition', type=str,
#                       help='Phase condition to simulate (e.g., randomp, narrowp1)')
    
#    parser.add_argument('--model', type=str, default='ERIC_2D',
#                       help='Model to use for simulation (default: ERIC_2D)')
    
    parser.add_argument('--all-conditions', action='store_true',
                       help='Run all available phase conditions')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save animation as GIF')
    
    parser.add_argument('--output-dir', type=str, default='truncnorm_outputs',
                       help='Directory to save output files (default: truncnorm_outputs)')
    
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
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
    simulator = JAXPhaseDynamicsSimulator(config=sim_config)
    
    # Load phase conditions
    simulator.load_phase_conditions_from_config(config_dict['phase_conditions'])
    
    # Load model metadata from config if available
    if 'models' in config_dict:
        for model_name, model_info in config_dict['models'].items():
            if model_name in simulator.model_metadata:
                # Update metadata from config
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
    save_gifs = not args.no_save
    
    if args.all_conditions:
        print(f"\nRunning all conditions with {len(models_to_run)} model(s)")
        for model in models_to_run:
            print(f"\n{'='*70}")
            print(f"MODEL: {model}")
            print(f"{'='*70}")
            simulator.run_all_conditions(model, save_gifs, args.output_dir)
    
    elif args.condition:
        # Convert conditions to list if not already
        conditions_to_run = args.condition if isinstance(args.condition, list) else [args.condition]
        
        # Validate all conditions
        for condition in conditions_to_run:
            if condition not in simulator.phase_conditions:
                print(f"Error: Unknown phase condition '{condition}'")
                print(f"Available conditions: {list(simulator.phase_conditions.keys())}")
                return
        
        # Print summary
        print(f"\nRunning {len(conditions_to_run)} condition(s) with {len(models_to_run)} model(s)")
        print(f"Total simulations: {len(conditions_to_run) * len(models_to_run)}")
        
        # Run all combinations
        for model in models_to_run:
            for condition in conditions_to_run:
                print(f"\n{'='*70}")
                print(f"Condition: {condition} | Model: {model}")
                print(f"{'='*70}")
                simulator.create_animation(condition, model, save_gifs, args.output_dir)
    
    else:
        print("Error: Must specify either --condition or --all-conditions")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()