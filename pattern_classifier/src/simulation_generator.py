"""
Simulation data generator for the pattern classification pipeline.
Generates ERIC model phase field simulations across parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
import logging
from typing import Tuple, Optional
import yaml


class ERICSimulationGenerator:
    """
    Generator for ERIC model simulations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sim_config = self.config.get('simulation', {})
        self.logger = logging.getLogger(__name__)
        
        # Setup output directories
        self.data_dir = Path(self.config['paths']['simulation_data'])
        self.image_dir = Path(self.config['paths']['phase_maps'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
    
    def omega_distribution(self, w_min: float, w_max: float, 
                          N: int, seed: int) -> np.ndarray:
        """Generate spatial distribution of natural frequencies."""
        np.random.seed(seed)
        omega = np.random.uniform(w_min, w_max, size=(N, N))
        return omega
    
    def initial_phase_distribution(self, N: int, seed: int) -> np.ndarray:
        """Generate initial phase distribution."""
        np.random.seed(seed + 100000)
        phase = np.random.uniform(-np.pi/2, np.pi/2, (N, N))
        return phase
    
    def ERIC_dynamics(self, phi: np.ndarray, N: int, K: float, 
                     omega: np.ndarray, Lambda: float) -> np.ndarray:
        """Compute ERIC model dynamics."""
        phi_up = np.roll(phi, -1, axis=0)
        phi_up[-1, :] = phi[-1, :]
        phi_down = np.roll(phi, 1, axis=0)
        phi_down[0, :] = phi[0, :]
        phi_left = np.roll(phi, -1, axis=1)
        phi_left[:, -1] = phi[:, -1]
        phi_right = np.roll(phi, 1, axis=1)
        phi_right[:, 0] = phi[:, 0]
        
        dphi_dt = np.zeros((N, N))
        dphi_dt += np.sin(phi_up - phi) + np.sin(phi_down - phi) + \
                   Lambda * (np.sin(phi_up - phi)**2) + Lambda * (np.sin(phi_down - phi)**2)
        dphi_dt += np.sin(phi_left - phi) + np.sin(phi_right - phi) + \
                   Lambda * (np.sin(phi_left - phi)**2) + Lambda * (np.sin(phi_right - phi)**2)
        dphi_dt = omega + K * dphi_dt
        
        return dphi_dt
    
    def integrate_RK4(self, phi: np.ndarray, dt: float, N: int, 
                     K: float, omega: np.ndarray, Lambda: float) -> np.ndarray:
        """Fourth-order Runge-Kutta integration step."""
        k1 = self.ERIC_dynamics(phi, N, K, omega, Lambda)
        k2 = self.ERIC_dynamics(phi + 0.5 * dt * k1, N, K, omega, Lambda)
        k3 = self.ERIC_dynamics(phi + 0.5 * dt * k2, N, K, omega, Lambda)
        k4 = self.ERIC_dynamics(phi + dt * k3, N, K, omega, Lambda)
        
        phi_new = phi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        phi_new = np.angle(np.exp(1j * phi_new))
        
        return phi_new
    
    def run_simulation(self, Lambda: float, seed: int) -> Tuple[np.ndarray, str]:
        """Run a single ERIC simulation to asymptotic state."""
        w_min = self.sim_config.get('w_min', 2*np.pi/180)
        w_max = self.sim_config.get('w_max', 2*np.pi/150)
        N = self.sim_config.get('N', 50)
        T = self.sim_config.get('T_final', 3000)
        dt = self.sim_config.get('dt', 0.1)
        
        Delta_omega = w_max - w_min
        K = self.sim_config.get('K_factor', 0.9) * Delta_omega
        
        omega = self.omega_distribution(w_min, w_max, N, seed)
        phi = self.initial_phase_distribution(N, seed)
        
        num_steps = int(T / dt) + 1
        for step in range(num_steps):
            phi = self.integrate_RK4(phi, dt, N, K, omega, Lambda)
        
        filename = f"Lambda_{Lambda:.2f}_seed_{seed:03d}"
        np.save(self.data_dir / f"{filename}.npy", phi)
        self._save_visualization(phi, Lambda, seed, T, filename)
        
        return phi, filename
    
    def _save_visualization(self, phi: np.ndarray, Lambda: float, 
                           seed: int, T: float, filename: str):
        """Create and save phase field visualization."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        im = ax.imshow(phi, cmap='twilight', norm=norm, interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        cbar.set_label('Phase', fontsize=14)
        
        ax.set_title(f'Λ = {Lambda:.2f}, seed = {seed}, t = {T:.0f} min', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        
        plt.savefig(self.image_dir / f"{filename}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def generate_parameter_sweep(self, Lambda_values: np.ndarray, 
                                n_realizations: int = 100,
                                n_cores: Optional[int] = None) -> pd.DataFrame:
        """Generate complete parameter sweep across Lambda values."""
        if n_cores is None:
            n_cores = max(1, cpu_count() - 1)
        
        all_params = []
        for Lambda in Lambda_values:
            for seed in range(n_realizations):
                all_params.append((Lambda, seed))
        
        self.logger.info(f"Generating {len(all_params)} simulations with {n_cores} cores...")
        start_time = time.time()
        
        with Pool(processes=n_cores) as pool:
            results = pool.starmap(self._run_simulation_wrapper, all_params)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed in {elapsed/3600:.2f} hours")
        
        log_data = []
        for (Lambda, seed), filename in zip(all_params, results):
            log_data.append({'Lambda': Lambda, 'seed': seed, 'filename': filename})
        
        df_log = pd.DataFrame(log_data).sort_values(['Lambda', 'seed']).reset_index(drop=True)
        df_log.to_csv('simulation_log.csv', index=False)
        
        return df_log
    
    def _run_simulation_wrapper(self, Lambda: float, seed: int) -> str:
        """Wrapper for parallel execution"""
        _, filename = self.run_simulation(Lambda, seed)
        return filename