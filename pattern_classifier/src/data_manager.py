"""
Data management for pattern classification project.
Handles loading, sampling, and organizing simulation data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import logging

class DataManager:
    """Manages simulation data and classification datasets"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataManager with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = {k: Path(v) for k, v in self.config['paths'].items()}
        self._setup_logging()
        self._ensure_directories()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def load_simulation_log(self) -> pd.DataFrame:
        """
        Load or create simulation log from data directory.
        
        Returns
        -------
        pd.DataFrame
            Simulation metadata with columns: Lambda, seed, filename
        """
        log_path = Path("simulation_log.csv")
        
        if log_path.exists():
            self.logger.info(f"Loading existing simulation log: {log_path}")
            return pd.read_csv(log_path)
        
        # Create log from simulation_data directory
        self.logger.info("Creating simulation log from data files...")
        
        data_files = list(self.paths['simulation_data'].glob("*.npy"))
        
        log_data = []
        for filepath in data_files:
            filename = filepath.stem
            # Parse filename: Lambda_X.XX_seed_YYY
            parts = filename.split('_')
            Lambda = float(parts[1])
            seed = int(parts[3])
            
            log_data.append({
                'Lambda': Lambda,
                'seed': seed,
                'filename': filename
            })
        
        df = pd.DataFrame(log_data)
        df = df.sort_values(['Lambda', 'seed']).reset_index(drop=True)
        df.to_csv(log_path, index=False)
        
        self.logger.info(f"Created simulation log with {len(df)} entries")
        return df
    
    def create_training_sample(self, 
                              force_resample: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified training sample across Lambda values.
        
        Parameters
        ----------
        force_resample : bool
            If True, create new sample even if one exists
        
        Returns
        -------
        training_df : pd.DataFrame
            Training samples to be manually labeled
        full_df : pd.DataFrame
            Complete dataset with training flags
        """
        output_path = self.paths['processed_data'] / "classification_dataset.csv"
        
        if output_path.exists() and not force_resample:
            self.logger.info(f"Loading existing classification dataset: {output_path}")
            df = pd.read_csv(output_path)
            training_df = df[df['is_training'] == True]
            return training_df, df
        
        # Create new sample
        df_log = self.load_simulation_log()
        
        samples_per_lambda = self.config['sampling']['samples_per_lambda']
        random_seed = self.config['sampling']['random_seed']
        
        self.logger.info(f"Creating stratified sample: {samples_per_lambda} per Lambda")
        
        training_samples = []
        lambda_values = sorted(df_log['Lambda'].unique())

        # Adaptive sampling

        for Lambda in lambda_values:
            subset = df_log[df_log['Lambda'] == Lambda]
            
            # Adaptive sampling based on Lambda value
            if 0.8 <= Lambda <= 2.0:  # Critical range
                n_samples = 12  # More samples
            else:  # Extreme values
                n_samples = 6   # Fewer samples
            
            n_samples = min(n_samples, len(subset))
            sampled = subset.sample(n=n_samples, random_state=random_seed)
            training_samples.append(sampled)

        # Fixed sampling
        
        #for Lambda in lambda_values:
        #    subset = df_log[df_log['Lambda'] == Lambda]
        #    n_samples = min(samples_per_lambda, len(subset))
        #    sampled = subset.sample(n=n_samples, random_state=random_seed)
        #    training_samples.append(sampled)
        
        training_df = pd.concat(training_samples, ignore_index=True)
        
        # Create full dataset with flags
        df_full = df_log.copy()
        df_full['is_training'] = df_full['filename'].isin(training_df['filename'])
        df_full['pattern_type'] = ''
        df_full['labeled'] = False
        df_full['prediction_confidence'] = np.nan
        
        # Save
        df_full.to_csv(output_path, index=False)
        
        n_train = training_df.shape[0]
        n_total = df_full.shape[0]
        
        self.logger.info(f"Training samples: {n_train} ({n_train/n_total*100:.1f}%)")
        self.logger.info(f"To be auto-classified: {n_total - n_train}")
        
        return training_df, df_full
    
    def load_classification_dataset(self) -> pd.DataFrame:
        """
        Load classification dataset.
        
        Returns
        -------
        pd.DataFrame
            Classification dataset with labels and flags
        """
        path = self.paths['processed_data'] / "classification_dataset.csv"
        
        if not path.exists():
            self.logger.warning("Classification dataset not found, creating new one...")
            _, df = self.create_training_sample()
            return df
        
        return pd.read_csv(path)
    
    def save_classification_dataset(self, df: pd.DataFrame):
        """Save classification dataset"""
        path = self.paths['processed_data'] / "classification_dataset.csv"
        df.to_csv(path, index=False)
        self.logger.info(f"Saved classification dataset: {path}")
    
    def get_training_statistics(self) -> Dict:
        """Get statistics on labeling progress"""
        df = self.load_classification_dataset()
        training = df[df['is_training'] == True]
        
        stats = {
            'total_training': len(training),
            'labeled': training['labeled'].sum(),
            'unlabeled': (~training['labeled']).sum(),
            'progress_pct': training['labeled'].mean() * 100,
            'pattern_counts': training[training['labeled']]['pattern_type'].value_counts().to_dict()
        }
        
        return stats
    
    def load_phasemap(self, filename: str) -> np.ndarray:
        """
        Load phasemap from .npy file.
        
        Parameters
        ----------
        filename : str
            Filename without extension
        
        Returns
        -------
        np.ndarray
            Phase field data
        """
        path = self.paths['simulation_data'] / f"{filename}.npy"
        return np.load(path)
    
    def get_image_path(self, filename: str) -> Path:
        """Get path to phase map visualization image"""
        return self.paths['phase_maps'] / f"{filename}.png"