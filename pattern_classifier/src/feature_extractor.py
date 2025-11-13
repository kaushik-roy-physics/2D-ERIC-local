"""
Feature extraction from phase field data.
"""

import numpy as np
from scipy import ndimage, fft
from skimage import feature
from scipy.stats import entropy
from typing import List, Dict
import logging

class PhasemapFeatureExtractor:
    """Extract quantitative features from phasemaps for classification"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        config : dict, optional
            Feature extraction configuration
        """
        self.config = config or {}
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, phi: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature vector from phase field.
        
        Parameters
        ----------
        phi : np.ndarray
            2D phase field array
        
        Returns
        -------
        np.ndarray
            Feature vector
        """
        features = []
        
        # Extract different feature types
        features.extend(self._radial_symmetry_features(phi))
        features.extend(self._spatial_correlation_features(phi))
        features.extend(self._fourier_features(phi))
        features.extend(self._gradient_features(phi))
        features.extend(self._topological_features(phi))
        features.extend(self._statistical_features(phi))
        
        return np.array(features)
    
    def _radial_symmetry_features(self, phi: np.ndarray) -> List[float]:
        """
        Measure radial symmetry (strong for target patterns).
        
        Target patterns show concentric structure with low radial variation
        at fixed distances from center.
        """
        if not self.feature_names or 'radial_mean' not in self.feature_names:
            self.feature_names.extend([
                'radial_mean', 'radial_std', 'radial_max', 'radial_corr',
                'radial_monotonicity'
            ])
        
        N = phi.shape[0]
        center = N // 2
        y, x = np.ogrid[:N, :N]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Compute radial profile
        r_int = r.astype(int)
        max_r = min(N//2, r_int.max())
        radial_bins = np.arange(0, max_r, 1)
        radial_std = []
        
        for rb in radial_bins:
            mask = (r_int == rb)
            if mask.sum() > 5:  # Require minimum samples
                radial_std.append(np.std(phi[mask]))
            else:
                radial_std.append(np.nan)
        
        radial_std = np.array(radial_std)
        valid_mask = ~np.isnan(radial_std)
        radial_std_valid = radial_std[valid_mask]
        
        if len(radial_std_valid) < 5:
            return [0, 0, 0, 0, 0]
        
        # Features
        mean_rad = np.mean(radial_std_valid)
        std_rad = np.std(radial_std_valid)
        max_rad = np.max(radial_std_valid)
        
        # Correlation with distance (indicates gradient)
        valid_bins = radial_bins[valid_mask]
        if len(valid_bins) > 1:
            corr = np.corrcoef(valid_bins, radial_std_valid)[0, 1]
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0
        
        # Monotonicity (smooth decay indicates organized pattern)
        diffs = np.diff(radial_std_valid)
        monotonicity = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0
        
        return [mean_rad, std_rad, max_rad, corr, monotonicity]
    
    def _spatial_correlation_features(self, phi: np.ndarray) -> List[float]:
        """
        Spatial autocorrelation at multiple length scales.
        
        Different patterns show distinct correlation decay profiles.
        """
        if 'corr_d1' not in self.feature_names:
            distances = [1, 3, 5, 10, 15, 20]
            self.feature_names.extend([f'corr_d{d}' for d in distances])
            self.feature_names.extend(['corr_decay', 'corr_peak'])
        
        # 2D autocorrelation via FFT
        phi_centered = phi - np.mean(phi)
        autocorr = fft.fft2(phi_centered)
        autocorr = np.abs(fft.ifft2(autocorr * np.conj(autocorr)))
        autocorr = fft.fftshift(autocorr)
        
        if autocorr.max() > 0:
            autocorr /= autocorr.max()
        
        N = phi.shape[0]
        center = N // 2
        y, x = np.ogrid[:N, :N]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Correlation at specific distances
        distances = [1, 3, 5, 10, 15, 20]
        corr_values = []
        
        for d in distances:
            mask = np.abs(r - d) < 1.5
            if mask.sum() > 0:
                corr_values.append(np.mean(autocorr[mask]))
            else:
                corr_values.append(0)
        
        # Decay rate
        decay_rate = np.mean(np.diff(corr_values)) if len(corr_values) > 1 else 0
        peak_corr = autocorr[center, center]
        
        return corr_values + [decay_rate, peak_corr]
    
    def _fourier_features(self, phi: np.ndarray) -> List[float]:
        """
        Frequency domain analysis.
        
        Spiral patterns show distinct peaks, target patterns show
        azimuthal symmetry in Fourier space.
        """
        if 'dominant_freq' not in self.feature_names:
            self.feature_names.extend([
                'dominant_freq', 'peak_power', 'low_freq_frac', 
                'spectral_entropy', 'freq_concentration'
            ])
        
        # 2D FFT
        phi_fft = fft.fft2(phi)
        phi_power = np.abs(fft.fftshift(phi_fft))**2
        
        N = phi.shape[0]
        center = N // 2
        
        # Radial power spectrum
        y, x = np.ogrid[:N, :N]
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
        
        max_r = min(N//2, r.max())
        radial_power = ndimage.mean(phi_power, labels=r, index=np.arange(0, max_r))
        
        # Normalize
        if radial_power.sum() > 0:
            radial_power_norm = radial_power / radial_power.sum()
        else:
            radial_power_norm = radial_power
        
        # Exclude DC component
        radial_power_norm = radial_power_norm[1:]
        
        if len(radial_power_norm) < 2:
            return [0, 0, 0, 0, 0]
        
        # Features
        dominant_freq = np.argmax(radial_power_norm) + 1
        peak_power = np.max(radial_power_norm)
        low_freq_frac = np.sum(radial_power_norm[:5]) / np.sum(radial_power_norm)
        
        # Entropy (lower = more concentrated power)
        spec_entropy = entropy(radial_power_norm + 1e-10)
        
        # Concentration (what fraction of power in top 3 modes)
        top_3_power = np.sum(np.sort(radial_power_norm)[-3:])
        
        return [dominant_freq, peak_power, low_freq_frac, spec_entropy, top_3_power]
    
    def _gradient_features(self, phi: np.ndarray) -> List[float]:
        """
        Phase gradient statistics.
        
        Sharp gradients indicate wave fronts or defects.
        """
        if 'grad_mean' not in self.feature_names:
            self.feature_names.extend([
                'grad_mean', 'grad_std', 'grad_p50', 'grad_p75',
                'grad_p90', 'grad_p95', 'grad_high_frac'
            ])
        
        grad_y, grad_x = np.gradient(phi)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        mean_grad = np.mean(grad_mag)
        std_grad = np.std(grad_mag)
        p50 = np.percentile(grad_mag, 50)
        p75 = np.percentile(grad_mag, 75)
        p90 = np.percentile(grad_mag, 90)
        p95 = np.percentile(grad_mag, 95)
        high_frac = np.mean(grad_mag > p90)
        
        return [mean_grad, std_grad, p50, p75, p90, p95, high_frac]
    
    def _topological_features(self, phi: np.ndarray) -> List[float]:
        """
        Topological features: defects, extrema, vorticity.
        
        Spiral patterns have topological defects at cores.
        """
        if 'n_maxima' not in self.feature_names:
            self.feature_names.extend([
                'n_maxima', 'n_minima', 'n_extrema_density',
                'curl_std', 'curl_max', 'curl_mean_abs'
            ])
        
        # Local extrema
        try:
            local_max = feature.peak_local_max(phi, min_distance=5)
            local_min = feature.peak_local_max(-phi, min_distance=5)
        except:
            local_max = np.array([])
            local_min = np.array([])
        
        n_max = len(local_max)
        n_min = len(local_min)
        extrema_density = (n_max + n_min) / (phi.shape[0] * phi.shape[1])
        
        # Curl (vorticity) - indicates rotation
        grad_y, grad_x = np.gradient(phi)
        curl = np.gradient(grad_x, axis=0) - np.gradient(grad_y, axis=1)
        
        curl_std = np.std(curl)
        curl_max = np.max(np.abs(curl))
        curl_mean_abs = np.mean(np.abs(curl))
        
        return [n_max, n_min, extrema_density, curl_std, curl_max, curl_mean_abs]
    
    def _statistical_features(self, phi: np.ndarray) -> List[float]:
        """Basic statistical features of phase field"""
        if 'phase_mean' not in self.feature_names:
            self.feature_names.extend([
                'phase_mean', 'phase_std', 'phase_skew', 'phase_kurtosis'
            ])
        
        from scipy.stats import skew, kurtosis
        
        phi_flat = phi.flatten()
        
        return [
            np.mean(phi_flat),
            np.std(phi_flat),
            skew(phi_flat),
            kurtosis(phi_flat)
        ]
    
    def extract_all_features(self, filenames: List[str], 
                            data_manager) -> np.ndarray:
        """
        Extract features from multiple phasemaps.
        
        Parameters
        ----------
        filenames : List[str]
            List of filenames (without extension)
        data_manager : DataManager
            Data manager for loading files
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        features_list = []
        
        self.logger.info(f"Extracting features from {len(filenames)} phasemaps...")
        
        for i, filename in enumerate(filenames):
            if (i + 1) % 50 == 0:
                self.logger.info(f"  Processed {i+1}/{len(filenames)}")
            
            try:
                phi = data_manager.load_phasemap(filename)
                features = self.extract_features(phi)
                features_list.append(features)
            except Exception as e:
                self.logger.error(f"Error extracting features from {filename}: {e}")
                # Add zero features as placeholder
                if len(features_list) > 0:
                    features_list.append(np.zeros_like(features_list[0]))
                else:
                    features_list.append(np.zeros(50))  # Approximate feature count
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names.copy()