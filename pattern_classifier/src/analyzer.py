"""
Analysis of pattern classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import logging

class PatternAnalyzer:
    """Analyze classified pattern data and generate figures"""
    
    def __init__(self, config: Dict, data_manager):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary
        data_manager : DataManager
            Data manager instance
        """
        self.config = config
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def calculate_pattern_fractions(self) -> pd.DataFrame:
        """
        Calculate fraction of each pattern type per Lambda.
        
        Returns
        -------
        pd.DataFrame
            Results with columns: Lambda, n_total, fraction_<pattern>
        """
        df = self.data_manager.load_classification_dataset()
        
        # Filter out unlabeled samples
        df_labeled = df[df['labeled'] == True].copy()
        
        if len(df_labeled) == 0:
            self.logger.warning("No labeled samples found")
            return pd.DataFrame()
        
        self.logger.info(f"Analyzing {len(df_labeled)} labeled samples...")
        
        results = []
        lambda_values = sorted(df_labeled['Lambda'].unique())
        
        for Lambda in lambda_values:
            subset = df_labeled[df_labeled['Lambda'] == Lambda]
            total = len(subset)
            
            if total == 0:
                continue
            
            # Count each pattern type
            pattern_counts = subset['pattern_type'].value_counts()
            
            result = {
                'Lambda': Lambda,
                'n_total': total
            }
            
            # Add fraction for each pattern category
            for pattern in self.config['patterns']['categories']:
                result[f'fraction_{pattern}'] = pattern_counts.get(pattern, 0) / total
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_pattern_evolution(self, 
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create publication-quality plot of pattern fractions vs Lambda.
        
        Parameters
        ----------
        save_path : Path, optional
            Path to save figure
        
        Returns
        -------
        plt.Figure
            Figure object
        """
        df_results = self.calculate_pattern_fractions()
        
        if len(df_results) == 0:
            self.logger.error("No results to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define colors and styles for each pattern - UPDATED
        pattern_styles = {
            'target': {
                'color': '#2E86AB', 
                'marker': 'o', 
                'linestyle': '-', 
                'linewidth': 2.5, 
                'markersize': 8, 
                'label': 'Target/Concentric',
                'zorder': 10
            },
            'spiral': {
                'color': '#A23B72', 
                'marker': 's', 
                'linestyle': '--',
                'linewidth': 2, 
                'markersize': 6, 
                'label': 'Spiral',
                'zorder': 9
            },
            'mixed': {
                'color': '#9B59B6', 
                'marker': 'D', 
                'linestyle': '-.',
                'linewidth': 2, 
                'markersize': 6, 
                'label': 'Mixed (Target+Spiral)',
                'zorder': 8
            },
            'sync': {
                'color': '#F18F01', 
                'marker': '^', 
                'linestyle': '--',
                'linewidth': 2, 
                'markersize': 6, 
                'label': 'Synchronized',
                'zorder': 7
            },
            'disorder': {
                'color': '#C73E1D', 
                'marker': 'v', 
                'linestyle': '--',
                'linewidth': 2, 
                'markersize': 6, 
                'label': 'Disordered',
                'zorder': 6
            },
            'multiple': {
                'color': '#6A4C93', 
                'marker': 'p', 
                'linestyle': ':',
                'linewidth': 1.5, 
                'markersize': 5, 
                'label': 'Multiple Sources',
                'zorder': 5
            },
            'other': {
                'color': '#8D8D8D', 
                'marker': 'x', 
                'linestyle': ':',
                'linewidth': 1.5, 
                'markersize': 5, 
                'label': 'Other',
                'zorder': 4
            }
        }
        
        # Plot each pattern type
        for pattern in self.config['patterns']['categories']:
            col = f'fraction_{pattern}'
            if col in df_results.columns:
                style = pattern_styles.get(pattern, {
                    'color': 'gray',
                    'marker': 'o',
                    'linestyle': '-',
                    'linewidth': 2,
                    'markersize': 5,
                    'label': pattern.capitalize(),
                    'zorder': 3
                })
                ax.plot(df_results['Lambda'], df_results[col], 
                       alpha=0.9, **style)
        
        # Formatting
        ax.set_xlabel(r'Coupling Parameter $\Lambda$', fontsize=16, fontweight='bold')
        ax.set_ylabel('Fraction of Initial Conditions', fontsize=14, fontweight='bold')
        ax.set_title('Pattern Emergence as Function of Coupling Parameter', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=11, framealpha=0.95, loc='best', ncol=2)
        
        # Set limits
        lambda_min = df_results['Lambda'].min()
        lambda_max = df_results['Lambda'].max()
        padding = (lambda_max - lambda_min) * 0.05
        
        ax.set_xlim(lambda_min - padding, lambda_max + padding)
        ax.set_ylim(-0.02, 1.02)
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved: {save_path}")
        
        return fig
    
    def generate_statistical_summary(self) -> Dict:
        """
        Generate statistical summary for manuscript.
        
        Returns
        -------
        Dict
            Statistical summary information
        """
        df_results = self.calculate_pattern_fractions()
        
        if len(df_results) == 0:
            return {}
        
        self.logger.info("\n" + "="*70)
        self.logger.info("STATISTICAL SUMMARY FOR REFEREE RESPONSE")
        self.logger.info("="*70)
        
        summary = {}
        
        # Get number of labeled samples
        df = self.data_manager.load_classification_dataset()
        n_labeled = df['labeled'].sum()
        summary['n_labeled_samples'] = int(n_labeled)
        
        # Load classifier metrics if available
        figures_dir = self.data_manager.paths['figures']
        eval_summary_path = figures_dir / 'evaluation_summary.txt'
        
        if eval_summary_path.exists():
            import re
            with open(eval_summary_path, 'r') as f:
                content = f.read()
                
                # Extract CV accuracy
                cv_match = re.search(r'Cross-Validation Accuracy: ([\d.]+) ± ([\d.]+)', content)
                if cv_match:
                    cv_mean = float(cv_match.group(1))
                    cv_std = float(cv_match.group(2))
                    summary['cv_accuracy'] = f"{cv_mean:.3f} ± {cv_std:.3f}"
        
        # Target wave analysis
        target_col = 'fraction_target'
        if target_col in df_results.columns:
            target_data = df_results[['Lambda', target_col]].copy()
            
            # Find where target patterns dominate
            target_dominant = target_data[target_data[target_col] > 0.5]
            
            if len(target_dominant) > 0:
                lambda_range = (target_dominant['Lambda'].min(), 
                              target_dominant['Lambda'].max())
                max_fraction = target_data[target_col].max()
                max_lambda = target_data.loc[target_data[target_col].idxmax(), 'Lambda']
                
                summary['target_dominance_range'] = lambda_range
                summary['target_max_fraction'] = max_fraction
                summary['target_max_lambda'] = max_lambda
                
                self.logger.info(f"\nTarget waves dominate (>50%) for Λ ∈ [{lambda_range[0]:.2f}, {lambda_range[1]:.2f}]")
                self.logger.info(f"Maximum target fraction: {max_fraction:.1%} at Λ = {max_lambda:.2f}")
            
            # Transition points
            self.logger.info(f"\nPattern transitions:")
            for i in range(len(target_data)-1):
                frac_current = target_data.iloc[i][target_col]
                frac_next = target_data.iloc[i+1][target_col]
                lambda_next = target_data.iloc[i+1]['Lambda']
                
                if frac_current < 0.5 and frac_next >= 0.5:
                    self.logger.info(f"  Onset of target dominance near Λ ≈ {lambda_next:.2f}")
                    summary['target_onset'] = lambda_next
                
                if frac_current >= 0.5 and frac_next < 0.5:
                    self.logger.info(f"  Loss of target dominance near Λ ≈ {lambda_next:.2f}")
                    summary['target_loss'] = lambda_next
        
        self.logger.info("\n" + "="*70)
        
        # Save summary to file
        summary_path = self.data_manager.paths['results'] / 'statistical_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("STATISTICAL SUMMARY\n")
            f.write("="*70 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Summary saved: {summary_path}")
        
        return summary
    

    def generate_classifier_performance_summary(self) -> Dict:
        """
        Generate summary of classifier performance for manuscript.
        
        Returns
        -------
        Dict
            Classifier performance metrics and statistics
        """
        figures_dir = self.data_manager.paths['figures']
        eval_summary_path = figures_dir / 'evaluation_summary.txt'
        
        if not eval_summary_path.exists():
            self.logger.warning("Classifier evaluation summary not found. Run training first.")
            return {}
        
        # Parse evaluation summary
        summary = {}
        with open(eval_summary_path, 'r') as f:
            content = f.read()
            
            # Extract key metrics using regex or simple parsing
            import re
            
            # Extract CV accuracy
            cv_match = re.search(r'Cross-Validation Accuracy: ([\d.]+) ± ([\d.]+)', content)
            if cv_match:
                summary['cv_accuracy_mean'] = float(cv_match.group(1))
                summary['cv_accuracy_std'] = float(cv_match.group(2))
            
            # Extract test accuracy
            test_match = re.search(r'Test Accuracy: ([\d.]+)', content)
            if test_match:
                summary['test_accuracy'] = float(test_match.group(1))
            
            # Extract confidence statistics
            high_conf_match = re.search(r'confidence > 0.8: ([\d.]+)%', content)
            if high_conf_match:
                summary['high_confidence_pct'] = float(high_conf_match.group(1))
            
            low_conf_match = re.search(r'confidence < 0.5: ([\d.]+)%', content)
            if low_conf_match:
                summary['low_confidence_pct'] = float(low_conf_match.group(1))
        
        # Save classifier summary 
        classifier_summary_path = self.data_manager.paths['results'] / 'classifier_summary.txt'
        with open(classifier_summary_path, 'w') as f:
            f.write("CLASSIFIER PERFORMANCE SUMMARY FOR MANUSCRIPT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Cross-validation accuracy: {summary.get('cv_accuracy_mean', 'N/A'):.4f} ± ")
            f.write(f"{summary.get('cv_accuracy_std', 'N/A'):.4f}\n")
            f.write(f"Test set accuracy: {summary.get('test_accuracy', 'N/A'):.4f}\n")
            f.write(f"High confidence predictions (>0.8): {summary.get('high_confidence_pct', 'N/A'):.1f}%\n")
            f.write(f"Low confidence predictions (<0.5): {summary.get('low_confidence_pct', 'N/A'):.1f}%\n")
            f.write("\nClassifier figures available in: outputs/figures/\n")
            f.write("  - confusion_matrices.pdf\n")
            f.write("  - per_class_metrics.pdf\n")
            f.write("  - feature_importance.pdf\n")
            f.write("  - confidence_distribution.pdf\n")
            f.write("  - evaluation_summary.txt (full details)\n")
        
        self.logger.info(f"Classifier summary saved: {classifier_summary_path}")
        
        return summary