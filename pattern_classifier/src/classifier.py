"""
Machine learning classifier for pattern recognition.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from typing import Tuple, Dict, Optional

class PatternClassifier:
    """ML classifier for phasemap pattern recognition"""
    
    def __init__(self, config: Dict):
        """
        Initialize classifier with configuration.
        
        Parameters
        ----------
        config : Dict
            Model configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.scaler = StandardScaler()
        self.classifier = self._build_classifier()
        self.classes_ = None
        self.feature_names = None
        self.is_trained = False
    
    def _build_classifier(self) -> RandomForestClassifier:
        """Build classifier from config"""
        model_config = self.config.get('model', {})
        
        return RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 200),
            max_depth=model_config.get('max_depth', 15),
            min_samples_split=model_config.get('min_samples_split', 5),
            random_state=model_config.get('random_state', 42),
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: Optional[list] = None) -> Dict:
        """
        Train classifier with cross-validation.
        
        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix (n_samples, n_features)
        y_train : np.ndarray
            Labels (n_samples,)
        feature_names : list, optional
            Feature names for interpretability
        
        Returns
        -------
        Dict
            Training results including CV scores
        """
        self.logger.info(f"Training classifier on {X_train.shape[0]} samples...")
        self.logger.info(f"Feature dimensions: {X_train.shape[1]}")
        
        self.feature_names = feature_names
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        self.logger.info("Class distribution:")
        for cls, count in zip(unique, counts):
            self.logger.info(f"  {cls}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Cross-validation
        cv_folds = self.config['model'].get('cv_folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.classifier, X_scaled, y_train, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        self.logger.info(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Train on full dataset
        self.classifier.fit(X_scaled, y_train)
        self.classes_ = self.classifier.classes_
        self.is_trained = True
        
        # Training accuracy
        y_pred_train = self.classifier.predict(X_scaled)
        train_acc = accuracy_score(y_train, y_pred_train)
        self.logger.info(f"Training accuracy: {train_acc:.3f}")
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_acc,
            'feature_importances': self.classifier.feature_importances_,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'classes': self.classes_
        }
        
        return results

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: Optional[list] = None,
                          save_dir: Optional[Path] = None) -> Dict:
        """
        Train classifier and perform comprehensive evaluation.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        feature_names : list, optional
            Feature names
        save_dir : Path, optional
            Directory to save evaluation results
        
        Returns
        -------
        Dict
            Complete evaluation results
        """
        # Train
        train_results = self.train(X_train, y_train, feature_names)
        
        # Evaluate
        eval_results = self.evaluate(X_test, y_test, save_path=save_dir)
        
        # Combine results
        results = {**train_results, **eval_results}
        
        # Generate comprehensive evaluation report
        if save_dir:
            self._generate_evaluation_report(results, save_dir)
        
        return results
    
    def _generate_evaluation_report(self, results: Dict, save_dir: Path):
        """
        Generate comprehensive evaluation report with multiple metrics.
        
        Creates:
        - Confusion matrix (normalized and raw)
        - Feature importance plot
        - Per-class performance metrics
        - Learning curves
        - Prediction confidence distribution
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion matrices (both normalized and raw counts)
        self._plot_confusion_matrices(results['confusion_matrix'], save_dir)
        
        # 2. Feature importance
        self._plot_feature_importance(save_dir, top_n=20)
        
        # 3. Per-class metrics
        self._plot_per_class_metrics(results, save_dir)
        
        # 4. Prediction confidence distribution
        if 'probabilities' in results:
            self._plot_confidence_distribution(results['probabilities'], save_dir)
        
        # 5. Text summary
        self._write_evaluation_summary(results, save_dir)
        
        self.logger.info(f"Comprehensive evaluation saved to: {save_dir}")
    
    def _plot_confusion_matrices(self, cm: np.ndarray, save_dir: Path):
        """Plot both normalized and raw confusion matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.classes_,
                   yticklabels=self.classes_,
                   cbar_kws={'label': 'Fraction'},
                   ax=axes[0])
        
        axes[0].set_title('Normalized Confusion Matrix', fontsize=14, pad=15)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Raw counts confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.classes_,
                   yticklabels=self.classes_,
                   cbar_kws={'label': 'Count'},
                   ax=axes[1])
        
        axes[1].set_title('Raw Counts Confusion Matrix', fontsize=14, pad=15)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        save_path = save_dir / 'confusion_matrices.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrices saved: {save_path}")
    
    def _plot_per_class_metrics(self, results: Dict, save_dir: Path):
        """Plot per-class precision, recall, and F1-score."""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Get predictions
        y_true = results.get('y_true')
        y_pred = results.get('predictions')
        
        if y_true is None or y_pred is None:
            self.logger.warning("Cannot plot per-class metrics: missing predictions")
            return
        
        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.classes_, zero_division=0
        )
        
        # Create bar plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(len(self.classes_))
        width = 0.6
        
        # Precision
        axes[0, 0].bar(x, precision, width, color='steelblue', alpha=0.8)
        axes[0, 0].set_ylabel('Precision', fontsize=12)
        axes[0, 0].set_title('Precision per Class', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.classes_, rotation=45, ha='right')
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].axhline(y=precision.mean(), color='red', linestyle='--', 
                           label=f'Mean: {precision.mean():.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[0, 1].bar(x, recall, width, color='seagreen', alpha=0.8)
        axes[0, 1].set_ylabel('Recall', fontsize=12)
        axes[0, 1].set_title('Recall per Class', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.classes_, rotation=45, ha='right')
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].axhline(y=recall.mean(), color='red', linestyle='--',
                           label=f'Mean: {recall.mean():.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-score
        axes[1, 0].bar(x, f1, width, color='coral', alpha=0.8)
        axes[1, 0].set_ylabel('F1-Score', fontsize=12)
        axes[1, 0].set_title('F1-Score per Class', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.classes_, rotation=45, ha='right')
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].axhline(y=f1.mean(), color='red', linestyle='--',
                           label=f'Mean: {f1.mean():.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Support (sample counts)
        axes[1, 1].bar(x, support, width, color='mediumpurple', alpha=0.8)
        axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
        axes[1, 1].set_title('Support per Class', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.classes_, rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'per_class_metrics.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Per-class metrics saved: {save_path}")
    
    def _plot_confidence_distribution(self, probabilities: np.ndarray, save_dir: Path):
        """Plot distribution of prediction confidence scores."""
        # Maximum probability for each prediction
        max_probs = np.max(probabilities, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(max_probs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=max_probs.mean(), color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {max_probs.mean():.3f}')
        axes[0].axvline(x=np.median(max_probs), color='green', linestyle='--',
                        linewidth=2, label=f'Median: {np.median(max_probs):.3f}')
        axes[0].set_xlabel('Prediction Confidence', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Prediction Confidence', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Cumulative distribution
        sorted_probs = np.sort(max_probs)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        
        axes[1].plot(sorted_probs, cumulative, linewidth=2, color='darkblue')
        axes[1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, 
                        label='90% threshold')
        axes[1].axvline(x=0.8, color='orange', linestyle='--', alpha=0.7,
                        label='Confidence = 0.8')
        axes[1].set_xlabel('Prediction Confidence', fontsize=12)
        axes[1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1].set_title('Cumulative Distribution of Confidence', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'confidence_distribution.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        pct_high_conf = np.mean(max_probs > 0.8) * 100
        pct_low_conf = np.mean(max_probs < 0.5) * 100
        
        self.logger.info(f"Confidence distribution saved: {save_path}")
        self.logger.info(f"  High confidence (>0.8): {pct_high_conf:.1f}%")
        self.logger.info(f"  Low confidence (<0.5): {pct_low_conf:.1f}%")
    
    def _write_evaluation_summary(self, results: Dict, save_dir: Path):
        """Write comprehensive evaluation summary to text file."""
        summary_path = save_dir / 'evaluation_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PATTERN CLASSIFIER EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"Cross-Validation Accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
            f.write(f"Training Accuracy: {results['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Number of Training Samples: {results['n_samples']}\n")
            f.write(f"Number of Features: {results['n_features']}\n")
            f.write(f"Number of Classes: {len(results['classes'])}\n")
            f.write(f"Classes: {', '.join(results['classes'])}\n\n")
            
            # Classification report
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-"*80 + "\n")
            f.write(results['classification_report'])
            f.write("\n\n")
            
            # Feature importance (top 10)
            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-"*80 + "\n")
            if self.feature_names:
                importances = results['feature_importances']
                indices = np.argsort(importances)[::-1][:10]
                for i, idx in enumerate(indices, 1):
                    f.write(f"{i:2d}. {self.feature_names[idx]:30s} : {importances[idx]:.4f}\n")
            f.write("\n")
            
            # Confidence statistics
            if 'probabilities' in results:
                max_probs = np.max(results['probabilities'], axis=1)
                f.write("PREDICTION CONFIDENCE STATISTICS\n")
                f.write("-"*80 + "\n")
                f.write(f"Mean Confidence: {max_probs.mean():.4f}\n")
                f.write(f"Median Confidence: {np.median(max_probs):.4f}\n")
                f.write(f"Std Dev: {max_probs.std():.4f}\n")
                f.write(f"Percentage with confidence > 0.8: {np.mean(max_probs > 0.8)*100:.1f}%\n")
                f.write(f"Percentage with confidence < 0.5: {np.mean(max_probs < 0.5)*100:.1f}%\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Evaluation summary written: {summary_path}")
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pattern types and confidence.
        
        Parameters
        ----------
        X_test : np.ndarray
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        probabilities : np.ndarray
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        X_scaled = self.scaler.transform(X_test)
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                save_path: Optional[Path] = None) -> Dict:
        """Evaluate classifier performance on test set."""
        predictions, probabilities = self.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        
        self.logger.info(f"\nTest Accuracy: {accuracy:.3f}")
        self.logger.info("\nClassification Report:")
        report = classification_report(y_test, predictions)
        self.logger.info(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.classes_)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'y_true': y_test  # Add this line
        }
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_dir: Path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.classes_,
                   yticklabels=self.classes_,
                   cbar_kws={'label': 'Fraction'})
        
        plt.title('Normalized Confusion Matrix', fontsize=14, pad=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = save_dir / 'confusion_matrix.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved: {save_path}")
    
    def _plot_feature_importance(self, save_dir: Path, top_n: int = 20):
        """Plot and save feature importance"""
        if self.feature_names is None:
            self.logger.warning("Feature names not available, skipping importance plot")
            return
        
        importance = self.classifier.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, pad=15)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = save_dir / 'feature_importance.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance saved: {save_path}")
    
    def save_model(self, save_dir: Path):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / 'classifier.joblib'
        scaler_path = save_dir / 'scaler.joblib'
        metadata_path = save_dir / 'metadata.joblib'
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'classes': self.classes_,
            'feature_names': self.feature_names,
            'config': self.config
        }, metadata_path)
        
        self.logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, load_dir: Path):
        """Load trained model and scaler"""
        model_path = load_dir / 'classifier.joblib'
        scaler_path = load_dir / 'scaler.joblib'
        metadata_path = load_dir / 'metadata.joblib'
        
        if not all([p.exists() for p in [model_path, scaler_path, metadata_path]]):
            raise FileNotFoundError(f"Model files not found in {load_dir}")
        
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)
        
        self.classes_ = metadata['classes']
        self.feature_names = metadata['feature_names']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {load_dir}")