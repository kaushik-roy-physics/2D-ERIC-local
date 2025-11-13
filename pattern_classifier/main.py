"""
Main script for pattern classification workflow. 
Part of the supplemental material accompanying the paper:
Foci, waves, excitability: self-organization of phase waves in a model of asymmetrically coupled embryonic oscillators by Kaushik Roy and Paul Francois

Author: Kaushik Roy
Email: kr70@rice.edu
"""

import argparse
import logging
from pathlib import Path
import sys

from src.data_manager import DataManager
from src.feature_extractor import PhasemapFeatureExtractor
from src.classifier import PatternClassifier
from src.analyzer import PatternAnalyzer

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pattern_classification.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_simulations(args):
    """Step 0: Generate simulation dataset"""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STEP 0: GENERATING SIMULATION DATASET")
    logger.info("="*70)
    
    from src.simulation_generator import ERICSimulationGenerator
    
    # Load config to get Lambda range
    import yaml
    import numpy as np
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    analysis_config = config.get('analysis', {})
    lambda_min, lambda_max = analysis_config.get('lambda_range', [0, 2.5])
    n_lambda = analysis_config.get('n_lambda_points', 50)
    n_realizations = analysis_config.get('n_realizations_per_lambda', 100)
    
    Lambda_values = np.linspace(lambda_min, lambda_max, n_lambda)
    
    generator = ERICSimulationGenerator(args.config)
    df_log = generator.generate_parameter_sweep(
        Lambda_values, 
        n_realizations=n_realizations,
        n_cores=args.n_cores
    )
    
    logger.info(f"\nGenerated {len(df_log)} simulations")
    logger.info("Next step: python main.py prepare")

def prepare_training_data(args):
    """Step 1: Prepare training sample"""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STEP 1: PREPARING TRAINING SAMPLE")
    logger.info("="*70)
    
    dm = DataManager(args.config)
    training_df, full_df = dm.create_training_sample(force_resample=args.force_resample)
    
    logger.info(f"\nTraining sample created: {len(training_df)} samples")
    logger.info("Next step: Run labeling tool")
    logger.info("streamlit run src/gui/labeling_app.py")

def train_classifier(args):
    """Step 2: Train ML classifier"""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STEP 2: TRAINING CLASSIFIER")
    logger.info("="*70)
    
    # Initialize components
    dm = DataManager(args.config)
    extractor = PhasemapFeatureExtractor(dm.config)
    classifier = PatternClassifier(dm.config)
    
    # Load labeled data
    df = dm.load_classification_dataset()
    training_data = df[(df['is_training'] == True) & (df['labeled'] == True)]
    
    if len(training_data) == 0:
        logger.error("No labeled training data found!")
        logger.error("Please label training samples using: streamlit run src/gui/labeling_app.py")
        return
    
    logger.info(f"Found {len(training_data)} labeled samples")
    
    # Split into train/test (80/20)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        training_data, 
        test_size=0.2, 
        random_state=42,
        stratify=training_data['pattern_type']
    )
    
    logger.info(f"Training set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Extract features
    logger.info("Extracting features...")
    X_train = extractor.extract_all_features(train_df['filename'].values, dm)
    y_train = train_df['pattern_type'].values
    
    X_test = extractor.extract_all_features(test_df['filename'].values, dm)
    y_test = test_df['pattern_type'].values
    
    feature_names = extractor.get_feature_names()
    
    # Train and evaluate
    logger.info("Training classifier...")
    figures_dir = dm.paths['figures']
    results = classifier.train_and_evaluate(
        X_train, y_train, X_test, y_test, 
        feature_names=feature_names,
        save_dir=figures_dir
    )
    
    # Save model
    model_dir = dm.paths['models']
    classifier.save_model(model_dir)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"CV Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    logger.info(f"Test Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Model saved to: {model_dir}")
    logger.info(f"Evaluation figures saved to: {figures_dir}")

def classify_all(args):
    """Step 3: Classify all remaining data"""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STEP 3: CLASSIFYING FULL DATASET")
    logger.info("="*70)
    
    # Initialize components
    dm = DataManager(args.config)
    extractor = PhasemapFeatureExtractor(dm.config)
    classifier = PatternClassifier(dm.config)
    
    # Load trained model
    model_dir = dm.paths['models']
    try:
        classifier.load_model(model_dir)
    except FileNotFoundError:
        logger.error("Trained model not found!")
        logger.error("Please train classifier first: python main.py train")
        return
    
    # Load dataset
    df = dm.load_classification_dataset()
    
    # Get unlabeled samples
    unlabeled = df[(df['is_training'] == False) | (~df['labeled'])]
    
    if len(unlabeled) == 0:
        logger.info("All samples already classified!")
        return
    
    logger.info(f"Classifying {len(unlabeled)} samples...")
    
    # Extract features
    X_test = extractor.extract_all_features(
        unlabeled['filename'].values,
        dm
    )
    
    # Predict
    predictions, probabilities = classifier.predict(X_test)
    
    # Store predictions
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
        original_idx = unlabeled.index[idx]
        df.at[original_idx, 'pattern_type'] = pred
        df.at[original_idx, 'labeled'] = True
        df.at[original_idx, 'prediction_confidence'] = prob.max()
    
    # Save
    dm.save_classification_dataset(df)
    
    logger.info("Classification complete!")
    logger.info("\nPattern distribution:")
    logger.info(df['pattern_type'].value_counts())

def analyze_results(args):
    """Step 4: Analyze results and generate figures"""
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STEP 4: ANALYZING RESULTS")
    logger.info("="*70)
    
    # Initialize
    dm = DataManager(args.config)
    analyzer = PatternAnalyzer(dm.config, dm)
    
    # Generate analysis
    logger.info("Calculating pattern fractions...")
    df_results = analyzer.calculate_pattern_fractions()
    
    if len(df_results) == 0:
        logger.error("No classified data found!")
        return
    
    # Save results table
    results_path = dm.paths['results'] / 'pattern_fractions.csv'
    df_results.to_csv(results_path, index=False)
    logger.info(f"Results table saved: {results_path}")
    
    # Generate plot
    logger.info("Generating publication figure...")
    fig_path = dm.paths['figures'] / 'pattern_fractions_vs_lambda.pdf'
    analyzer.plot_pattern_evolution(fig_path)
    
    # Generate statistical summary
    logger.info("Generating statistical summary...")
    summary = analyzer.generate_statistical_summary()
    
    # Generate classifier performance summary
    logger.info("Summarizing classifier performance...")
    classifier_summary = analyzer.generate_classifier_performance_summary()
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Figure: {fig_path}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Summary: {dm.paths['results'] / 'statistical_summary.txt'}")
    logger.info(f"Classifier Summary: {dm.paths['results'] / 'classifier_summary.txt'}")
    logger.info(f"Classifier Figures: {dm.paths['figures']}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pattern Classification Workflow for ERIC Simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
  0. generate  - Generate simulation dataset (ERIC model parameter sweep)
  1. prepare   - Create stratified training sample for manual labeling
  2. label     - Launch labeling GUI (streamlit run src/gui/labeling_app.py)
  3. train     - Train ML classifier on labeled data with evaluation
  4. classify  - Classify all remaining unlabeled data
  5. analyze   - Generate final results, figures, and manuscript text
  6. all       - Run steps 3-5 sequentially (assumes labeling complete)

Complete Workflow Example:
  # Step 0: Generate simulation data (run once)
  python main.py generate --n-cores 8
  
  # Step 1: Create training sample
  python main.py prepare
  
  # Step 2: Label training data interactively
  streamlit run src/gui/labeling_app.py
  
  # Step 3: Train classifier and evaluate performance
  python main.py train
  
  # Step 4: Classify remaining data
  python main.py classify
  
  # Step 5: Analyze and generate results
  python main.py analyze
  
  # Alternative: Run steps 3-5 together (if labeling done)
  python main.py all

  If simulations already exist:
  python main.py prepare
  streamlit run src/gui/labeling_app.py
  python main.py all

Output Locations:
  - Simulation data:        data/simulation_data/*.npy
  - Phase map images:       data/phase_maps/*.png
  - Trained model:          outputs/models/
  - Evaluation figures:     outputs/figures/
  - Results & statistics:   outputs/results/


Configuration:
  Edit config/config.yaml to modify:
  - Simulation parameters (Lambda range, grid size, integration time)
  - Pattern categories and labeling shortcuts
  - ML model hyperparameters
  - Sampling strategy for training set

For Help:
  python main.py <command> --help
        """
    )
    
    parser.add_argument(
        'command',
        choices=['generate', 'prepare', 'train', 'classify', 'analyze', 'all'],
        help='Command to execute (see workflow steps above)'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--force-resample',
        action='store_true',
        help='Force recreation of training sample (prepare command only)'
    )
    
    parser.add_argument(
        '--n-cores',
        type=int,
        default=None,
        help='Number of CPU cores for parallel processing (generate command). Default: auto-detect'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Execute command
    if args.command == 'generate':
        generate_simulations(args)
    
    elif args.command == 'prepare':
        prepare_training_data(args)
    
    elif args.command == 'train':
        train_classifier(args)
    
    elif args.command == 'classify':
        classify_all(args)
    
    elif args.command == 'analyze':
        analyze_results(args)
    
    elif args.command == 'all':
        logger.warning("="*70)
        logger.warning("RUNNING STEPS: TRAIN → CLASSIFY → ANALYZE")
        logger.warning("="*70)
        logger.warning("Note: This assumes training data has been manually labeled")
        logger.warning("If labeling is incomplete, press Ctrl+C and run:")
        logger.warning("  streamlit run src/gui/labeling_app.py")
        logger.warning("="*70)
        
        try:
            input("Press Enter to continue or Ctrl+C to abort...")
        except KeyboardInterrupt:
            logger.info("\nAborted by user")
            return
        
        train_classifier(args)
        classify_all(args)
        analyze_results(args)
    
    logger.info("\n" + "="*70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*70)
    logger.info("\nNext steps:")
    
    if args.command == 'generate':
        logger.info("  1. Run: python main.py prepare")
        logger.info("  2. Label data: streamlit run src/gui/labeling_app.py")
    elif args.command == 'prepare':
        logger.info("  1. Label data: streamlit run src/gui/labeling_app.py")
        logger.info("  2. Train model: python main.py train")
    elif args.command == 'train':
        logger.info("  1. Classify data: python main.py classify")
        logger.info("  2. Or run both: python main.py all")
    elif args.command == 'classify':
        logger.info("  1. Generate results: python main.py analyze")
    elif args.command == 'analyze':
        logger.info("  ✓ All analysis complete!")
        logger.info("  - Check outputs/figures/ for plots")
        logger.info("  - Check outputs/results/ for statistics and text")
    
    logger.info("="*70)


if __name__ == "__main__":
    main()