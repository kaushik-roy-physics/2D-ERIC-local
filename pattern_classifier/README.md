# Pattern Classification for 2D ERIC Model Simulations

A comprehensive machine learning pipeline for automated classification of spatiotemporal patterns in phase field simulations of the ERIC (Elliptic Radial Isochron Cycle) model and studying the evolution of patterns as a function of the model parameter $\Lambda$ which represents the asymmetry in the ERIC model coupling function. This project implements physics-informed feature extraction and Random Forest classification to distinguish between target waves, spirals, synchronized states, and other pattern types emerging from locally coupled phase oscillators.

## Overview

This repository provides tools for:
- Generating ERIC model simulations across parameter space
- Creating stratified training samples for manual pattern labeling
- Interactive GUI-based labeling of phase map patterns
- Physics-informed feature extraction from phase field data
- Training and evaluating Random Forest classifiers
- Analyzing pattern statistics across coupling parameter $\Lambda$
- Generating figures and statistical summaries
.

## Project Structure
```
pattern_classifier/
├── config/
│   └── config.yaml              # Central configuration file
├── data/
│   ├── simulation_data/         # .npy files with phase field data
│   ├── phase_maps/              # .png visualizations of phase maps
│   └── processed/               # Generated datasets and labels
├── src/
│   ├── __init__.py
│   ├── simulation_generator.py  # ERIC model simulation engine
│   ├── data_manager.py          # Data loading and sampling utilities
│   ├── feature_extractor.py     # Physics-informed feature extraction
│   ├── classifier.py            # Random Forest classifier wrapper
│   ├── analyzer.py              # Pattern statistics and plotting
│   └── gui/
│       ├── __init__.py
│       └── labeling_app.py      # Streamlit-based labeling interface
├── outputs/
│   ├── models/                  # Trained classifier and scaler
│   ├── figures/                 # Generated plots and confusion matrices
│   └── results/                 # Statistical summaries and CSV files
├── requirements.txt
├── main.py                      # Main workflow orchestration script
└── README.md
```

## Requirements

### Python Environment

- **Python**: ≥3.8
- **Operating System**: Linux, macOS, or Windows with WSL (GPU optional but not required)

### Core Dependencies

The project relies on scientific computing and machine learning libraries:
```
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
scipy>=1.7.0           # Scientific algorithms (FFT, signal processing)
scikit-learn>=1.0.0    # Machine learning (Random Forest, preprocessing)
scikit-image>=0.19.0   # Image processing (feature detection)
matplotlib>=3.4.0      # Plotting and visualization
seaborn>=0.11.0        # Statistical visualization
streamlit>=1.20.0      # Interactive GUI for labeling
Pillow>=9.0.0          # Image handling
PyYAML>=6.0            # Configuration file parsing
joblib>=1.1.0          # Model serialization
```

## Installation: Pip Install (Recommended)

# Clone the repository
```bash
git clone https://github.com/kaushik-roy-physics/2D-ERIC-local.git
cd 2D-ERIC-local/pattern_classifier
```
## Configuration

All simulation parameters, file paths, and classifier hyperparameters are centralized in `config/config.yaml`. Key configuration sections:

### Paths Configuration
```yaml
paths:
  simulation_data: "data/simulation_data"    # Phase field .npy files
  phase_maps: "data/phase_maps"              # Phase map visualizations
  processed_data: "data/processed"           # Classification datasets
  models: "outputs/models"                   # Trained classifiers
  figures: "outputs/figures"                 # Generated plots
  results: "outputs/results"                 # Statistical outputs
```

### Simulation Parameters
```yaml
simulation:
  w_min: 0.03490658503988659    # Minimum natural frequency (2π/180 rad/min)
  w_max: 0.04188790204786391    # Maximum natural frequency (2π/150 rad/min)
  N: 50                          # Lattice size (50×50 grid)
  T_final: 3000                  # Integration time (minutes)
  dt: 0.1                        # Time step for RK4 integration
  K_factor: 0.9                  # Coupling strength factor
```

### Analysis Parameters
```yaml
analysis:
  lambda_range: [0, 2.5]              # Range of coupling parameter Λ
  n_lambda_points: 50                 # Number of Λ values to sample
  n_realizations_per_lambda: 100      # Random seeds per Λ value
```

### Pattern Categories
```yaml
patterns:
  categories:
    - 'target'      # Target/concentric waves
    - 'spiral'      # Spiral waves
    - 'mixed'       # Mixed target+spiral landscapes
    - 'sync'        # Synchronized states
    - 'disorder'    # Disordered configurations
    - 'multiple'    # Multiple competing sources
    - 'other'       # Transitional/ambiguous states
  
  shortcuts:        # Keyboard shortcuts for labeling GUI
    target: 't'
    spiral: 's'
    mixed: 'm'
    sync: 'y'
    disorder: 'd'
    multiple: 'u'
    other: 'o'
```

### ML Model Parameters
```yaml
model:
  n_estimators: 200          # Number of trees in Random Forest
  max_depth: 15              # Maximum depth of each tree
  min_samples_split: 5       # Minimum samples required to split node
  cv_folds: 5                # Number of cross-validation folds
  random_state: 42           # Random seed for reproducibility
```

## Workflow

The complete workflow consists of six main steps:

### Step 0: Generate Simulation Data (Optional)

If you need to generate new simulations:
```bash
python main.py generate --n-cores 8
```

**What it does**:
- Generates ERIC model simulations across the $\Lambda$ parameter space
- Uses 4th-order Runge-Kutta integration (RK4) with $\Delta t = 0.1$ min
- Integrates each simulation to $t = 3000$ min to reach asymptotic states
- Saves phase fields as `.npy` files in `data/simulation_data/`
- Creates phase map visualizations as `.png` in `data/phase_maps/`
- Generates `simulation_log.csv` with metadata

**Options**:
- `--n-cores N`: Number of CPU cores for parallel processing (default: auto-detect)
- Uses multiprocessing for parallelization across parameter/seed combinations

**Output**:
```
data/simulation_data/Lambda_1.79_seed_042.npy
data/phase_maps/Lambda_1.79_seed_042.png
simulation_log.csv
```

**Time estimate**: ~2-3 hours for 5000 simulations (depending on CPU cores)

### Step 1: Prepare Training Sample

Create a stratified sample for manual labeling:
```bash
python main.py prepare
```

**What it does**:
- Loads `simulation_log.csv` containing all simulation metadata
- Performs adaptive stratified sampling across $\Lambda$ values:
  - 12 samples per $\Lambda$ in critical regime ($0.8 \leq \Lambda \leq 2.0$)
  - 6 samples per $\Lambda$ outside critical regime
- Creates `data/processed/classification_dataset.csv` with columns:
  - `Lambda`: Coupling parameter value
  - `seed`: Random seed for initial conditions
  - `filename`: Base filename (without extension)
  - `is_training`: Boolean flag (True for training samples)
  - `pattern_type`: Pattern category (empty initially)
  - `labeled`: Boolean flag (False initially)
  - `prediction_confidence`: Float (NaN initially)

**Options**:
- `--force-resample`: Force recreation of training sample (overwrites existing)

**Output**:
```
data/processed/classification_dataset.csv (~500 training samples)
```

### Step 2: Label Training Data

Launch the interactive labeling GUI:
```bash
streamlit run src/gui/labeling_app.py
```

**What it does**:
- Opens web-based GUI in your default browser (typically http://localhost:8501)
- Displays phase map visualizations with metadata ($\Lambda$, seed, current label status)
- Provides keyboard shortcuts and buttons for rapid classification
- Shows pattern category descriptions and examples
- Tracks labeling progress with statistics
- Auto-saves labels to `classification_dataset.csv` after each classification

**GUI Features**:
- **Navigation**: Previous/Next buttons, jump to specific sample, first/last
- **Filtering**: Show all, unlabeled only, labeled only, specific Λ range
- **Quick labeling**: Single-key shortcuts (t/s/m/y/d/u/o) or click buttons
- **Auto-advance**: Optionally jump to next unlabeled sample after labeling
- **Progress tracking**: Visual progress bar and pattern distribution statistics
- **Undo**: Revert incorrect labels

**Labeling Guidelines**:
Refer to pattern descriptions in the GUI or `docs/` for detailed criteria. In brief:
- **Target**: 1-3 circular/concentric pacemakers with organized wave fronts
- **Spiral**: Rotating spiral arms around phase singularities
- **Mixed**: Coexistence of both target and spiral patterns
- **Synchronized**: Uniform phase across lattice (no spatial structure)
- **Disordered**: Random, incoherent phase distribution
- **Multiple**: Many ($\geq 5$) competing pacemakers with fragmented waves
- **Other**: Transitional or ambiguous states

**Time estimate**: $\sim 0.5 - 1$ hr to label 500 samples (depending on experience)

### Step 3: Train Classifier

Train Random Forest on labeled data:
```bash
python main.py train
```

**What it does**:
1. Loads labeled training samples from `classification_dataset.csv`
2. Splits into 80% training / 20% test sets (stratified by class)
3. Extracts ~50 physics-informed features from each phase map:
   - Radial symmetry (5 features)
   - Spatial autocorrelation (8 features)
   - Fourier spectral properties (5 features)
   - Gradient statistics (7 features)
   - Topological characteristics (6 features)
   - Statistical moments (4 features)
4. Standardizes features (zero mean, unit variance) using StandardScaler
5. Trains Random Forest with 5-fold cross-validation
6. Evaluates on test set
7. Generates comprehensive evaluation metrics and figures:
   - Confusion matrices (normalized and raw counts)
   - Per-class precision/recall/F1-scores
   - Feature importance ranking
   - Prediction confidence distribution
   - Text summary report
8. Saves trained model, scaler, and metadata

**Output**:
```
outputs/models/
├── classifier.joblib           # Trained Random Forest model
├── scaler.joblib               # Feature StandardScaler
└── metadata.joblib             # Classes, feature names, config

outputs/figures/
├── confusion_matrices.pdf      # Normalized + raw confusion matrices
├── per_class_metrics.pdf       # Precision/recall/F1 bar charts
├── feature_importance.pdf      # Top 20 feature importance
├── confidence_distribution.pdf # Confidence histogram + CDF
└── evaluation_summary.txt      # Complete text report

Logged to console:
- Cross-validation accuracy (mean ± std)
- Test set accuracy
- Classification report (precision/recall/F1 per class)
```

**Time estimate**: ~2-5 minutes

### Step 4: Classify Full Dataset

Apply trained classifier to all unlabeled simulations:
```bash
python main.py classify
```

**What it does**:
1. Loads trained classifier and scaler from `outputs/models/`
2. Identifies unlabeled samples in `classification_dataset.csv`
3. Extracts features from each unlabeled phase map
4. Applies StandardScaler transformation
5. Predicts pattern type and confidence for each sample
6. Updates `classification_dataset.csv` with predictions:
   - Sets `pattern_type` to predicted class
   - Sets `labeled` to True
   - Sets `prediction_confidence` to max class probability
7. Saves updated dataset

**Output**:
```
data/processed/classification_dataset.csv (updated with predictions)
```

**Time estimate**: ~10-20 minutes for 4500 samples

### Step 5: Analyze Results

Generate pattern statistics and publication figures:
```bash
python main.py analyze
```

**What it does**:
1. Loads fully classified dataset
2. Calculates pattern fractions per $\Lambda$ value:
   - Groups samples by $\Lambda$
   - Computes fraction of each pattern type
   - Total sample count per $\Lambda$
3. Generates statistical summaries:
   - Target wave dominance regime ($\Lambda$ range where >50% target)
   - Peak target fraction and corresponding $\Lambda$
   - Transition points (onset/loss of dominance)
4. Creates publication-quality plots
5. Generates referee response text template
6. Creates classifier performance summary for manuscript

**Output**:
```
outputs/results/
├── pattern_fractions.csv           # Pattern statistics per Λ
├── statistical_summary.txt         # Key findings summary
├── referee_response.txt            # Template response text
└── classifier_summary.txt          # ML performance summary

outputs/figures/
└── pattern_fractions_vs_lambda.pdf # Main results figure
```

**Time estimate**: ~1-2 minutes

### Step 6: All-in-One Workflow

Run training, classification, and analysis sequentially:
```bash
python main.py all
```

**What it does**:
- Executes `train` → `classify` → `analyze` in sequence
- Useful after completing manual labeling
- Skips the labeling step (assumes training data is labeled)

**Warning**: This command will pause and ask for confirmation before proceeding, ensuring you've completed manual labeling.


Contact
Kaushik Roy
Email: kr70@rice.edu
GitHub: https://github.com/kaushik-roy-physics
