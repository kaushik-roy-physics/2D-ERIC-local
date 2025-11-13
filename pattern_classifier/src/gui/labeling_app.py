"""
Streamlit-based labeling interface for phase map classification.

Run with: streamlit run src/gui/labeling_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_manager import DataManager

class LabelingApp:
    """Streamlit app for manual pattern labeling"""
    
    def __init__(self):
        """Initialize labeling application"""
        st.set_page_config(
            page_title="Pattern Labeling Tool",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize data manager
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        
        self.data_manager = st.session_state.data_manager
        
        # Load configuration
        self.pattern_categories = self.data_manager.config['patterns']['categories']
        self.shortcuts = self.data_manager.config['patterns']['shortcuts']
        
        # Initialize session state variables
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        # Current position
        if 'current_idx' not in st.session_state:
            st.session_state.current_idx = 0
        
        # Load dataset
        if 'df' not in st.session_state:
            st.session_state.df = self.data_manager.load_classification_dataset()
        
        # Training samples list
        if 'training_samples' not in st.session_state:
            df = st.session_state.df
            st.session_state.training_samples = df[df['is_training'] == True].index.tolist()
        
        # Filter state
        if 'filter_type' not in st.session_state:
            st.session_state.filter_type = 'all'
        
        # Navigation triggers
        if 'nav_action' not in st.session_state:
            st.session_state.nav_action = None
        
        # Auto-advance setting
        if 'auto_advance' not in st.session_state:
            st.session_state.auto_advance = True
    
    def run(self):
        """Main application loop"""
        st.title("🔬 Pattern Classification Labeling Tool")
        st.markdown("---")
        
        # Process any pending navigation actions
        self._process_navigation()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_image_panel()
        
        with col2:
            self._render_labeling_panel()
        
        # Navigation
        self._render_navigation()
    
    def _process_navigation(self):
        """Process any pending navigation actions"""
        if st.session_state.nav_action is not None:
            action = st.session_state.nav_action
            
            if action == 'next':
                self._next_sample()
            elif action == 'previous':
                self._previous_sample()
            elif action == 'first':
                st.session_state.current_idx = 0
            elif action == 'last':
                st.session_state.current_idx = len(st.session_state.training_samples) - 1
            elif isinstance(action, int):
                st.session_state.current_idx = action
            
            # Clear the action
            st.session_state.nav_action = None
    
    def _render_sidebar(self):
        """Render sidebar with statistics and controls"""
        st.sidebar.header("📊 Progress")
        
        stats = self.data_manager.get_training_statistics()
        
        # Progress bar
        progress = stats['progress_pct'] / 100
        st.sidebar.progress(progress)
        st.sidebar.metric("Labeled", f"{stats['labeled']}/{stats['total_training']}")
        st.sidebar.metric("Remaining", stats['unlabeled'])
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        st.session_state.auto_advance = st.sidebar.checkbox(
            "Auto-advance after labeling",
            value=st.session_state.auto_advance
        )
        
        st.sidebar.markdown("---")
        
        # Pattern distribution
        st.sidebar.subheader("Pattern Distribution")
        if stats['pattern_counts']:
            for pattern, count in stats['pattern_counts'].items():
                st.sidebar.text(f"{pattern}: {count}")
        else:
            st.sidebar.info("No labels yet")
        
        st.sidebar.markdown("---")
        
        # Shortcuts reference
        st.sidebar.subheader("⌨️ Keyboard Shortcuts")
        st.sidebar.text("Use Quick Label buttons:")
        for pattern, shortcut in self.shortcuts.items():
            st.sidebar.code(f"{shortcut.upper()} → {pattern.capitalize()}")
        
        st.sidebar.markdown("---")
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        
        filter_option = st.sidebar.selectbox(
            "Show",
            ["All training samples", "Unlabeled only", "Labeled only", 
             "Specific Lambda range"],
            key='filter_selector'
        )
        
        # Apply filter if changed
        if filter_option == "Specific Lambda range":
            df = st.session_state.df
            lambda_min = float(df['Lambda'].min())
            lambda_max = float(df['Lambda'].max())
            
            lambda_range = st.sidebar.slider(
                "Lambda range",
                lambda_min, lambda_max,
                (lambda_min, lambda_max),
                step=0.1,
                key='lambda_filter'
            )
            
            if st.sidebar.button("Apply Lambda Filter"):
                self._apply_lambda_filter(lambda_range)
                st.rerun()
                
        elif filter_option == "Unlabeled only":
            if st.session_state.filter_type != 'unlabeled':
                self._filter_unlabeled()
                st.session_state.filter_type = 'unlabeled'
                st.rerun()
                
        elif filter_option == "Labeled only":
            if st.session_state.filter_type != 'labeled':
                self._filter_labeled()
                st.session_state.filter_type = 'labeled'
                st.rerun()
        else:
            if st.session_state.filter_type != 'all':
                self._reset_filter()
                st.session_state.filter_type = 'all'
                st.rerun()
        
        # Export
        st.sidebar.markdown("---")
        if st.sidebar.button("💾 Save Progress", width='stretch'):
            self._save_progress()
            st.sidebar.success("✓ Progress saved!")

    def _render_image_panel(self):
        """Render image display panel"""
        st.subheader("Phase Map Visualization")
        
        # Validate current index
        if st.session_state.current_idx >= len(st.session_state.training_samples):
            st.session_state.current_idx = len(st.session_state.training_samples) - 1
        
        if st.session_state.current_idx < 0:
            st.session_state.current_idx = 0
        
        # Get current sample
        training_idx = st.session_state.training_samples[st.session_state.current_idx]
        row = st.session_state.df.iloc[training_idx]
        
        # Load and display image with reduced size
        img_path = self.data_manager.get_image_path(row['filename'])
        
        if img_path.exists():
            img = Image.open(img_path)
            # Display with width=500 to reduce pixelation
            st.image(img, width=500)
        else:
            st.error(f"Image not found: {img_path}")
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        col1.metric("Lambda (Λ)", f"{row['Lambda']:.2f}")
        col2.metric("Seed", row['seed'])
        
        status = "✓ Labeled" if row['labeled'] else "○ Unlabeled"
        status_color = "green" if row['labeled'] else "orange"
        col3.markdown(f"**Status:** :{status_color}[{status}]")

    
    def _render_labeling_panel(self):
        """Render labeling controls"""
        st.subheader("Classification")
        
        training_idx = st.session_state.training_samples[st.session_state.current_idx]
        row = st.session_state.df.iloc[training_idx]
        
        # Current label
        if row['labeled']:
            current_label = row['pattern_type']
            st.success(f"Current: **{current_label.upper()}**")
        else:
            st.warning("**Unlabeled**")
        
        st.markdown("---")
        
        # Pattern descriptions
        st.markdown("### Pattern Categories")

        descriptions = {
            'target': """🎯 **Target/Concentric**
            - ONE dominant pacemaker (or 2-3 well-separated ones)
            - Clear circular/concentric waves radiating from center(s)
            - Organized, smooth wave fronts
            - High spatial coherence
            - Example: Bull's-eye pattern""",
            
            'spiral': """🌀 **Spiral**
            - Rotating spiral arms around topological defect(s)
            - Clear sense of rotation
            - Typically 1-2 spiral cores
            - Arms wind outward from center
            - Example: Galaxy-like pattern""",
            
            'mixed': """🔄 **Mixed**
            - Coexistence of BOTH target and spiral patterns
            - Multiple pattern types in same domain
            - Spatial competition between patterns
            - May show target waves with nearby spirals
            - Example: Patchwork of targets and spirals""",
            
            'sync': """〰️ **Synchronized**
            - Uniform phase across entire lattice
            - No spatial structure or gradients
            - All oscillators in phase
            - Appears as single color/shade
            - Example: Flat, homogeneous pattern""",
            
            'disorder': """📊 **Disordered**
            - No coherent spatial structure
            - Random, incoherent phase distribution
            - No organized waves or patterns
            - High spatial heterogeneity
            - Example: Static/noise-like""",
            
            'multiple': """⚡ **Multiple Sources**
            - MANY competing pacemakers (5+ sources)
            - Fragmented wave patterns
            - Wave collisions/interference visible
            - Irregular, patchy boundaries
            - Low overall organization
            - Example: Tessellated/mosaic pattern""",
            
            'other': """❓ **Other**
            - Transitional states between patterns
            - Unclear or ambiguous patterns
            - Mixed pattern types that don't fit 'mixed'
            - Use when none of the above clearly fits"""
        }
        
        with st.expander("Show pattern descriptions", expanded=False):
            for pattern in self.pattern_categories:
                st.markdown(descriptions.get(pattern, pattern))
        
        st.markdown("---")
        
        # Quick label buttons
        st.markdown("### Quick Label")
        
        # Create grid of buttons
        for i in range(0, len(self.pattern_categories), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(self.pattern_categories):
                    pattern = self.pattern_categories[i + j]
                    shortcut = self.shortcuts.get(pattern, '')
                    
                    button_label = f"{pattern.upper()}"
                    if shortcut:
                        button_label += f" ({shortcut.upper()})"
                    
                    if cols[j].button(
                        button_label,
                        key=f"label_btn_{pattern}",
                        width='stretch'
                    ):
                        self._label_current(pattern)
                        if st.session_state.auto_advance:
                            self._advance_to_next_unlabeled()
                        st.rerun()
        
        st.markdown("---")
        
        # Undo button
        if row['labeled']:
            if st.button("🔄 Undo Label", width='stretch'):
                self._undo_label(training_idx)
                st.rerun()
    
    def _render_navigation(self):
        """Render navigation controls"""
        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First", width='stretch', key="btn_first"):
                st.session_state.nav_action = 'first'
                st.rerun()
        
        with col2:
            if st.button("◀️ Previous", width='stretch', key="btn_prev"):
                st.session_state.nav_action = 'previous'
                st.rerun()
        
        with col3:
            # Jump to specific sample
            max_idx = len(st.session_state.training_samples) - 1
            
            # Use a form to prevent auto-rerun on input change
            with st.form(key='jump_form'):
                jump_to = st.number_input(
                    "Jump to sample",
                    min_value=0,
                    max_value=max_idx,
                    value=st.session_state.current_idx,
                    step=1
                )
                
                if st.form_submit_button("Go"):
                    st.session_state.nav_action = jump_to
                    st.rerun()
        
        with col4:
            if st.button("Next ▶️", width='stretch', key="btn_next"):
                st.session_state.nav_action = 'next'
                st.rerun()
        
        with col5:
            if st.button("Last ⏭️", width='stretch', key="btn_last"):
                st.session_state.nav_action = 'last'
                st.rerun()
        
        # Position indicator
        current = st.session_state.current_idx + 1
        total = len(st.session_state.training_samples)
        
        # Show current Lambda and seed
        training_idx = st.session_state.training_samples[st.session_state.current_idx]
        row = st.session_state.df.iloc[training_idx]
        
        st.markdown(
            f"<center><b>Sample {current} of {total}</b> | "
            f"Λ={row['Lambda']:.2f}, seed={row['seed']}</center>",
            unsafe_allow_html=True
        )
    
    def _label_current(self, pattern: str):
        """Label current sample"""
        training_idx = st.session_state.training_samples[st.session_state.current_idx]
        st.session_state.df.at[training_idx, 'pattern_type'] = pattern
        st.session_state.df.at[training_idx, 'labeled'] = True
        self._save_progress()
    
    def _undo_label(self, idx: int):
        """Undo label for a sample"""
        st.session_state.df.at[idx, 'pattern_type'] = ''
        st.session_state.df.at[idx, 'labeled'] = False
        self._save_progress()
    
    def _next_sample(self):
        """Navigate to next sample"""
        max_idx = len(st.session_state.training_samples) - 1
        if st.session_state.current_idx < max_idx:
            st.session_state.current_idx += 1
    
    def _previous_sample(self):
        """Navigate to previous sample"""
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
    
    def _advance_to_next_unlabeled(self):
        """Advance to next unlabeled sample automatically"""
        max_idx = len(st.session_state.training_samples) - 1
        current_idx = st.session_state.current_idx
        
        # Try to find next unlabeled sample
        for i in range(current_idx + 1, len(st.session_state.training_samples)):
            training_idx = st.session_state.training_samples[i]
            if not st.session_state.df.iloc[training_idx]['labeled']:
                st.session_state.current_idx = i
                return
        
        # If no unlabeled found ahead, wrap around
        for i in range(0, current_idx):
            training_idx = st.session_state.training_samples[i]
            if not st.session_state.df.iloc[training_idx]['labeled']:
                st.session_state.current_idx = i
                return
        
        # If all labeled, just go to next
        if current_idx < max_idx:
            st.session_state.current_idx += 1
    
    def _save_progress(self):
        """Save labeling progress"""
        self.data_manager.save_classification_dataset(st.session_state.df)
    
    def _apply_lambda_filter(self, lambda_range: tuple):
        """Filter samples by Lambda range"""
        df = st.session_state.df
        training = df[df['is_training'] == True]
        
        filtered = training[
            (training['Lambda'] >= lambda_range[0]) &
            (training['Lambda'] <= lambda_range[1])
        ]
        
        st.session_state.training_samples = filtered.index.tolist()
        st.session_state.current_idx = 0
    
    def _filter_unlabeled(self):
        """Show only unlabeled samples"""
        df = st.session_state.df
        training = df[df['is_training'] == True]
        unlabeled = training[~training['labeled']]
        
        st.session_state.training_samples = unlabeled.index.tolist()
        st.session_state.current_idx = 0
    
    def _filter_labeled(self):
        """Show only labeled samples"""
        df = st.session_state.df
        training = df[df['is_training'] == True]
        labeled = training[training['labeled']]
        
        st.session_state.training_samples = labeled.index.tolist()
        st.session_state.current_idx = 0
    
    def _reset_filter(self):
        """Reset to show all training samples"""
        df = st.session_state.df
        training = df[df['is_training'] == True]
        
        st.session_state.training_samples = training.index.tolist()
        st.session_state.current_idx = 0


def main():
    """Main entry point"""
    app = LabelingApp()
    app.run()


if __name__ == "__main__":
    main()