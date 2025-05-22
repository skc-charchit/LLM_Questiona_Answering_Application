"""
Main entry point for the PDF Question Answering Application.
This file imports and runs the Streamlit application.
"""

import os
import sys
import warnings

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """Run the PDF QA application."""
    # Import here to ensure the path is set up correctly
    from src.app import PDFQAApp
    app = PDFQAApp()
    app.run()

if __name__ == "__main__":
    main()
