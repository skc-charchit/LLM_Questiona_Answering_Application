"""
Alternative entry point for the Streamlit app
"""

import streamlit as st
from src.app import PDFQAApp

# Run the app
app = PDFQAApp()
app.run()
