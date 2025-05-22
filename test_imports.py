"""
Test script to check imports
"""

import sys
import os

# Print Python path
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nCurrent directory:", os.getcwd())

try:
    print("\nTrying to import from src.tasks.loader...")
    from src.tasks.loader import DocumentLoader
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nTrying to import from src.tasks.embeddings...")
    from src.tasks.embeddings import EmbeddingManager
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nTrying to import from src.pipelines.qa_chain...")
    from src.pipelines.qa_chain import QAChain
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\nTrying to import from src.app...")
    from src.app import PDFQAApp
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
