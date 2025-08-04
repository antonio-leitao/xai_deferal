import os
import sys

 # Add the 'defer' directory to the Python path dynamically
defer_path = os.path.join(os.path.dirname(__file__), "defer")
sys.path.insert(0, defer_path)