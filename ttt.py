import quimb as qt
import os
import sys

print(f"Python Executable: {sys.executable}")
print(f"Quimb Version: {qt.__version__}")
print(f"Quimb Loaded from: {os.path.dirname(qt.__file__)}")
