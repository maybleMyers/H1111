#!/usr/bin/env python3
"""
Environment setup script for VideoX-Fun.
This ensures the videox_fun package can be imported correctly.
"""

import os
import sys

def setup_videox_fun_path():
    """Add VideoX-Fun to Python path if not already present"""
    # Get the VideoX-Fun root directory (where this script is located)
    videox_fun_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add to sys.path if not already present
    if videox_fun_root not in sys.path:
        sys.path.insert(0, videox_fun_root)
        print(f"Added {videox_fun_root} to Python path")
    
    # Verify we can import the module
    try:
        import videox_fun.dist
        print("✅ videox_fun.dist imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import videox_fun.dist: {e}")
        return False

if __name__ == "__main__":
    success = setup_videox_fun_path()
    sys.exit(0 if success else 1)