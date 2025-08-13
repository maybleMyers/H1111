#!/usr/bin/env python3
"""
Installation script for VideoX-Fun within H1111 directory structure.
This script properly configures the VideoX-Fun package for use without setuptools issues.
"""

import os
import sys
import subprocess
import site

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True

def main():
    # Get the current directory (H1111)
    h1111_dir = os.path.dirname(os.path.abspath(__file__))
    videox_fun_dir = os.path.join(h1111_dir, "wan", "origin-VideoX-Fun")
    
    print(f"H1111 directory: {h1111_dir}")
    print(f"VideoX-Fun directory: {videox_fun_dir}")
    
    if not os.path.exists(videox_fun_dir):
        print(f"Error: VideoX-Fun directory not found at {videox_fun_dir}")
        return False
    
    if not os.path.exists(os.path.join(videox_fun_dir, "videox_fun")):
        print(f"Error: videox_fun package not found in {videox_fun_dir}")
        return False
    
    # Install requirements from VideoX-Fun
    requirements_file = os.path.join(videox_fun_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        print("Installing VideoX-Fun requirements...")
        if not run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file]):
            print("Warning: Some requirements may have failed to install")
    
    # Create a .pth file to add VideoX-Fun to Python path
    site_packages = site.getsitepackages()[0]
    pth_file = os.path.join(site_packages, "videox-fun.pth")
    
    print(f"Creating .pth file: {pth_file}")
    try:
        with open(pth_file, 'w') as f:
            f.write(f"{videox_fun_dir}\n")
        print("Successfully created .pth file")
    except Exception as e:
        print(f"Error creating .pth file: {e}")
        print("Trying alternative approach...")
        
        # Alternative: Add to PYTHONPATH via environment
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if videox_fun_dir not in current_pythonpath:
            new_pythonpath = f"{videox_fun_dir}:{current_pythonpath}" if current_pythonpath else videox_fun_dir
            print(f"Add this to your shell configuration:")
            print(f"export PYTHONPATH=\"{new_pythonpath}\"")
    
    # Test the installation
    print("Testing installation...")
    test_cmd = [sys.executable, "-c", "import videox_fun.dist; print('VideoX-Fun installed successfully!')"]
    if run_command(test_cmd, cwd=videox_fun_dir):
        print("✅ VideoX-Fun installation successful!")
        return True
    else:
        print("❌ VideoX-Fun installation failed!")
        print("Manual steps:")
        print(f"1. cd {videox_fun_dir}")
        print(f"2. Add {videox_fun_dir} to your PYTHONPATH")
        print(f"3. Test with: python -c 'import videox_fun.dist'")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)