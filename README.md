This is a very early release and there are many bugs I am still working on


![GUI Screenshot](images/screenshot.png)

# H1111

This is a GUI for tech wizard kohya-ss's musubi tuner's inference script.

## Requirements
- Python 3.10
- CUDA 12.4

## Basic Installation (Windows)

First, open PowerShell and navigate to your desired installation directory. Then run these commands:

```powershell
git clone https://github.com/maybleMyers/H1111
cd musubi-tuner
python -m venv env
./env/scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt
pip install ascii-magic matplotlib tensorboard wheel gradio tiktoken
```

### Optional: Install Xformers
```powershell
pip install --no-deps xformers --index-url https://download.pytorch.org/whl/cu124
```

### Optional: Install Flash Attention
Note: This can take 1+ hour to install even on a good CPU, but provides faster generation.
```powershell
pip install flash-attn --no-build-isolation
```
```