![GUI Screenshot](images/screenshot.png)

#H1111

This is a gui for tech wizard kohya-ss's musubi tuner's inference script. It is styled after AUTOMATIC1111 for sdxl

This is only verified to work with python 3.10 and cuda 12.4 

Basic installation is as follows for windows using powershell:

cd H1111
python -m venv env
env/scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt
pip install ascii-magic matplotlib tensorboard wheel gradio tiktoken

if you want to use xformers:
pip install --no-deps xformers --index-url https://download.pytorch.org/whl/cu124

if you want to use flash attention:
pip install flash-attn --no-build-isolation
This takes a long time to install, like 1+ hr on a good cpu.

----

This is a very early release and there are many bugs I am still working on.