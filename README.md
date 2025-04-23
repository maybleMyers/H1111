To use this branch  

git clone https://github.com/maybleMyers/H1111 kohya-framepack -b kohya-framepack  
python -m venv env  or if you have multiple python snakes:  
python3.10 -m venv env   
env/scripts/activate  

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
pip install -r requirementsFP.txt  
python h1111.py  

download these 5 files from https://huggingface.co/maybleMyers/framepack_h1111 and put them in your hunyuan folder, or reference where they are in the gui if you have already aquired them.  

FramePackI2V_HY_bf16.safetensors  

clip_l.safetensors  

llava_llama3_fp16.safetensors  

model.safetensors  

pytorch_model.pt  

Lora might work? It is too different of a model for them to work well.  Use a lot of block swap this is a different back end than the official repo.   