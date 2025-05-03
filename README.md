![GUI Screenshot](images/screenshot.png)

# H1111



This is a GUI for tech wizard kohya-ss's musubi tuner's inference script.
https://github.com/kohya-ss/musubi-tuner

It allows inference with these models:  
FramePack  
Hunyuan-t2v  
Hunyuan-i2v  
Hunyuan-v2v  
WanX-t2v  
WanX-i2v  
WanX-v2v  
SkyReels-i2v  
SkyReels-t2v  

I have mostly been workiing on the framepack tab and the WanX-i2v tab. They are the best to use right now. WanX-i2v is used for skyreels v2 and the fun control models.    

If you are running out of vram use block swapping and some form of attention besides sdpa or torch and use split attention. Sage attention is the fastest/lowest vram but difficult to install in windows. I would say the easiest to get to run is xformers attention, you can usually get it with "pip install xformers".

Best quality will be obtained with only enabling block swapping and using the fp16 model with sdpa attention. You can speed things up with cfg skip, fp8 scaled, slg skip is small speedup, sage attention is fastest but all speedups come with quality degradations. I designed this to try to focus on quality over speed.

If you are using a lora that you didn't train with musubi you need to drag it to the convert lora tab and convert it to the default format. It should spit it out into the /lora folder.

If you need additional installation instructions or information create an issue and I will try to help. Also there are alot of settings notes on the musubi github linked above.  

For torch 2.7.0 and windows installation try:  
pip install typing-extensions  
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128  
pip install -r requirementsTorch27.txt  

## To Use FramePack

Install as normal, then install the frame pack requirements in requirementsFP.txt.  
pip install -r requirementsFP.txt  

download these 5 files from https://huggingface.co/maybleMyers/framepack_h1111 and put them in a subfolder named hunyuan (H1111/hunyuan), or reference where they are in the gui if you have already aquired them.  

FramePackI2V_HY_bf16.safetensors  

clip_l.safetensors  

llava_llama3_fp16.safetensors  

model.safetensors  

pytorch_model.pt  

Lora trained with musubi tuner's framepack training confirmed to work great. Normal lora trained for hunyuan kinda suck. Use a lot of block swap this is a different back end than the official repo. If you select fp8 and fp8 scaled it will all fit on a 24gb gpu for fastest speed, about 3s/it or 1:17 per second of video w/ a 4090.  Best quality will be obtained with just block swapping/sdpa attention/full model though.  

Put loras in a /lora subfolder, if not trained with musubi you need to convert them.  

Here is an example prompt for a 5 second video with 4 sections using sectional prompting, also supports longer videos with indexes ie 0-2  ;;;3-5 etc:  

0:A cinematic video showcases a cute blue penguin wearing sunglasses. The penguin runs quickly into mcdonalds.;;;1:The penguin runs quickly into mcdonalds and jumps up on a table and starts eating his food. The penguin's name is Piplup he is a famous Pokemon actor. The video is a fast action sequence animation showing the penguin running into a mcdonalds an jumping up onto a table.;;;2:The penguin is seated at a table and is enjoying his happy meal. The penguin's name is Piplup he is a famous Pokemon actor. The video is a fast action sequence animation showing the penguin running into a mcdonalds and jumping up onto a table.;;;3:The penguin is seated at a table and is happily enjoying his happy meal. The penguin's name is Piplup he is a famous Pokemon actor. The penguin flexes his huge arm muscles at the end of the video.  

I have added support for 4 sectional images during inference. It works best when the images are close together. Refer to the screen shot for an example of a working 5 second video.  

For more details on using framepack with musubi go here https://github.com/kohya-ss/musubi-tuner/blob/main/docs/framepack.md  

## To Use the new Skyreels-V2 models

I have provided these 2 at https://huggingface.co/maybleMyers/wan_files_for_h1111  

 SkyReels-V2-I2V-14B-720P-FP16.safetensors  
 SkyReels-V2-I2V-14B-540P-FP16.safetensors  

You can just drop them into the wan folder and use them in the WanX-i2v tab. Skyreels-V2 is a fine tune from Wan2.1.  
If you have download the kijai variants the will not work because he added extra keys to the model.  

## To Use WanX

To use wanX download these and toss them in the wan subfolder:
Download the T5 `models_t5_umt5-xxl-enc-bf16.pth`, vae `Wan2.1_VAE.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` from the following page: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main    

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models  
ie : wan2.1_i2v_720p_14B_fp16.safetensors  

For the fun control option in WanX-i2v I recommend the fp16 weights here: https://huggingface.co/maybleMyers/wan_files_for_h1111/tree/main  
Wan2.1-Fun-14B-Control_fp16.safetensors  

git pull to update the installation
pip install -r requirements.txt

I have tested the 14B i2v and t2v models so far to be working

## changlog
5/3/2025  
    Add support for Wan2.1 i2v-14B-FC-1.1. It is a fun control model and is very good. Use it in the WanX-i2v tab and make sure to select the task i2v-14B-FC-1.1 at the bottom of the page.  Download the weights from https://huggingface.co/maybleMyers/wan_files_for_h1111  
4/30/2025  
    Previews for framepack.    
4/29/2025  
    Add initial preview support to the wanX-i2v tab based. If you want to use them use the preview branch. Thanks to Sarania.   
    Wan2.1-Fun-V1.1-14B-InP-FP16.safetensors is available at https://huggingface.co/maybleMyers/wan_files_for_h1111  
    Fix bug in hunyuan-t2v not loading lora.  
4/26/2025  
    Add SkyReels-V2-I2V-14B-720P-FP16.safetensors to supported models.  
    Added alot better options for Framepack including working sectional images, Thanks to kohya!  
4/25/2025  
    Framepack backend updates for better LoRa support for LoRa's trained with musubi tuner. Also better weighting options.  
4/24/2025  
    Update FramePack backend to musubi backend instead of original. Offers much improved speed and some quality improvements.  
    Add support for torch 2.7.0 + cuda 12.8  
4/18/2025  
    Add initial support for FramePack. https://github.com/lllyasviel/FramePack  
4/15/2025  
    Add much improved functionality for the wan fun control model. Added strength imrpovements and dropoff code to choose when to apply the control video. Thanks wordbrew.  
4/3/2025  
    Add support for hunyuan i2v model. Download the clip vision from https://huggingface.co/maybleMyers/H1111_Hunyuan_i2v And download the official model from hunyuan's website and rename it to mp_rank_00_model_states_i2v.pt https://huggingface.co/tencent/HunyuanVideo-I2V/tree/main/hunyuan-video-i2v-720p/transformers  add both to your hunyuan folder.  
3/29/2025  
    Added support for fun models! download dit from https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control and specify correct task type and dit location. I renamed it from diffusion_pytorch_model to Wan2.1-Fun-14B-control. Works in the normal WanX-i2v tab when you select the control option at the bottom of the page.  
3/23/2025  
    Added Wanx cfg skip functionality to skip cfg guidance during inference for faster generations but less following of the prompt  
3/22/2025  
    Added WanX-i2v end frame functionality  
3/20/2025  
    Added WanX-v2v functionality.  
3/18/2025  
    Added Skip Layer Guidance for WanX-i2v.  
3/13/2025  
    Added extend video functionality to WanX-i2v. It kind of works .  
3/12/2025  
    Added ability to send the last frame of a video to the input in WanX-i2v. Also you can now use this to extend the video. You can do multiple batches at each step and pick the best extended video then generate an even longer one.  
3/9/2025  
    Added batching ability for a folder full of images in WanX-i2v tab. Added flash attn for windows prebuilt wheel.  
3/8/2025  
    Added support for wan lora's. Remember to convert them first in the convert lora tab.  
3/5/2025  
    Added ability to batch a folder of images with skyreels i2v, so you can make a video with every image in a folder.
3/2/2025  
    Added initial support for wanX-2.1 Image to Video and Text to Video inference.  
3/1/2025  
    Added support for Skyreels Video to Video and Text to Video.   
2/23/2025  
    Added initial support for skyreels-V1 using musubi's skyreel implementation. (thanks  sdbds)
download models from https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy and add them to your hunyuan folder
skyreels_hunyuan_i2v_bf16.safetensors
skyreels_hunyuan_t2v_bf16.safetensors

## Requirements

- Python 3.10
- CUDA 12.4

## Basic Installation (Linux)

Tested on ubuntu 24

to update navigate to H1111 and git pull

```powershell
git clone https://github.com/maybleMyers/H1111
cd H1111
python -m venv env
#(if you have another version of python do python3.10 -m venv env after you install it with sudo apt install python3.10 python3.10-venv python3.10-distutils)
source env/bin/activate 
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install sageattention==1.0.6
might need python3.10-dev as well for sage attention to work

```

run with  
source env/bin/activate  
python h1111.py

for GPU1
CUDA_VISIBLE_DEVICES=1 python h1111.py

## Basic Installation (Windows)



First, open PowerShell and navigate to your desired installation directory. Then run these commands:

```powershell
git clone https://github.com/maybleMyers/H1111
cd H1111
python -m venv env
./env/scripts/activate
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt

```

## To run

```
env/scripts/activate
python h1111.py
```

open 127.0.0.1:7860 in a browser

You can set cuda device to 1,2,3,4,5,6,7 etc in the env once activated in a separate terminal to run unlimited copies at once if you have another gpu.

## to use stock hunyuan models

https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt  

https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt  

https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp16.safetensors  

https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors  

#fp8 dit model

https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial/resolve/main/mp_rank_00_model_states_fp8.safetensors  

place models in H1111/hunyuan folder

### Optional: Install Xformers
```powershell
pip install --no-deps xformers --index-url https://download.pytorch.org/whl/cu124
```

### Optional: Install Flash Attention
Note: This can take 1-5 hour to install even on a good CPU, but provides faster generation.  
I have uploaded a wheel for windows users to match cuda 12.4 and python 3.10.(thanks lldacing)
https://huggingface.co/maybleMyers/wan_files_for_h1111/resolve/main/flash_attn-2.7.4%2Bcu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl?download=true  

```powershell
pip install flash-attn --no-build-isolation

If you have downloaded the wheel you can install it with:

pip install "flash_attn-2.7.4+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"
```
```
