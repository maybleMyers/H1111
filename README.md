This branch is in active development, so could break or change alot.  
You can find all of the files needed for wan2.2 at https://huggingface.co/maybleMyers/wan_files_for_h1111/tree/main  
so far all models work. you can use the t2v models by not using an input image in the wan2.2 tab.  

This script is designed to allow inference with the same quality as the source https://github.com/Wan-Video/Wan2.2 .  
Most of the changes are to provide better memory optimization. This repository is built for quality not speed. With a 4090 generating a 720p video will take around 1 hr.    

Put all the models in the wan subfolder. To run the 14B-i2v model you need these:  
wan/wan22_i2v_14B_low_noise_bf16.safetensors  
wan/wan22_i2v_14B_high_noise_bf16.safetensors  
wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  
wan/Wan2.1_VAE.pth  
wan/models_t5_umt5-xxl-enc-bf16.pth  

You need to install some extra packages probably:
pip install imageio librosa pyloudnorm soundfile xfuser

v2v seems to be working well now with the t2v model.
Most warnings are safe to ignore and will probably be removed later.

A cool thing to try are the mixed weight models, I preserved all the weights that would not safely convert to fp16 in fp32 without increasing model size very much. Select preserve mixed weight dtype at the bottom of the page, download the models named like this: wan22_i2v_14B_high_noise_fp32_and_fp16.safetensors.

If you want to use infinitetalk you need to follow the installation instructions on their github to get it to work and download the same models you use for multitalk. https://github.com/MeiGen-AI/InfiniteTalk . But in my tests multitalk is way better than infinitetalk.  

## Changlog
9/8/2025  
    Added support for video extension, wan one frame support, infinitetalk support, context windows for wan 2.2, video extension for wan 2.2.  
