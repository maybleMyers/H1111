

### Lightx2v Support

This branch adds support for the lightx2v loras. https://huggingface.co/lightx2v/Wan2.2-Lightning is their repo, thanks to the lightx2v team.  

Download normal wan2.2 models, ie:  
https://huggingface.co/maybleMyers/wan_files_for_h1111/blob/main/wan22_i2v_14B_low_noise_bf16.safetensors  
https://huggingface.co/maybleMyers/wan_files_for_h1111/blob/main/wan22_i2v_14B_high_noise_bf16.safetensors  
https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/blob/main/Wan2.1_VAE.pth  
https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/blob/main/models_t5_umt5-xxl-enc-bf16.pth  
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/blob/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth  

And put them in the wan subfolder. If you already have them you could try inputting the location into the gui, it might work.  

I have verified that these loras work with the i2v models so far.

https://huggingface.co/maybleMyers/wan_files_for_h1111/blob/main/Wan2.2-T2V-A14B-4steps-lora-250928_high_noise_model_MUSUBI.safetensors  
https://huggingface.co/maybleMyers/wan_files_for_h1111/blob/main/Wan2.2-T2V-A14B-4steps-lora-250928_low_noise_model_MUSUBI.safetensors  

This is the old lora, use it for high and low noise.  

https://huggingface.co/maybleMyers/wan_files_for_h1111/blob/main/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64_MUSUBI.safetensors  

Use them in the wan2.2 tab and set sampling steps to like 4-8, Guidance Scale to 1 and step_distill as the Sample Solver.  
It seems like higher weights work better, 1.4-2 seems good.  
put the loras in the lora subfolder.