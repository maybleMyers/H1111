![GUI Screenshot](images/screenshot.png)  

To use Pusa, install h1111 as normal,  
activate your venv then
goto pusa/PusaV1 and run pip install -e .

download pusa loras and put them pusa/pusa_lora You need high and low noise models official versions. To make other wan2.2 lora's work convert them to the other format and place them in the pusa/pusa_lora subfolder.  
Here are the pusa loras: https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1/tree/main  
This implementation is using the diffsynth backend, not musubi. You need to adjust persistent parameters to your gpu. for 480p 10.6 works on a 32gb gpu.  
I have only tested it with wan22_t2v_14B_high_noise_bf16.safetensors and  wan22_t2v_14B_low_noise_bf16.safetensors from https://huggingface.co/maybleMyers/wan_files_for_h1111/ you should probably use the bf16 models for compatibility.