To use Pusa
activate your venv then
goto pusa/PusaV1 and run pip install -e .

download pusa lora and put them pusa/pusa_lora You need high and low noise models official versions. To make other wan2.2 lora's work convert them to the other format and place them in the pusa/pusa_lora subfolder.  
Here are the pusa loras: https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1/tree/main  
This implementation is using the diffsynth backend, not musubi. You need to adjust persistent parameters to your gpu. for 480p 10.6 works on a 32gb gpu.  