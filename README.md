![GUI Screenshot](images/screenshot.png)  

To use Pusa, install h1111 as normal,  
activate your venv then
goto pusa/PusaV1 and run pip install -e .

download pusa loras and put them pusa/pusa_lora You need high and low noise models official versions. To make other wan2.2 lora's work convert them to the other format and place them in the pusa/pusa_lora subfolder.  
Here are the pusa loras: https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1/tree/main  
This implementation is using the diffsynth backend, not musubi. You need to adjust persistent parameters to your gpu. for 480p 10.6 works on a 32gb gpu.  
I have only tested it with wan22_t2v_14B_high_noise_bf16.safetensors and  wan22_t2v_14B_low_noise_bf16.safetensors from https://huggingface.co/maybleMyers/wan_files_for_h1111/ you should probably use the bf16 models for compatibility.  

Thanks to these people for making pusa possible:

```bibtex
@article{liu2025pusa,
  title={PUSA V1. 0: Surpassing Wan-I2V with $500 Training Cost by Vectorized Timestep Adaptation},
  author={Liu, Yaofang and Ren, Yumeng and Artola, Aitor and Hu, Yuxuan and Cun, Xiaodong and Zhao, Xiaotong and Zhao, Alan and Chan, Raymond H and Zhang, Suiyun and Liu, Rui and others},
  journal={arXiv preprint arXiv:2507.16116},
  year={2025}
}

@misc{Liu2025pusa,
  title={Pusa: Thousands Timesteps Video Diffusion Model},
  author={Yaofang Liu and Rui Liu},
  year={2025},
  url={https://github.com/Yaofang-Liu/Pusa-VidGen},
}

@article{liu2024redefining,
  title={Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach},
  author={Liu, Yaofang and Ren, Yumeng and Cun, Xiaodong and Artola, Aitor and Liu, Yang and Zeng, Tieyong and Chan, Raymond H and Morel, Jean-michel},
  journal={arXiv preprint arXiv:2410.03160},
  year={2024}
}