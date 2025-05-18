# --- START OF FILE f1_video_cli_local_extension.py ---

import os
import torch
import traceback
import einops
import numpy as np
import argparse
import math
import decord
from tqdm import tqdm
import pathlib
from datetime import datetime
import imageio_ffmpeg
import tempfile
import shutil
import subprocess
import sys

from PIL import Image

# --- Imports from fpack_generate_video.py's ecosystem ---
from frame_pack.hunyuan_video_packed import load_packed_model
from frame_pack.framepack_utils import (
    load_vae,
    load_text_encoder1,
    load_text_encoder2,
    load_image_encoders
)
from frame_pack.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from frame_pack.utils import crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from frame_pack.k_diffusion_hunyuan import sample_hunyuan
from frame_pack.clip_vision import hf_clip_vision_encode
from frame_pack.bucket_tools import find_nearest_bucket
from diffusers_helper.utils import save_bcthw_as_mp4
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, \
                                   move_model_to_device_with_memory_preservation, \
                                   offload_model_from_device_for_memory_preservation, \
                                   fake_diffusers_current_device, DynamicSwapInstaller, \
                                   unload_complete_models, load_model_as_complete

from networks import lora_framepack
try:
    from lycoris.kohya import create_network_from_weights
except ImportError:
    pass
from base_wan_generate_video import merge_lora_weights


# --- Global Model Variables ---
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None

high_vram = False
free_mem_gb = 0.0

outputs_folder = './outputs/' # Default, can be overridden by --output_dir

@torch.no_grad()
def video_encode(video_path, resolution, no_resize, vae_model, vae_batch_size=16, device="cuda", width=None, height=None):
    video_path = str(pathlib.Path(video_path).resolve())
    print(f"Processing video for encoding: {video_path}")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU for video_encode")
        device = "cpu"

    try:
        print("Initializing VideoReader...")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        if fps == 0:
             print("Warning: VideoReader reported FPS as 0. Attempting to get it via OpenCV.")
             import cv2
             cap = cv2.VideoCapture(video_path)
             fps_cv = cap.get(cv2.CAP_PROP_FPS)
             cap.release()
             if fps_cv > 0:
                 fps = fps_cv
                 print(f"Using FPS from OpenCV: {fps}")
             else:
                 raise ValueError("Failed to determine FPS for the input video.")

        num_real_frames = len(vr)
        print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

        latent_size_factor = 4
        num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
        if num_frames != num_real_frames:
            print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")

        if num_frames == 0:
            raise ValueError(f"Video too short ({num_real_frames} frames) or becomes 0 after truncation. Needs at least {latent_size_factor} frames.")
        num_real_frames = num_frames

        print("Reading video frames...")
        frames_np_all = vr.get_batch(range(num_real_frames)).asnumpy()
        print(f"Frames read: {frames_np_all.shape}")

        native_height, native_width = frames_np_all.shape[1], frames_np_all.shape[2]
        print(f"Native video resolution: {native_width}x{native_height}")

        target_h_arg = native_height if height is None else height
        target_w_arg = native_width if width is None else width

        if not no_resize:
            actual_target_height, actual_target_width = find_nearest_bucket(target_h_arg, target_w_arg, resolution=resolution)
            print(f"Adjusted resolution for VAE encoding: {actual_target_width}x{actual_target_height}")
        else:
            actual_target_width = (native_width // 8) * 8
            actual_target_height = (native_height // 8) * 8
            if actual_target_width != native_width or actual_target_height != native_height:
                 print(f"Using native resolution, adjusted to be divisible by 8: {actual_target_width}x{actual_target_height}")
            else:
                print(f"Using native resolution without resizing: {actual_target_width}x{actual_target_height}")

        processed_frames_list = []
        for frame_idx in range(frames_np_all.shape[0]):
            frame = frames_np_all[frame_idx]
            frame_resized_np = resize_and_center_crop(frame, target_width=actual_target_width, target_height=actual_target_height)
            processed_frames_list.append(frame_resized_np)

        processed_frames_np_stack = np.stack(processed_frames_list)
        print(f"Frames preprocessed: {processed_frames_np_stack.shape}")

        input_image_np_for_clip = processed_frames_np_stack[0]

        print("Converting frames to tensor...")
        frames_pt = torch.from_numpy(processed_frames_np_stack).float() / 127.5 - 1.0
        frames_pt = frames_pt.permute(0, 3, 1, 2)
        frames_pt = frames_pt.unsqueeze(0).permute(0, 2, 1, 3, 4)
        print(f"Tensor shape for VAE: {frames_pt.shape}")

        input_video_pixels_cpu = frames_pt.clone().cpu() # For returning raw pixels if needed

        print(f"Moving VAE and tensor to device: {device}")
        vae_model.to(device)
        frames_pt = frames_pt.to(device)

        print(f"Encoding input video frames with VAE (batch size: {vae_batch_size})")
        all_latents_list = []
        vae_model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, frames_pt.shape[2], vae_batch_size), desc="VAE Encoding Video Frames", mininterval=0.1):
                batch_frames_pt = frames_pt[:, :, i:i + vae_batch_size]
                try:
                    batch_latents = vae_encode(batch_frames_pt, vae_model)
                    all_latents_list.append(batch_latents.cpu())
                except RuntimeError as e:
                    print(f"Error during VAE encoding: {str(e)}")
                    if "out of memory" in str(e).lower() and device == "cuda":
                        print("CUDA out of memory during VAE encoding. Try reducing --vae_batch_size or use CPU for VAE.")
                    raise

        history_latents_cpu = torch.cat(all_latents_list, dim=2)
        print(f"History latents shape (original video): {history_latents_cpu.shape}")

        start_latent_cpu = history_latents_cpu[:, :, :1].clone()
        print(f"Start latent shape (for conditioning): {start_latent_cpu.shape}")

        if device == "cuda":
            vae_model.to(cpu)
            torch.cuda.empty_cache()
            print("VAE moved back to CPU, CUDA cache cleared")

        return start_latent_cpu, input_image_np_for_clip, history_latents_cpu, fps, actual_target_height, actual_target_width, input_video_pixels_cpu

    except Exception as e:
        print(f"Error in video_encode: {str(e)}")
        traceback.print_exc()
        raise

def set_mp4_comments_imageio_ffmpeg(input_file, comments):
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return False
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        command = [
            ffmpeg_path, '-i', input_file, '-metadata', f'comment={comments}',
            '-c:v', 'copy', '-c:a', 'copy', '-y', temp_file
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            shutil.move(temp_file, input_file)
            print(f"Successfully added comments to {input_file}")
            return True
        else:
            if os.path.exists(temp_file): os.remove(temp_file)
            print(f"Error: FFmpeg failed with message:\n{result.stderr}")
            return False
    except Exception as e:
        if 'temp_file' in locals() and os.path.exists(temp_file): os.remove(temp_file)
        print(f"Error saving prompt to video metadata, ffmpeg may be required: "+str(e))
        return False

@torch.no_grad()
def do_extension_work(
    input_video_path, prompt, n_prompt, seed,
    resolution_max_dim,
    additional_second_length, # Duration of the extension
    latent_window_size, steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, no_resize, mp4_crf,
    num_clean_frames, vae_batch_size
):
    global high_vram, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer

    print('--- Starting Video Extension Work ---')

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        print('Text encoding for extension...')
        target_text_enc_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if text_encoder: fake_diffusers_current_device(text_encoder, target_text_enc_device)
            if text_encoder_2: load_model_as_complete(text_encoder_2, target_device=target_text_enc_device)
        else:
            if text_encoder: text_encoder.to(target_text_enc_device)
            if text_encoder_2: text_encoder_2.to(target_text_enc_device)

        llama_vec_gpu, clip_l_pooler_gpu = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1.0:
            llama_vec_n_gpu, clip_l_pooler_n_gpu = torch.zeros_like(llama_vec_gpu), torch.zeros_like(clip_l_pooler_gpu)
        else:
            llama_vec_n_gpu, clip_l_pooler_n_gpu = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec_padded_cpu, llama_attention_mask_cpu = crop_or_pad_yield_mask(llama_vec_gpu.cpu(), length=512)
        llama_vec_n_padded_cpu, llama_attention_mask_n_cpu = crop_or_pad_yield_mask(llama_vec_n_gpu.cpu(), length=512)
        clip_l_pooler_cpu = clip_l_pooler_gpu.cpu()
        clip_l_pooler_n_cpu = clip_l_pooler_n_gpu.cpu()

        print('Encoding input video for extension base...')
        video_encode_device = str(gpu if torch.cuda.is_available() else cpu)
        start_latent_cpu, input_image_np_for_clip, video_latents_history_cpu, fps, height, width, _ = video_encode(
            input_video_path, resolution_max_dim, no_resize, vae, vae_batch_size=vae_batch_size, device=video_encode_device
        )
        if fps <= 0:
            raise ValueError("FPS from input video is 0 or invalid. Cannot proceed with extension.")


        print('CLIP Vision encoding for extension...')
        target_img_enc_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if image_encoder: load_model_as_complete(image_encoder, target_device=target_img_enc_device)
        else:
            if image_encoder: image_encoder.to(target_img_enc_device)

        image_encoder_output = hf_clip_vision_encode(input_image_np_for_clip, feature_extractor, image_encoder)
        image_encoder_last_hidden_state_cpu = image_encoder_output.last_hidden_state.cpu()

        target_transformer_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if transformer: move_model_to_device_with_memory_preservation(transformer, target_device=target_transformer_device, preserved_memory_gb=gpu_memory_preservation)
        else:
            if transformer: transformer.to(target_transformer_device)

        cond_device = transformer.device
        cond_dtype = transformer.dtype

        llama_vec = llama_vec_padded_cpu.to(device=cond_device, dtype=cond_dtype)
        llama_attention_mask = llama_attention_mask_cpu.to(device=cond_device)
        llama_vec_n = llama_vec_n_padded_cpu.to(device=cond_device, dtype=cond_dtype)
        llama_attention_mask_n = llama_attention_mask_n_cpu.to(device=cond_device)
        clip_l_pooler = clip_l_pooler_cpu.to(device=cond_device, dtype=cond_dtype)
        clip_l_pooler_n = clip_l_pooler_n_cpu.to(device=cond_device, dtype=cond_dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state_cpu.to(device=cond_device, dtype=cond_dtype)
        start_latent_for_cond = start_latent_cpu.to(device=cond_device, dtype=torch.float32)


        num_output_pixel_frames_per_section = latent_window_size * 4 
        if num_output_pixel_frames_per_section == 0:
             raise ValueError("latent_window_size * 4 is zero, cannot calculate total_extension_latent_sections.")
        total_extension_latent_sections = int(max(round((additional_second_length * fps) / num_output_pixel_frames_per_section), 1))

        print(f"Input video FPS: {fps}, Target additional length: {additional_second_length}s")
        print(f"Generating {total_extension_latent_sections} new sections for extension (approx {total_extension_latent_sections * num_output_pixel_frames_per_section / fps:.2f}s).")

        job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + \
                 f"_framepackf1-vidEXT_{width}x{height}_{additional_second_length:.1f}s_seed{seed}_s{steps}_gs{gs}_cfg{cfg}"

        rnd = torch.Generator("cpu").manual_seed(seed)
        
        history_latents_combined_cpu = video_latents_history_cpu.clone()
        
        print("Decoding original input video content for appending...")
        target_vae_device_for_initial_decode = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if vae: load_model_as_complete(vae, target_device=target_vae_device_for_initial_decode)
        else:
            if vae: vae.to(target_vae_device_for_initial_decode)
        
        initial_video_pixels_cpu = vae_decode(video_latents_history_cpu.to(target_vae_device_for_initial_decode), vae).cpu()
        history_pixels_decoded_cpu = initial_video_pixels_cpu.clone() 
        
        if not high_vram and vae: unload_complete_models(vae)

        total_current_pixel_frames_count = history_pixels_decoded_cpu.shape[2]
        previous_video_path_for_cleanup = None

        for section_index in range(total_extension_latent_sections):
            print(f"--- F1 Extension: Seed {seed}: Section {section_index + 1}/{total_extension_latent_sections} ---")

            if transformer: transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)

            progress_bar_sampler = tqdm(total=steps, desc=f"Sampling Extension Section {section_index+1}/{total_extension_latent_sections}", file=sys.stdout)

            def sampler_callback_cli(d):
                progress_bar_sampler.update(1)

            available_latents_count_cpu = history_latents_combined_cpu.shape[2]
            pixel_frames_to_generate_this_step = latent_window_size * 4 - 3
            adjusted_latent_frames_for_output = (pixel_frames_to_generate_this_step + 3) // 4 

            effective_clean_frames_count = max(0, num_clean_frames -1) if num_clean_frames > 1 else 0
            effective_clean_frames_count = min(effective_clean_frames_count, available_latents_count_cpu - 2) if available_latents_count_cpu > 2 else 0
            
            num_2x_frames_count = min(2, max(0, available_latents_count_cpu - effective_clean_frames_count -1)) if available_latents_count_cpu > effective_clean_frames_count +1 else 0
            num_4x_frames_count = min(16, max(0, available_latents_count_cpu - effective_clean_frames_count - num_2x_frames_count)) if available_latents_count_cpu > effective_clean_frames_count + num_2x_frames_count else 0
            
            total_context_latents_count = num_4x_frames_count + num_2x_frames_count + effective_clean_frames_count
            total_context_latents_count = min(total_context_latents_count, available_latents_count_cpu)

            indices_tensor_gpu = torch.arange(0, sum([1, num_4x_frames_count, num_2x_frames_count, effective_clean_frames_count, adjusted_latent_frames_for_output])).unsqueeze(0).to(cond_device)
            
            clean_latent_indices_start_gpu, \
            clean_latent_4x_indices_gpu, \
            clean_latent_2x_indices_gpu, \
            clean_latent_1x_indices_gpu, \
            latent_indices_for_denoising_gpu = indices_tensor_gpu.split(
                [1, num_4x_frames_count, num_2x_frames_count, effective_clean_frames_count, adjusted_latent_frames_for_output], dim=1
            )
            clean_latent_indices_combined_gpu = torch.cat([clean_latent_indices_start_gpu, clean_latent_1x_indices_gpu], dim=1)

            context_latents_for_split_cpu = history_latents_combined_cpu[:, :, -total_context_latents_count:, :, :] if total_context_latents_count > 0 else history_latents_combined_cpu[:,:,:1,:,:].clone() 

            clean_latents_4x_gpu = torch.empty((1,16,0,height//8,width//8), device=cond_device, dtype=torch.float32) 
            clean_latents_2x_gpu = torch.empty((1,16,0,height//8,width//8), device=cond_device, dtype=torch.float32)
            clean_latents_1x_gpu = torch.empty((1,16,0,height//8,width//8), device=cond_device, dtype=torch.float32)

            current_offset_in_context_cpu = 0
            if num_4x_frames_count > 0 and total_context_latents_count > 0:
                slice_end = min(current_offset_in_context_cpu + num_4x_frames_count, context_latents_for_split_cpu.shape[2])
                clean_latents_4x_gpu = context_latents_for_split_cpu[:, :, current_offset_in_context_cpu:slice_end].to(device=cond_device, dtype=torch.float32)
                current_offset_in_context_cpu += clean_latents_4x_gpu.shape[2]
            
            if num_2x_frames_count > 0 and current_offset_in_context_cpu < context_latents_for_split_cpu.shape[2]:
                slice_end = min(current_offset_in_context_cpu + num_2x_frames_count, context_latents_for_split_cpu.shape[2])
                clean_latents_2x_gpu = context_latents_for_split_cpu[:, :, current_offset_in_context_cpu:slice_end].to(device=cond_device, dtype=torch.float32)
                current_offset_in_context_cpu += clean_latents_2x_gpu.shape[2]

            if effective_clean_frames_count > 0 and current_offset_in_context_cpu < context_latents_for_split_cpu.shape[2]:
                slice_end = min(current_offset_in_context_cpu + effective_clean_frames_count, context_latents_for_split_cpu.shape[2])
                clean_latents_1x_gpu = context_latents_for_split_cpu[:, :, current_offset_in_context_cpu:slice_end].to(device=cond_device, dtype=torch.float32)
            
            clean_latents_for_sampler_gpu = torch.cat([start_latent_for_cond, clean_latents_1x_gpu], dim=2)
            
            generated_latents_gpu_step = sample_hunyuan( 
                transformer=transformer, sampler='unipc', width=width, height=height,
                frames=pixel_frames_to_generate_this_step, 
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=cond_device, dtype=cond_dtype, 
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices_for_denoising_gpu, 
                clean_latents=clean_latents_for_sampler_gpu, 
                clean_latent_indices=clean_latent_indices_combined_gpu,
                clean_latents_2x=clean_latents_2x_gpu, 
                clean_latent_2x_indices=clean_latent_2x_indices_gpu,
                clean_latents_4x=clean_latents_4x_gpu, 
                clean_latent_4x_indices=clean_latent_4x_indices_gpu,
                callback=sampler_callback_cli,
            ) 
            if progress_bar_sampler: progress_bar_sampler.close()

            history_latents_combined_cpu = torch.cat([history_latents_combined_cpu, generated_latents_gpu_step.cpu()], dim=2)
            
            target_vae_device = str(gpu if torch.cuda.is_available() else cpu)
            if not high_vram: 
                if transformer: offload_model_from_device_for_memory_preservation(transformer, target_device=target_transformer_device, preserved_memory_gb=gpu_memory_preservation)
                if vae: load_model_as_complete(vae, target_device=target_vae_device)
            else: 
                if vae: vae.to(target_vae_device)
            
            num_latents_for_stitch_decode = latent_window_size * 2 
            num_latents_for_stitch_decode = min(num_latents_for_stitch_decode, history_latents_combined_cpu.shape[2])
            latents_for_current_part_decode_gpu = history_latents_combined_cpu[:, :, -num_latents_for_stitch_decode:].to(target_vae_device)
            
            pixels_for_current_part_decoded_cpu = vae_decode(
                latents_for_current_part_decode_gpu,
                vae
            ).cpu()

            overlap_for_soft_append = latent_window_size * 4 - 3 
            overlap_for_soft_append = min(overlap_for_soft_append, history_pixels_decoded_cpu.shape[2], pixels_for_current_part_decoded_cpu.shape[2])

            if overlap_for_soft_append <= 0: 
                 history_pixels_decoded_cpu = torch.cat([history_pixels_decoded_cpu, pixels_for_current_part_decoded_cpu[:,:,history_pixels_decoded_cpu.shape[2]:] ], dim=2) 
            else:
                # Corrected call to soft_append_bcthw
                history_pixels_decoded_cpu = soft_append_bcthw(
                    history_pixels_decoded_cpu, # Positional argument 1: history
                    pixels_for_current_part_decoded_cpu, # Positional argument 2: current
                    overlap=overlap_for_soft_append # Keyword argument: overlap
                )

            total_current_pixel_frames_count = history_pixels_decoded_cpu.shape[2] 

            if not high_vram: 
                if vae: unload_complete_models(vae) 
    
            current_output_filename = os.path.join(outputs_folder, f'{job_id}_part{section_index + 1}_totalframes{history_pixels_decoded_cpu.shape[2]}.mp4')
            save_bcthw_as_mp4(history_pixels_decoded_cpu, current_output_filename, fps=fps, crf=mp4_crf)
            print(f"MP4 Preview for section {section_index + 1} saved: {current_output_filename}")
            set_mp4_comments_imageio_ffmpeg(current_output_filename, f"Prompt: {prompt} | Neg: {n_prompt} | Seed: {seed}");
    
            if previous_video_path_for_cleanup is not None and os.path.exists(previous_video_path_for_cleanup):
                try:
                    os.remove(previous_video_path_for_cleanup)
                    print(f"Cleaned up previous part: {previous_video_path_for_cleanup}")
                except Exception as e_del:
                    print(f"Error deleting previous partial video {previous_video_path_for_cleanup}: {e_del}")
            previous_video_path_for_cleanup = current_output_filename
        
        final_video_path_for_item = previous_video_path_for_cleanup
        print(f"Final video for seed {seed} (extension 1) saved as: {final_video_path_for_item}")

    except Exception as e_outer:
        traceback.print_exc()
        print(f"Error during extension generation: {e_outer}")

    finally:
        if not high_vram: 
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        print("--- Extension work cycle finished. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FramePack F1 Video Extension CLI")
    
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file for extension.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for video generation.')
    parser.add_argument('--n_prompt', type=str, default="", help='Negative prompt.')
    parser.add_argument('--seed', type=int, default=31337, help='Seed for generation.')
    parser.add_argument('--resolution_max_dim', type=int, default=640, help='Target resolution (max width or height for bucket search).')
    parser.add_argument('--total_second_length', type=float, default=5.0, help='Additional video length to generate (seconds).') 
    parser.add_argument('--latent_window_size', type=int, default=9, help='Latent window size (frames).')
    parser.add_argument('--steps', type=int, default=25, help='Number of inference steps.')
    parser.add_argument('--cfg', type=float, default=1.0, help='CFG Scale (Classifier Free Guidance).')
    parser.add_argument('--gs', type=float, default=3.0, help='Distilled CFG Scale (Embedded CFG).')
    parser.add_argument('--rs', type=float, default=0.0, help='CFG Re-Scale (usually 0.0).')
    parser.add_argument('--gpu_memory_preservation', type=float, default=6.0, help='GPU memory to preserve (GB) for low VRAM mode.')
    parser.add_argument('--use_teacache', action='store_true', default=False, help='Enable TeaCache.')
    parser.add_argument('--no_resize', action='store_true', default=False, help='Force original video resolution for input video encoding.')
    parser.add_argument('--mp4_crf', type=int, default=16, help='MP4 CRF value (0-51, lower is better quality).')
    parser.add_argument('--num_clean_frames', type=int, default=5, help='Number of 1x context frames from input video history for DiT conditioning.')
    parser.add_argument('--vae_batch_size', type=int, default=-1, help='VAE batch size for input video encoding. Default: auto based on VRAM.')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help="Directory to save output videos.")

    parser.add_argument('--dit', type=str, required=True, help="Path to local DiT model weights file or directory.")
    parser.add_argument('--vae', type=str, required=True, help="Path to local VAE model weights file or directory.")
    parser.add_argument('--text_encoder1', type=str, required=True, help="Path to Text Encoder 1 (Llama) WEIGHT FILE.")
    parser.add_argument('--text_encoder2', type=str, required=True, help="Path to Text Encoder 2 (CLIP) WEIGHT FILE.")
    parser.add_argument('--image_encoder', type=str, required=True, help="Path to Image Encoder (SigLIP) WEIGHT FILE.")
    
    parser.add_argument('--attn_mode', type=str, default="torch", help="Attention mode for DiT.")
    parser.add_argument('--fp8_llm', action='store_true', help="Use fp8 for Text Encoder 1 (Llama).")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="Chunk size for CausalConv3d in VAE.")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None, help="Spatial tile sample min size for VAE.")
    
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path(s).")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=[1.0], help="LoRA multiplier(s).")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns.")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns.")

    args = parser.parse_args()
    
    current_device_str = str(gpu if torch.cuda.is_available() else cpu)
    args.device = current_device_str 

    for model_arg_name in ['dit', 'vae', 'text_encoder1', 'text_encoder2', 'image_encoder']:
        path_val = getattr(args, model_arg_name)
        if not os.path.exists(path_val): 
            parser.error(f"Path for --{model_arg_name} not found: {path_val}")

    outputs_folder = args.output_dir 
    os.makedirs(outputs_folder, exist_ok=True)
    print(f"Outputting extensions to: {outputs_folder}")

    free_mem_gb = get_cuda_free_memory_gb(gpu if torch.cuda.is_available() else None)
    high_vram = free_mem_gb > 40 
    print(f'Free VRAM {free_mem_gb:.2f} GB. High-VRAM Mode: {high_vram}')

    if args.vae_batch_size == -1: 
        if free_mem_gb >= 18: args.vae_batch_size = 64 
        elif free_mem_gb >= 10: args.vae_batch_size = 32
        else: args.vae_batch_size = 16 
        print(f"Auto-set VAE batch size to: {args.vae_batch_size}")
    
    print("Loading models for extension...")
    loading_device_str = str(cpu) 

    transformer = load_packed_model(
        device=loading_device_str, 
        dit_path=args.dit,
        attn_mode=args.attn_mode, 
        loading_device=loading_device_str 
    )
    print("DiT loaded.")

    if args.lora_weight is not None and len(args.lora_weight) > 0:
        print("Merging LoRA weights for extension...")
        if len(args.lora_multiplier) == 1 and len(args.lora_weight) > 1:
            args.lora_multiplier = args.lora_multiplier * len(args.lora_weight)
        elif len(args.lora_multiplier) != len(args.lora_weight):
            parser.error(f"Number of LoRA weights ({len(args.lora_weight)}) and multipliers ({len(args.lora_multiplier)}) must match, or provide a single multiplier.")
        try:
            merge_lora_weights(
                network_module=lora_framepack,
                text_encoder=None,
                unet=transformer,
                models=args.lora_weight,
                ratios=args.lora_multiplier,
                device=torch.device(loading_device_str),
                args_for_lora=args
            )
            print("LoRA weights merged successfully.")
        except Exception as e_lora:
            print(f"Error merging LoRA weights: {e_lora}")
            traceback.print_exc()

    vae = load_vae(
        vae_path=args.vae, 
        vae_chunk_size=args.vae_chunk_size, 
        vae_spatial_tile_sample_min_size=args.vae_spatial_tile_sample_min_size, 
        device=loading_device_str 
    )
    print("VAE loaded.")

    tokenizer, text_encoder = load_text_encoder1(args, device=loading_device_str) 
    print("Text Encoder 1 and Tokenizer 1 loaded.")
    tokenizer_2, text_encoder_2 = load_text_encoder2(args)
    print("Text Encoder 2 and Tokenizer 2 loaded.")
    feature_extractor, image_encoder = load_image_encoders(args)
    print("Image Encoder and Feature Extractor loaded.")

    all_models_list = [transformer, vae, text_encoder, text_encoder_2, image_encoder]
    for model_obj in all_models_list:
        if model_obj is not None:
            model_obj.eval().requires_grad_(False)

    if transformer: transformer.to(dtype=torch.bfloat16)
    if vae: vae.to(dtype=torch.float16) 
    if image_encoder: image_encoder.to(dtype=torch.float16)
    if text_encoder: text_encoder.to(dtype=torch.float16) 
    if text_encoder_2: text_encoder_2.to(dtype=torch.float16)
    
    if transformer:
        transformer.high_quality_fp32_output_for_inference = True 
        print('Transformer: high_quality_fp32_output_for_inference = True')
    
    if vae and not high_vram: 
        vae.enable_slicing()
        vae.enable_tiling()

    target_gpu_device_str = str(gpu if torch.cuda.is_available() else cpu)
    if not high_vram and torch.cuda.is_available():
        print("Low VRAM mode: Setting up dynamic swapping for DiT and Text Encoder 1.")
        if transformer: DynamicSwapInstaller.install_model(transformer, device=target_gpu_device_str)
        if text_encoder: DynamicSwapInstaller.install_model(text_encoder, device=target_gpu_device_str)
        if vae: vae.to(cpu)
        if text_encoder_2: text_encoder_2.to(cpu)
        if image_encoder: image_encoder.to(cpu)
    elif torch.cuda.is_available(): 
        print(f"High VRAM mode: Moving all models to {target_gpu_device_str}.")
        for model_obj in all_models_list:
            if model_obj is not None: model_obj.to(target_gpu_device_str)
    else:
        print("Running on CPU. Models remain on CPU.")
    
    print("All models loaded and configured for extension.")
    
    actual_gs_cli = args.gs
    if args.cfg > 1.0: 
        actual_gs_cli = 1.0 
        print(f"CFG > 1.0 detected ({args.cfg}), overriding GS to 1.0 from {args.gs}.")

    do_extension_work(
        input_video_path=args.input_video, 
        prompt=args.prompt, 
        n_prompt=args.n_prompt, 
        seed=args.seed,
        resolution_max_dim=args.resolution_max_dim, 
        additional_second_length=args.total_second_length,
        latent_window_size=args.latent_window_size, 
        steps=args.steps, 
        cfg=args.cfg, 
        gs=actual_gs_cli, 
        rs=args.rs, 
        gpu_memory_preservation=args.gpu_memory_preservation, 
        use_teacache=args.use_teacache, 
        no_resize=args.no_resize, 
        mp4_crf=args.mp4_crf, 
        num_clean_frames=args.num_clean_frames, 
        vae_batch_size=args.vae_batch_size
    )

    print("Video extension process completed.")