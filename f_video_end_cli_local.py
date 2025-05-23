# --- START OF FILE framepack_video_endframe_cli_local.py ---

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

# --- Imports from fpack_generate_video.py's ecosystem (or similar paths) ---
# Assuming these are available in the execution environment or PYTHONPATH
# For a self-contained script, these might need to be vendored or adjusted.
try:
    from frame_pack.hunyuan_video_packed import load_packed_model
    from frame_pack.framepack_utils import (
        load_vae,
        load_text_encoder1,
        load_text_encoder2,
        load_image_encoders
    )
    from frame_pack.hunyuan import encode_prompt_conds, vae_decode, vae_encode # vae_decode_fake might be needed for previews if added
    from frame_pack.utils import crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
    from frame_pack.k_diffusion_hunyuan import sample_hunyuan
    from frame_pack.clip_vision import hf_clip_vision_encode
    from frame_pack.bucket_tools import find_nearest_bucket
    from diffusers_helper.utils import save_bcthw_as_mp4 # from a common helper library
    from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, \
                                       move_model_to_device_with_memory_preservation, \
                                       offload_model_from_device_for_memory_preservation, \
                                       fake_diffusers_current_device, DynamicSwapInstaller, \
                                       unload_complete_models, load_model_as_complete
    # For LoRA
    from networks import lora_framepack 
    try:
        from lycoris.kohya import create_network_from_weights
    except ImportError:
        pass # Lycoris optional
    from base_wan_generate_video import merge_lora_weights # Assuming this is accessible
except ImportError as e:
    print(f"Error importing FramePack related modules: {e}. Ensure they are in PYTHONPATH.")
    sys.exit(1)


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
                 # Fallback FPS if all else fails
                 fps = 25 
                 print(f"Failed to determine FPS for the input video. Defaulting to {fps} FPS.")


        num_real_frames = len(vr)
        print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

        latent_size_factor = 4 # Hunyuan VAE downsamples by 8, but generation often uses 4x frame groups
        num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
        if num_frames != num_real_frames:
            print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility (multiple of {latent_size_factor})")

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

        input_image_np_for_clip_first = processed_frames_np_stack[0]
        input_image_np_for_clip_last = processed_frames_np_stack[-1]


        print("Converting frames to tensor...")
        frames_pt = torch.from_numpy(processed_frames_np_stack).float() / 127.5 - 1.0
        frames_pt = frames_pt.permute(0, 3, 1, 2) # B, H, W, C -> B, C, H, W
        frames_pt = frames_pt.unsqueeze(0).permute(0, 2, 1, 3, 4) # B, C, H, W -> 1, C, B, H, W (as VAE expects 1,C,F,H,W)
        print(f"Tensor shape for VAE: {frames_pt.shape}")

        input_video_pixels_cpu = frames_pt.clone().cpu() 

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
        end_of_input_video_latent_cpu = history_latents_cpu[:, :, -1:].clone()
        print(f"Start latent shape (for conditioning): {start_latent_cpu.shape}")
        print(f"End of input video latent shape: {end_of_input_video_latent_cpu.shape}")


        if device == "cuda":
            vae_model.to(cpu) # Move VAE back to CPU
            torch.cuda.empty_cache()
            print("VAE moved back to CPU, CUDA cache cleared")

        return (start_latent_cpu, input_image_np_for_clip_first, 
                history_latents_cpu, fps, 
                actual_target_height, actual_target_width, 
                input_video_pixels_cpu, 
                end_of_input_video_latent_cpu, input_image_np_for_clip_last)

    except Exception as e:
        print(f"Error in video_encode: {str(e)}")
        traceback.print_exc()
        raise

@torch.no_grad()
def image_encode(image_np, target_width, target_height, vae_model, image_encoder_model, feature_extractor_model, device="cuda"):
    """
    Encode a single image into a latent and compute its CLIP vision embedding.
    """
    global high_vram # Use global high_vram status
    print("Processing single image for encoding (e.g., end_frame)...")
    try:
        print(f"Using target resolution for image encoding: {target_width}x{target_height}")

        processed_image_np = resize_and_center_crop(image_np, target_width=target_width, target_height=target_height)

        image_pt = torch.from_numpy(processed_image_np).float() / 127.5 - 1.0
        image_pt = image_pt.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # N C F H W (N=1, F=1)
        
        target_vae_device = device
        if not high_vram: load_model_as_complete(vae_model, target_device=target_vae_device)
        else: vae_model.to(target_vae_device)
        image_pt_device = image_pt.to(target_vae_device)
        
        latent = vae_encode(image_pt_device, vae_model).cpu() # Encode and move to CPU
        print(f"Single image VAE output shape (latent): {latent.shape}")

        if not high_vram: unload_complete_models(vae_model) # Offload VAE if low VRAM

        target_img_enc_device = device
        if not high_vram: load_model_as_complete(image_encoder_model, target_device=target_img_enc_device)
        else: image_encoder_model.to(target_img_enc_device)

        clip_embedding_output = hf_clip_vision_encode(processed_image_np, feature_extractor_model, image_encoder_model)
        clip_embedding = clip_embedding_output.last_hidden_state.cpu() # Encode and move to CPU
        print(f"Single image CLIP embedding shape: {clip_embedding.shape}")

        if not high_vram: unload_complete_models(image_encoder_model) # Offload image encoder if low VRAM
        
        if device == "cuda":
            torch.cuda.empty_cache()
            # print("CUDA cache cleared after single image_encode")

        return latent, clip_embedding, processed_image_np

    except Exception as e:
        print(f"Error in image_encode: {str(e)}")
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
def do_generation_work(
    input_video_path, prompt, n_prompt, seed,
    end_frame_path, end_frame_weight, # New arguments
    resolution_max_dim,
    additional_second_length, 
    latent_window_size, steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, no_resize, mp4_crf,
    num_clean_frames, vae_batch_size
):
    global high_vram, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer

    print('--- Starting Video Generation (with End Frame support) ---')

    try:
        # --- Text Encoding ---
        print('Text encoding...')
        target_text_enc_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if text_encoder: fake_diffusers_current_device(text_encoder, target_text_enc_device) # DynamicSwapInstaller for text_encoder
            if text_encoder_2: load_model_as_complete(text_encoder_2, target_device=target_text_enc_device)
        else:
            if text_encoder: text_encoder.to(target_text_enc_device)
            if text_encoder_2: text_encoder_2.to(target_text_enc_device)

        llama_vec_gpu, clip_l_pooler_gpu = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1.0: # Note: Original FramePack usually uses gs, cfg=1 means gs is active
            llama_vec_n_gpu, clip_l_pooler_n_gpu = torch.zeros_like(llama_vec_gpu), torch.zeros_like(clip_l_pooler_gpu)
        else: # If cfg > 1.0, it implies standard CFG, so n_prompt is used. gs should be 1.0 in this case.
            llama_vec_n_gpu, clip_l_pooler_n_gpu = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Store on CPU
        llama_vec_padded_cpu, llama_attention_mask_cpu = crop_or_pad_yield_mask(llama_vec_gpu.cpu(), length=512)
        llama_vec_n_padded_cpu, llama_attention_mask_n_cpu = crop_or_pad_yield_mask(llama_vec_n_gpu.cpu(), length=512)
        clip_l_pooler_cpu = clip_l_pooler_gpu.cpu()
        clip_l_pooler_n_cpu = clip_l_pooler_n_gpu.cpu()
        
        if not high_vram: unload_complete_models(text_encoder_2) # text_encoder is managed by DynamicSwap

        # --- Video and End Frame Encoding ---
        print('Encoding input video...')
        video_encode_device = str(gpu if torch.cuda.is_available() else cpu)
        (start_latent_input_cpu, input_image_np_first, 
         video_latents_history_cpu, fps, height, width, 
         input_video_pixels_cpu,
         end_of_input_video_latent_cpu, input_image_np_last) = video_encode(
            input_video_path, resolution_max_dim, no_resize, vae, 
            vae_batch_size=vae_batch_size, device=video_encode_device,
            width=None, height=None # video_encode will use resolution_max_dim
        )
        if fps <= 0: raise ValueError("FPS from input video is 0 or invalid.")

        end_latent_from_file_cpu, end_clip_embedding_from_file_cpu = None, None
        if end_frame_path:
            print(f"Encoding provided end frame from: {end_frame_path}")
            end_frame_pil = Image.open(end_frame_path).convert("RGB")
            end_frame_np = np.array(end_frame_pil)
            end_latent_from_file_cpu, end_clip_embedding_from_file_cpu, _ = image_encode(
                end_frame_np, target_width=width, target_height=height, 
                vae_model=vae, image_encoder_model=image_encoder, 
                feature_extractor_model=feature_extractor, device=video_encode_device
            )

        # --- CLIP Vision Encoding for first and last frames of input video ---
        print('CLIP Vision encoding for input video frames...')
        target_img_enc_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram: load_model_as_complete(image_encoder, target_device=target_img_enc_device)
        else: image_encoder.to(target_img_enc_device)

        # For original FramePack, image_embeddings in sample_hunyuan often comes from the *start* image.
        # Script 2 uses end_of_input_video_embedding or a blend with the explicit end_frame.
        # We will follow script 2 for conditioning.
        # start_clip_embedding_cpu = hf_clip_vision_encode(input_image_np_first, feature_extractor, image_encoder).last_hidden_state.cpu()
        end_of_input_video_clip_embedding_cpu = hf_clip_vision_encode(input_image_np_last, feature_extractor, image_encoder).last_hidden_state.cpu()

        if not high_vram: unload_complete_models(image_encoder)

        # Determine final image embedding for sampling loop
        if end_clip_embedding_from_file_cpu is not None:
            print(f"Blending end-of-input-video embedding with provided end_frame embedding (weight: {end_frame_weight})")
            final_clip_embedding_for_sampling_cpu = \
                (1.0 - end_frame_weight) * end_of_input_video_clip_embedding_cpu + \
                end_frame_weight * end_clip_embedding_from_file_cpu
        else:
            print("Using end-of-input-video's last frame embedding for image conditioning.")
            final_clip_embedding_for_sampling_cpu = end_of_input_video_clip_embedding_cpu.clone()
        
        # --- Prepare for Sampling Loop ---
        target_transformer_device = str(gpu if torch.cuda.is_available() else cpu)
        if not high_vram:
            if transformer: move_model_to_device_with_memory_preservation(transformer, target_device=target_transformer_device, preserved_memory_gb=gpu_memory_preservation)
        else:
            if transformer: transformer.to(target_transformer_device)
        
        cond_device = transformer.device
        cond_dtype = transformer.dtype

        # Move conditioning tensors to transformer's device and dtype
        llama_vec = llama_vec_padded_cpu.to(device=cond_device, dtype=cond_dtype)
        llama_attention_mask = llama_attention_mask_cpu.to(device=cond_device) # Mask is usually bool/int
        clip_l_pooler = clip_l_pooler_cpu.to(device=cond_device, dtype=cond_dtype)
        llama_vec_n = llama_vec_n_padded_cpu.to(device=cond_device, dtype=cond_dtype)
        llama_attention_mask_n = llama_attention_mask_n_cpu.to(device=cond_device)
        clip_l_pooler_n = clip_l_pooler_n_cpu.to(device=cond_device, dtype=cond_dtype)
        
        # This is the image embedding that will be used in the sampling loop
        image_embeddings_for_sampling_loop = final_clip_embedding_for_sampling_cpu.to(device=cond_device, dtype=cond_dtype)
        
        # start_latent_for_initial_cond_gpu is the first frame of input video, used for clean_latents_pre
        # However, script 2 uses `video_latents[:, :, -min(effective_clean_frames, video_latents.shape[2]):]` for clean_latents_pre.
        # And `start_latent` for sample_hunyuan's `clean_latents` is `torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)`
        # For backward generation, the "start_latent" concept for `sample_hunyuan`'s `clean_latents` argument
        # is often the *last frame of the input video* when generating the chunk closest to the input video.
        # Let's use end_of_input_video_latent_cpu for this role when appropriate.

        num_output_pixel_frames_per_section = latent_window_size * 4 # Not -3 here, as this is for total section calc
        if num_output_pixel_frames_per_section == 0:
             raise ValueError("latent_window_size * 4 is zero, cannot calculate total_extension_latent_sections.")
        total_extension_latent_sections = int(max(round((additional_second_length * fps) / num_output_pixel_frames_per_section), 1))

        print(f"Input video FPS: {fps}, Target additional length: {additional_second_length}s")
        print(f"Generating {total_extension_latent_sections} new sections for extension (approx {total_extension_latent_sections * num_output_pixel_frames_per_section / fps:.2f}s).")

        job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + \
                 f"_framepack-vidEndFrm_{width}x{height}_{additional_second_length:.1f}s_seed{seed}_s{steps}_gs{gs}_cfg{cfg}"

        rnd = torch.Generator("cpu").manual_seed(seed)
        
        # Initialize history for generated latents (starts empty or with end_latent_from_file)
        if end_latent_from_file_cpu is not None:
            # This assumes end_latent_from_file_cpu is [1,C,1,H,W], we might need more frames if it's a seed
            # Script 2's logic for clean_latents_post when is_end_of_video seems to use just 1 frame.
            history_latents_generated_cpu = end_latent_from_file_cpu.clone() 
        else:
            channels_dim = video_latents_history_cpu.shape[1] # Get from input video latents
            latent_h, latent_w = height // 8, width // 8
            history_latents_generated_cpu = torch.empty((1, channels_dim, 0, latent_h, latent_w), dtype=torch.float32, device='cpu')
        
        # Initialize history for decoded pixels (starts empty)
        history_pixels_decoded_cpu = None
        
        total_generated_latent_frames_count = history_latents_generated_cpu.shape[2]
        previous_video_path_for_cleanup = None

        # Backward generation loop (from demo_gradio_video+endframe.py)
        latent_paddings = list(reversed(range(total_extension_latent_sections)))
        if total_extension_latent_sections > 4: # Heuristic from script 2
            latent_paddings = [3] + [2] * (total_extension_latent_sections - 3) + [1, 0]

        for loop_idx, latent_padding_val in enumerate(latent_paddings):
            current_section_num_from_end = loop_idx + 1
            is_start_of_extension = (latent_padding_val == 0) # This is the chunk closest to input video
            is_end_of_extension = (latent_padding_val == latent_paddings[0]) # This is the chunk furthest from input video

            print(f"--- Generating Extension: Seed {seed}: Section {current_section_num_from_end}/{total_extension_latent_sections} (backward), padding={latent_padding_val} ---")

            if transformer: transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else 0)
            progress_bar_sampler = tqdm(total=steps, desc=f"Sampling Extension Section {current_section_num_from_end}/{total_extension_latent_sections}", file=sys.stdout, dynamic_ncols=True)
            def sampler_callback_cli(d): progress_bar_sampler.update(1)

            # Context frame calculation (from demo_gradio_video+endframe.py worker)
            # `available_frames` for context refers to previously *generated* frames or input video frames
            # For `clean_latents_pre`, it's always from `video_latents_history_cpu`
            # For `clean_latents_post`, `_2x`, `_4x`, it's from `history_latents_generated_cpu`
            
            effective_clean_frames_count = max(0, num_clean_frames - 1) if num_clean_frames > 1 else 1
            
            # For clean_latents_pre (from input video)
            # If is_start_of_extension, we might want stronger anchoring to input video. Script 2 uses full `effective_clean_frames_count`.
            clean_latent_pre_frames_num = effective_clean_frames_count
            if is_start_of_extension: # Closest to input video
                 clean_latent_pre_frames_num = 1 # Script 2 uses 1 to avoid jumpcuts from input video when generating chunk closest to it.

            # For clean_latents_post, _2x, _4x (from previously generated extension chunks)
            available_generated_latents = history_latents_generated_cpu.shape[2]
            
            # `post_frames_num` is for clean_latents_post
            post_frames_num = 1 if is_end_of_extension and end_latent_from_file_cpu is not None else effective_clean_frames_count
            if is_end_of_extension and end_latent_from_file_cpu is not None: post_frames_num = 1 # script 2 detail for end_latent

            num_2x_frames_count = min(2, max(0, available_generated_latents - post_frames_num -1))
            num_4x_frames_count = min(16, max(0, available_generated_latents - post_frames_num - num_2x_frames_count))
            
            # Latent indexing for sample_hunyuan (from script 2)
            latent_padding_size_for_indices = latent_padding_val * latent_window_size
            pixel_frames_to_generate_this_step = latent_window_size * 4 - 3
            
            indices_tensor_gpu = torch.arange(0, 
                clean_latent_pre_frames_num + 
                latent_padding_size_for_indices + 
                latent_window_size + # Note: script 2 uses latent_window_size here for `latent_indices` count
                post_frames_num + 
                num_2x_frames_count + 
                num_4x_frames_count
            ).unsqueeze(0).to(cond_device)

            (clean_latent_indices_pre_gpu, 
             blank_indices_gpu, # For padding
             latent_indices_for_denoising_gpu, # For new generation
             clean_latent_indices_post_gpu,
             clean_latent_2x_indices_gpu,
             clean_latent_4x_indices_gpu
            ) = indices_tensor_gpu.split(
                [clean_latent_pre_frames_num, latent_padding_size_for_indices, latent_window_size, 
                 post_frames_num, num_2x_frames_count, num_4x_frames_count], dim=1
            )
            clean_latent_indices_combined_gpu = torch.cat([clean_latent_indices_pre_gpu, clean_latent_indices_post_gpu], dim=1)

            # Prepare conditioning latents
            # clean_latents_pre_cpu: from end of input video
            actual_pre_frames_to_take = min(clean_latent_pre_frames_num, video_latents_history_cpu.shape[2])
            clean_latents_pre_cpu = video_latents_history_cpu[:, :, -actual_pre_frames_to_take:].clone()
            if clean_latents_pre_cpu.shape[2] < clean_latent_pre_frames_num and clean_latents_pre_cpu.shape[2] > 0: # Pad if necessary
                repeats = math.ceil(clean_latent_pre_frames_num / clean_latents_pre_cpu.shape[2])
                clean_latents_pre_cpu = clean_latents_pre_cpu.repeat(1,1,repeats,1,1)[:,:,:clean_latent_pre_frames_num]
            elif clean_latents_pre_cpu.shape[2] == 0 and clean_latent_pre_frames_num > 0: # Should not happen if video_latents_history_cpu is valid
                clean_latents_pre_cpu = torch.zeros((1,channels_dim,clean_latent_pre_frames_num,latent_h,latent_w),dtype=torch.float32)


            # clean_latents_post_cpu, _2x_cpu, _4x_cpu: from start of `history_latents_generated_cpu`
            current_offset_in_generated = 0
            
            # Post frames
            actual_post_frames_to_take = min(post_frames_num, history_latents_generated_cpu.shape[2])
            if is_end_of_extension and end_latent_from_file_cpu is not None:
                clean_latents_post_cpu = end_latent_from_file_cpu.clone() # Should be [1,C,1,H,W]
            else:
                clean_latents_post_cpu = history_latents_generated_cpu[:,:, current_offset_in_generated : current_offset_in_generated + actual_post_frames_to_take].clone()
            current_offset_in_generated += clean_latents_post_cpu.shape[2]
            
            if clean_latents_post_cpu.shape[2] < post_frames_num and clean_latents_post_cpu.shape[2] > 0: # Pad
                repeats = math.ceil(post_frames_num / clean_latents_post_cpu.shape[2])
                clean_latents_post_cpu = clean_latents_post_cpu.repeat(1,1,repeats,1,1)[:,:,:post_frames_num]
            elif clean_latents_post_cpu.shape[2] == 0 and post_frames_num > 0: # Fill with zeros if no history and no end_latent
                 clean_latents_post_cpu = torch.zeros((1,channels_dim,post_frames_num,latent_h,latent_w),dtype=torch.float32)

            # 2x frames
            actual_2x_frames_to_take = min(num_2x_frames_count, history_latents_generated_cpu.shape[2] - current_offset_in_generated)
            clean_latents_2x_cpu = history_latents_generated_cpu[:,:, current_offset_in_generated : current_offset_in_generated + actual_2x_frames_to_take].clone()
            current_offset_in_generated += clean_latents_2x_cpu.shape[2]
            if clean_latents_2x_cpu.shape[2] < num_2x_frames_count and clean_latents_2x_cpu.shape[2] > 0: # Pad
                repeats = math.ceil(num_2x_frames_count / clean_latents_2x_cpu.shape[2])
                clean_latents_2x_cpu = clean_latents_2x_cpu.repeat(1,1,repeats,1,1)[:,:,:num_2x_frames_count]
            elif clean_latents_2x_cpu.shape[2] == 0 and num_2x_frames_count > 0:
                clean_latents_2x_cpu = torch.zeros((1,channels_dim,num_2x_frames_count,latent_h,latent_w),dtype=torch.float32)

            # 4x frames
            actual_4x_frames_to_take = min(num_4x_frames_count, history_latents_generated_cpu.shape[2] - current_offset_in_generated)
            clean_latents_4x_cpu = history_latents_generated_cpu[:,:, current_offset_in_generated : current_offset_in_generated + actual_4x_frames_to_take].clone()
            if clean_latents_4x_cpu.shape[2] < num_4x_frames_count and clean_latents_4x_cpu.shape[2] > 0: # Pad
                repeats = math.ceil(num_4x_frames_count / clean_latents_4x_cpu.shape[2])
                clean_latents_4x_cpu = clean_latents_4x_cpu.repeat(1,1,repeats,1,1)[:,:,:num_4x_frames_count]
            elif clean_latents_4x_cpu.shape[2] == 0 and num_4x_frames_count > 0:
                clean_latents_4x_cpu = torch.zeros((1,channels_dim,num_4x_frames_count,latent_h,latent_w),dtype=torch.float32)

# Combine pre and post for `clean_latents` argument
            clean_latents_for_sampler_gpu = torch.cat([
                clean_latents_pre_cpu.to(device=cond_device, dtype=torch.float32), 
                clean_latents_post_cpu.to(device=cond_device, dtype=torch.float32)
            ], dim=2)

            # Ensure 2x and 4x latents are None if their frame counts are 0
            # The k_diffusion_hunyuan.sample_hunyuan and the DiT should handle None for these if indices are also empty.
            clean_latents_2x_gpu = None
            if num_2x_frames_count > 0 and clean_latents_2x_cpu.shape[2] > 0:
                clean_latents_2x_gpu = clean_latents_2x_cpu.to(device=cond_device, dtype=torch.float32)
            elif num_2x_frames_count > 0 and clean_latents_2x_cpu.shape[2] == 0: # Should have been filled with zeros if count > 0
                 print(f"Warning: num_2x_frames_count is {num_2x_frames_count} but clean_latents_2x_cpu is empty. Defaulting to None.")


            clean_latents_4x_gpu = None
            if num_4x_frames_count > 0 and clean_latents_4x_cpu.shape[2] > 0:
                clean_latents_4x_gpu = clean_latents_4x_cpu.to(device=cond_device, dtype=torch.float32)
            elif num_4x_frames_count > 0 and clean_latents_4x_cpu.shape[2] == 0:
                 print(f"Warning: num_4x_frames_count is {num_4x_frames_count} but clean_latents_4x_cpu is empty. Defaulting to None.")

            # Also, ensure indices are None or empty if counts are zero.
            # The split logic already ensures this if the split size is 0.
            # clean_latent_2x_indices_gpu will be shape (B, 0) if num_2x_frames_count is 0.
            # The DiT model should correctly interpret an empty indices tensor or None for the corresponding latent.
            generated_latents_gpu_step = sample_hunyuan( 
                transformer=transformer, sampler='unipc', width=width, height=height,
                frames=pixel_frames_to_generate_this_step, # Num frames for current chunk
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=cond_device, dtype=cond_dtype, 
                image_embeddings=image_embeddings_for_sampling_loop, # Use the blended/final one
                latent_indices=latent_indices_for_denoising_gpu, 
                clean_latents=clean_latents_for_sampler_gpu, 
                clean_latent_indices=clean_latent_indices_combined_gpu,
                clean_latents_2x=clean_latents_2x_gpu, # Can be None
                clean_latent_2x_indices=clean_latent_2x_indices_gpu if num_2x_frames_count > 0 else None, # Pass None if count is 0
                clean_latents_4x=clean_latents_4x_gpu, # Can be None
                clean_latent_4x_indices=clean_latent_4x_indices_gpu if num_4x_frames_count > 0 else None, # Pass None if count is 0
                callback=sampler_callback_cli,
            )
            if progress_bar_sampler: progress_bar_sampler.close()

            # If this was the chunk closest to input video, prepend the last frame of input video for smoother transition
            if is_start_of_extension:
                generated_latents_gpu_step = torch.cat([
                    end_of_input_video_latent_cpu.to(generated_latents_gpu_step), # Use actual last frame latent
                    generated_latents_gpu_step
                ], dim=2)

            # Prepend generated latents to history
            history_latents_generated_cpu = torch.cat([generated_latents_gpu_step.cpu(), history_latents_generated_cpu], dim=2)
            total_generated_latent_frames_count = history_latents_generated_cpu.shape[2]
            
            # --- Decode and Append Pixels ---
            target_vae_device = str(gpu if torch.cuda.is_available() else cpu)
            if not high_vram: 
                if transformer: offload_model_from_device_for_memory_preservation(transformer, target_device=target_transformer_device, preserved_memory_gb=gpu_memory_preservation)
                if vae: load_model_as_complete(vae, target_device=target_vae_device)
            else: 
                if vae: vae.to(target_vae_device)
            
            # Decode the newly generated part (or a relevant segment for stitching)
            # Script 2 decodes `real_history_latents[:, :, :section_latent_frames]`
            # section_latent_frames = (latent_window_size * 2 + 1) if is_start_of_video else (latent_window_size * 2)
            num_latents_to_decode_for_stitch = (latent_window_size * 2 + 1) if is_start_of_extension else (latent_window_size * 2)
            num_latents_to_decode_for_stitch = min(num_latents_to_decode_for_stitch, history_latents_generated_cpu.shape[2])
            
            latents_for_current_decode_gpu = history_latents_generated_cpu[:, :, :num_latents_to_decode_for_stitch].to(target_vae_device)
            
            pixels_for_current_part_decoded_cpu = vae_decode(latents_for_current_decode_gpu, vae).cpu()

            # Soft append pixels (current_pixels, history_pixels, overlap)
            overlap_for_soft_append = latent_window_size * 4 - 3 
            
            if history_pixels_decoded_cpu is None:
                history_pixels_decoded_cpu = pixels_for_current_part_decoded_cpu
            else:
                overlap_actual = min(overlap_for_soft_append, history_pixels_decoded_cpu.shape[2], pixels_for_current_part_decoded_cpu.shape[2])
                if overlap_actual <=0: # Should not happen with proper windowing
                    history_pixels_decoded_cpu = torch.cat([pixels_for_current_part_decoded_cpu, history_pixels_decoded_cpu], dim=2) # Simple prepend
                else:
                    history_pixels_decoded_cpu = soft_append_bcthw(
                        pixels_for_current_part_decoded_cpu, # Current (prepended)
                        history_pixels_decoded_cpu,          # History
                        overlap=overlap_actual
                    )
            
            if not high_vram: 
                if vae: unload_complete_models(vae) 
                if transformer and not is_start_of_extension : # Reload transformer for next iter
                     move_model_to_device_with_memory_preservation(transformer, target_device=target_transformer_device, preserved_memory_gb=gpu_memory_preservation)

            # Save intermediate video
            current_output_filename = os.path.join(outputs_folder, f'{job_id}_part{current_section_num_from_end}_totalframes{history_pixels_decoded_cpu.shape[2]}.mp4')
            save_bcthw_as_mp4(history_pixels_decoded_cpu, current_output_filename, fps=fps, crf=mp4_crf)
            print(f"MP4 Preview for section {current_section_num_from_end} saved: {current_output_filename}")
            set_mp4_comments_imageio_ffmpeg(current_output_filename, f"Prompt: {prompt} | Neg: {n_prompt} | Seed: {seed}");
    
            if previous_video_path_for_cleanup is not None and os.path.exists(previous_video_path_for_cleanup):
                try: os.remove(previous_video_path_for_cleanup)
                except Exception as e_del: print(f"Error deleting {previous_video_path_for_cleanup}: {e_del}")
            previous_video_path_for_cleanup = current_output_filename
            
            if is_start_of_extension: # Last iteration of backward loop
                break
        
        # --- Final Video Assembly ---
        print("Appending generated extension to the input video...")
        # input_video_pixels_cpu is (1, C, F_in, H, W)
        # history_pixels_decoded_cpu is (1, C, F_ext, H, W)
        final_video_pixels_cpu = torch.cat([input_video_pixels_cpu, history_pixels_decoded_cpu], dim=2)
        
        final_output_filename = os.path.join(outputs_folder, f'{job_id}_final.mp4')
        save_bcthw_as_mp4(final_video_pixels_cpu, final_output_filename, fps=fps, crf=mp4_crf)
        print(f"Final extended video saved: {final_output_filename}")
        set_mp4_comments_imageio_ffmpeg(final_output_filename, f"Prompt: {prompt} | Neg: {n_prompt} | Seed: {seed}");

        if previous_video_path_for_cleanup is not None and os.path.exists(previous_video_path_for_cleanup) and previous_video_path_for_cleanup != final_output_filename:
            try: os.remove(previous_video_path_for_cleanup)
            except Exception as e_del: print(f"Error deleting last part: {e_del}")

    except Exception as e_outer:
        traceback.print_exc()
        print(f"Error during generation: {e_outer}")
    finally:
        if not high_vram: 
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        print("--- Generation work cycle finished. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FramePack Video Generation CLI (with End Frame)")
    
    # Inputs
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for video generation.')
    parser.add_argument('--n_prompt', type=str, default="", help='Negative prompt.')
    parser.add_argument('--end_frame', type=str, default=None, help='Optional path to an image to guide the end of the video.')
    parser.add_argument('--end_frame_weight', type=float, default=1.0, help='Weight for the end_frame image conditioning (0.0 to 1.0). Default 1.0.')

    # Generation parameters
    parser.add_argument('--seed', type=int, default=31337, help='Seed for generation.')
    parser.add_argument('--resolution_max_dim', type=int, default=640, help='Target resolution (max width or height for bucket search).')
    parser.add_argument('--total_second_length', type=float, default=5.0, help='Additional video length to generate (seconds).') 
    parser.add_argument('--latent_window_size', type=int, default=9, help='Latent window size (frames for DiT). Orignal FramePack default is 9.')
    parser.add_argument('--steps', type=int, default=25, help='Number of inference steps.')
    parser.add_argument('--cfg', type=float, default=1.0, help='CFG Scale. If > 1.0, n_prompt is used and gs is set to 1.0. Default 1.0 (for distilled guidance).')
    parser.add_argument('--gs', type=float, default=10.0, help='Distilled CFG Scale (Embedded CFG for Original FramePack). Default 10.0.') # Original default
    parser.add_argument('--rs', type=float, default=0.0, help='CFG Re-Scale (usually 0.0).')
    parser.add_argument('--num_clean_frames', type=int, default=5, help='Number of 1x context frames for DiT conditioning. Script2 default 5.')
    
    # Technical parameters
    parser.add_argument('--gpu_memory_preservation', type=float, default=6.0, help='GPU memory to preserve (GB) for low VRAM mode.')
    parser.add_argument('--use_teacache', action='store_true', default=False, help='Enable TeaCache (if DiT supports it).')
    parser.add_argument('--no_resize', action='store_true', default=False, help='Force original video resolution for input video encoding (VAE).')
    parser.add_argument('--mp4_crf', type=int, default=16, help='MP4 CRF value (0-51, lower is better quality).')
    parser.add_argument('--vae_batch_size', type=int, default=-1, help='VAE batch size for input video encoding. Default: auto based on VRAM.')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help="Directory to save output videos.")

    # Model paths
    parser.add_argument('--dit', type=str, required=True, help="Path to local DiT model weights file or directory (e.g., for lllyasviel/FramePackI2V_HY).")
    parser.add_argument('--vae', type=str, required=True, help="Path to local VAE model weights file or directory.")
    parser.add_argument('--text_encoder1', type=str, required=True, help="Path to Text Encoder 1 (Llama) WEIGHT FILE.")
    parser.add_argument('--text_encoder2', type=str, required=True, help="Path to Text Encoder 2 (CLIP) WEIGHT FILE.")
    parser.add_argument('--image_encoder', type=str, required=True, help="Path to Image Encoder (SigLIP) WEIGHT FILE.")
    
    # Advanced model settings
    parser.add_argument('--attn_mode', type=str, default="torch", help="Attention mode for DiT (torch, flash, xformers, etc.).")
    parser.add_argument('--fp8_llm', action='store_true', help="Use fp8 for Text Encoder 1 (Llama).") # from fpack_generate_video
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="Chunk size for CausalConv3d in VAE.")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None, help="Spatial tile sample min size for VAE.")
    
    # LoRA
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
    print(f"Outputting videos to: {outputs_folder}")

    free_mem_gb = get_cuda_free_memory_gb(gpu if torch.cuda.is_available() else None)
    # Adjusted high_vram threshold, can be tuned
    high_vram = free_mem_gb > 30 # Example: 30GB+ for "high_vram"
    print(f'Free VRAM {free_mem_gb:.2f} GB. High-VRAM Mode: {high_vram}')

    if args.vae_batch_size == -1: 
        if free_mem_gb >= 18: args.vae_batch_size = 64 
        elif free_mem_gb >= 10: args.vae_batch_size = 32
        else: args.vae_batch_size = 16 
        print(f"Auto-set VAE batch size to: {args.vae_batch_size}")
    
    print("Loading models...")
    loading_device_str = str(cpu) # Load to CPU first

    transformer = load_packed_model(
        device=loading_device_str, 
        dit_path=args.dit,
        attn_mode=args.attn_mode, 
        loading_device=loading_device_str 
    )
    print("DiT loaded.")

    if args.lora_weight is not None and len(args.lora_weight) > 0:
        print("Merging LoRA weights...")
        if len(args.lora_multiplier) == 1 and len(args.lora_weight) > 1:
            args.lora_multiplier = args.lora_multiplier * len(args.lora_weight)
        elif len(args.lora_multiplier) != len(args.lora_weight):
            parser.error(f"Number of LoRA weights ({len(args.lora_weight)}) and multipliers ({len(args.lora_multiplier)}) must match, or provide a single multiplier.")
        
        try:
            # Mimic fpack_generate_video.py's LoRA args structure if needed by merge_lora_weights
            if not hasattr(args, 'lycoris'): args.lycoris = False 
            if not hasattr(args, 'save_merged_model'): args.save_merged_model = None 
            
            current_device_for_lora = torch.device(loading_device_str)
            merge_lora_weights(lora_framepack, transformer, args, current_device_for_lora)
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
    
    # For text_encoder loading, fpack_generate_video.py uses args.fp8_llm for text_encoder1
    # The f1_video_cli_local.py passes `args` directly. We'll do the same.
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

    # Set dtypes (Original FramePack typically bfloat16 for DiT, float16 for others)
    if transformer: transformer.to(dtype=torch.bfloat16)
    if vae: vae.to(dtype=torch.float16) 
    if image_encoder: image_encoder.to(dtype=torch.float16)
    if text_encoder: text_encoder.to(dtype=torch.float16) # Or bfloat16 if fp8_llm implies that
    if text_encoder_2: text_encoder_2.to(dtype=torch.float16)
    
    if transformer:
        transformer.high_quality_fp32_output_for_inference = True # Common setting
        print('Transformer: high_quality_fp32_output_for_inference = True')
    
    if vae and not high_vram: 
        vae.enable_slicing()
        vae.enable_tiling()

    target_gpu_device_str = str(gpu if torch.cuda.is_available() else cpu)
    if not high_vram and torch.cuda.is_available():
        print("Low VRAM mode: Setting up dynamic swapping for DiT and Text Encoder 1.")
        if transformer: DynamicSwapInstaller.install_model(transformer, device=target_gpu_device_str)
        if text_encoder: DynamicSwapInstaller.install_model(text_encoder, device=target_gpu_device_str)
        # Other models (VAE, TE2, ImgEnc) will be loaded/offloaded as needed by `load_model_as_complete` / `unload_complete_models`
        if vae: vae.to(cpu)
        if text_encoder_2: text_encoder_2.to(cpu)
        if image_encoder: image_encoder.to(cpu)
    elif torch.cuda.is_available(): 
        print(f"High VRAM mode: Moving all models to {target_gpu_device_str}.")
        for model_obj in all_models_list:
            if model_obj is not None: model_obj.to(target_gpu_device_str)
    else:
        print("Running on CPU. Models remain on CPU.")
    
    print("All models loaded and configured.")
    
    # Adjust gs if cfg > 1.0 (standard CFG mode)
    actual_gs_cli = args.gs
    if args.cfg > 1.0: 
        actual_gs_cli = 1.0 # For standard CFG, distilled guidance is turned off
        print(f"CFG > 1.0 detected ({args.cfg}), this implies standard CFG. Overriding GS to 1.0 from {args.gs}.")

    do_generation_work(
        input_video_path=args.input_video, 
        prompt=args.prompt, 
        n_prompt=args.n_prompt, 
        seed=args.seed,
        end_frame_path=args.end_frame,
        end_frame_weight=args.end_frame_weight,
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

    print("Video generation process completed.")