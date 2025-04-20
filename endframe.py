from diffusers_helper.hf_login import login

import os
import random

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, end_frame, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, save_section_frames, section_settings=None):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # セクション設定の前処理
        def get_section_settings_map(section_settings):
            """
            section_settings: DataFrame List of formats [[number, image, prompt], ...] → {section number: (image, prompt)}dict
            """
            result = {}
            if section_settings is not None:
                for row in section_settings:
                    if row and row[0] is not None:
                        sec_num = int(row[0])
                        img = row[1]
                        prm = row[2] if len(row) > 2 else ""
                        result[sec_num] = (img, prm)
            return result

        section_map = get_section_settings_map(section_settings)
        section_numbers_sorted = sorted(section_map.keys()) if section_map else []

        def get_section_info(i_section):
            """
            i_section: int
            section_map: {Section number: (Image, prompt)}
            If there is no specification, the next section, if not None
            """
            if not section_map:
                return None, None, None
            # i_section以降で最初に見つかる設定
            for sec in range(i_section, max(section_numbers_sorted)+1):
                if sec in section_map:
                    img, prm = section_map[sec]
                    return sec, img, prm
            return None, None, None

        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        def preprocess_image(img):
            H, W, C = img.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            img_np = resize_and_center_crop(img, target_width=width, target_height=height)
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, height, width

        input_image_np, input_image_pt, height, width = preprocess_image(input_image)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        # end_frameも同じタイミングでencode
        if end_frame is not None:
            end_frame_np, end_frame_pt, _, _ = preprocess_image(end_frame)
            end_frame_latent = vae_encode(end_frame_pt, vae)
        else:
            end_frame_latent = None
            
        # create section_latents here
        section_latents = None
        if section_map:
            section_latents = {}
            for sec_num, (img, prm) in section_map.items():
                if img is not None:
                    # 画像をVAE encode
                    img_np, img_pt, _, _ = preprocess_image(img)
                    section_latents[sec_num] = vae_encode(img_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for i_section, latent_padding in enumerate(latent_paddings):
            is_first_section = i_section == 0
            is_last_section = latent_padding == 0
            use_end_latent = is_last_section and end_frame is not None
            latent_padding_size = latent_padding * latent_window_size
            # set current_latent here
            # セクションごとのlatentを使う場合
            if section_map and section_latents is not None and len(section_latents) > 0:
                # i_section以上で最小のsection_latentsキーを探す
                valid_keys = [k for k in section_latents.keys() if k >= i_section]
                if valid_keys:
                    use_key = min(valid_keys)
                    current_latent = section_latents[use_key]
                    print(f"[section_latent] section {i_section}: use section {use_key} latent (section_map keys: {list(section_latents.keys())})")
                    print(f"[section_latent] current_latent id: {id(current_latent)}, min: {current_latent.min().item():.4f}, max: {current_latent.max().item():.4f}, mean: {current_latent.mean().item():.4f}")
                else:
                    current_latent = start_latent
                    print(f"[section_latent] section {i_section}: use start_latent (no section_latent >= {i_section})")
                    print(f"[section_latent] current_latent id: {id(current_latent)}, min: {current_latent.min().item():.4f}, max: {current_latent.max().item():.4f}, mean: {current_latent.mean().item():.4f}")
            else:
                current_latent = start_latent
                print(f"[section_latent] section {i_section}: use start_latent (no section_latents)")
                print(f"[section_latent] current_latent id: {id(current_latent)}, min: {current_latent.min().item():.4f}, max: {current_latent.max().item():.4f}, mean: {current_latent.mean().item():.4f}")

            if is_first_section and end_frame_latent is not None:
                history_latents[:, :, 0:1, :, :] = end_frame_latent

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = current_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # Save the final frame of each section as a still image (with section numbers).
            if save_section_frames and history_pixels is not None:
                try:
                    if i_section == 0 or current_pixels is None:
                        # The first section is history_pixels the end of
                        last_frame = history_pixels[0, :, -1, :, :]
                    else:
                        # From the second section onward, current_pixels the end of
                        last_frame = current_pixels[0, :, -1, :, :]
                    last_frame = einops.rearrange(last_frame, 'c h w -> h w c')
                    last_frame = last_frame.cpu().numpy()
                    last_frame = np.clip((last_frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                    last_frame = resize_and_center_crop(last_frame, target_width=width, target_height=height)
                    if is_first_section and end_frame is None:
                        Image.fromarray(last_frame).save(os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png'))
                    else:
                        Image.fromarray(last_frame).save(os.path.join(outputs_folder, f'{job_id}_{i_section}.png'))
                except Exception as e:
                    print(f"[WARN] セクション{ i_section }最終フレーム画像保存時にエラー: {e}")

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, end_frame, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, save_section_frames, section_settings):
    global stream
    assert input_image is not None, 'No input image!'

    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
        # Update the seed field of the UI with random values.
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
    else:
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    stream = AsyncStream()

    async_run(worker, input_image, end_frame, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, save_section_frames, section_settings)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False), gr.update()
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            end_frame = gr.Image(sources='upload', type="numpy", label="Final Frame (Optional)", height=320)
            prompt = gr.Textbox(label="Prompt", value='', lines=8)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Row():
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])             
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                # Use Random Initial value of the seed
                use_random_seed_default = True
                seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 31337

                use_random_seed = gr.Checkbox(label="Use Random Seed", value=use_random_seed_default)

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=seed_default, precision=0)

                def set_random_seed(is_checked):
                    if is_checked:
                        return random.randint(0, 2**32 - 1)
                    else:
                        return gr.update()
                use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                # Added a checkbox to save still images for each section (default ON)
                save_section_frames = gr.Checkbox(label="Save still images for each section", value=True, info="Save the final frame of each section as a still image (default ON)")

                # Section settings (Change from DataFrame to individual input fields)
                section_number_inputs = []
                section_image_inputs = []
                section_prompt_inputs = []  # Keep it as an empty list.
                with gr.Group():
                    gr.Markdown("### Section Settings. The section number counts from the end of the video. (Optional. If not specified, the usual Image/prompt will be used.)")
                    for i in range(3):
                        with gr.Row():
                            section_number = gr.Number(label=f"Section number{i+1}", value=None, precision=0)
                            section_image = gr.Image(label=f"Keyframe image{i+1}", sources="upload", type="numpy", height=200)
                            section_number_inputs.append(section_number)
                            section_image_inputs.append(section_image)
                # section_settings compiles the values of the three input fields into a list.
                def collect_section_settings(*args):
                    # args: [num1, img1, num2, img2, ...]
                    return [[args[i], args[i+1], ""] for i in range(0, len(args), 2)]
                section_settings = gr.State([[None, None, ""] for _ in range(3)])
                section_inputs = []
                for i in range(3):
                    section_inputs.extend([section_number_inputs[i], section_image_inputs[i]])
                # Store the summed section_inputs in the section_settings State.
                def update_section_settings(*args):
                    return collect_section_settings(*args)
                # Update the section_settings state when section_inputs changes.
                for inp in section_inputs:
                    inp.change(fn=update_section_settings, inputs=section_inputs, outputs=section_settings)

        with gr.Column():
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
    ips = [input_image, end_frame, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, save_section_frames, section_settings]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
)