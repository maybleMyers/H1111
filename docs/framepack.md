# FramePack

## Overview / 概要

This document describes the usage of the [FramePack](https://github.com/lllyasviel/FramePack) architecture within the Musubi Tuner framework. FramePack is a novel video generation architecture developed by lllyasviel.

Key differences from HunyuanVideo:
- FramePack only supports Image-to-Video (I2V) generation. Text-to-Video (T2V) is not supported.
- It utilizes a different DiT model architecture and requires an additional Image Encoder. VAE is same as HunyuanVideo. Text Encoders seem to be the same as HunyuanVideo but we employ the original FramePack method to utilize them.
- Caching and training scripts are specific to FramePack (`fpack_*.py`).
- Due to its progressive generation nature, VRAM usage can be significantly lower, especially for longer videos, compared to other architectures.

This feature is experimental.

<details>
<summary>日本語</summary>
このドキュメントは、Musubi Tunerフレームワーク内での[FramePack](https://github.com/lllyasviel/FramePack) アーキテクチャの使用法について説明しています。FramePackは、lllyasviel氏にによって開発された新しいビデオ生成アーキテクチャです。

HunyuanVideoとの主な違いは次のとおりです。
- FramePackは、画像からビデオ（I2V）生成のみをサポートしています。テキストからビデオ（T2V）はサポートされていません。
- 異なるDiTモデルアーキテクチャを使用し、追加の画像エンコーダーが必要です。VAEはHunyuanVideoと同じです。テキストエンコーダーはHunyuanVideoと同じと思われますが、FramePack公式と同じ方法で推論を行っています。
- キャッシングと学習スクリプトはFramePack専用（`fpack_*.py`）です。
- セクションずつ生成するため、他のアーキテクチャと比較して、特に長いビデオの場合、VRAM使用量が大幅に少なくなる可能性があります。

この機能は実験的なものですです。
</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, Text Encoder 1 (LLaMA), Text Encoder 2 (CLIP), and Image Encoder (SigLIP) models specifically for FramePack. Several download options are available for each component.

### DiT Model

Choose one of the following methods:

1.  **From lllyasviel's Hugging Face repo:** Download the three `.safetensors` files (starting with `diffusion_pytorch_model-00001-of-00003.safetensors`) from [lllyasviel/FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY). Specify the path to the first file (`...-00001-of-00003.safetensors`) as the `--dit` argument.
2.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the model might be downloaded locally. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--lllyasviel--FramePackI2V_HY/snapshots/<hex-uuid-folder>`.
3.  **From Kijai's Hugging Face repo:** Download the single file `FramePackI2V_HY_bf16.safetensors` from [Kijai/HunyuanVideo_comfy](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_bf16.safetensors). Specify the path to this file as the `--dit` argument.

### VAE Model

Choose one of the following methods:

1.  **Use official HunyuanVideo VAE:** Follow the instructions in the main [README.md](../README.md#model-download).
2.  **From hunyuanvideo-community Hugging Face repo:** Download `vae/diffusion_pytorch_model.safetensors` from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo).
3.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the VAE might be downloaded locally within the HunyuanVideo community model snapshot. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Text Encoder 1 (LLaMA) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `split_files/text_encoders/llava_llama3_fp16.safetensors` from [Comfy-Org/HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged).
2.  **From hunyuanvideo-community Hugging Face repo:** Download the four `.safetensors` files (starting with `text_encoder/model-00001-of-00004.safetensors`) from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo). Specify the path to the first file (`...-00001-of-00004.safetensors`) as the `--text_encoder1` argument.
3.  **From local FramePack installation:** (Same as VAE) Specify the path to the HunyuanVideo community model snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Text Encoder 2 (CLIP) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `split_files/text_encoders/clip_l.safetensors` from [Comfy-Org/HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged).
2.  **From hunyuanvideo-community Hugging Face repo:** Download `text_encoder_2/model.safetensors` from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo).
3.  **From local FramePack installation:** (Same as VAE) Specify the path to the HunyuanVideo community model snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Image Encoder (SigLIP) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `sigclip_vision_patch14_384.safetensors` from [Comfy-Org/sigclip_vision_384](https://huggingface.co/Comfy-Org/sigclip_vision_384).
2.  **From lllyasviel's Hugging Face repo:** Download `image_encoder/model.safetensors` from [lllyasviel/flux_redux_bfl](https://huggingface.co/lllyasviel/flux_redux_bfl).
3.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the model might be downloaded locally. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots/<hex-uuid-folder>`.

<details>
<summary>日本語</summary>

DiT、VAE、テキストエンコーダー1（LLaMA）、テキストエンコーダー2（CLIP）、および画像エンコーダー（SigLIP）モデルは複数の方法でダウンロードできます。英語の説明を参考にして、ダウンロードしてください。

FramePack公式のリポジトリをクローンして実行した場合、モデルはローカルにダウンロードされている可能性があります。スナップショットディレクトリへのパスを指定してください。例：`path/to/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots/<hex-uuid-folder>`

HunyuanVideoの推論をComfyUIですでに行っている場合、いくつかのモデルはすでにダウンロードされている可能性があります。
</details>

## Pre-caching / 事前キャッシング

The default resolution for FramePack is 640x640. See [the source code](../frame_pack/bucket_tools.py) for the default resolution of each bucket. 

The dataset for training must be a video dataset. Image datasets are not supported. You can train on videos of any length. Specify `frame_extraction` as `full` and set `max_frames` to a sufficiently large value. However, if the video is too long, you may run out of VRAM during VAE encoding.

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for FramePack. You **must** provide the Image Encoder model.

```bash
python fpack_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model.safetensors \
    --image_encoder path/to/image_encoder_model.safetensors \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
```

Key differences from HunyuanVideo caching:
-   Uses `fpack_cache_latents.py`.
-   Requires the `--image_encoder` argument pointing to the downloaded SigLIP model.
-   You can use the `--latent_window_size` argument (default 9) which defines the size of the latent sections FramePack processes (omitted in the example). This value should typically not be changed unless you understand the implications.
-   The script generates multiple cache files per video, each corresponding to a different section, with the section index appended to the filename (e.g., `..._frame_pos-0000-count_...` becomes `..._frame_pos-0000-0000-count_...`, `..._frame_pos-0000-0001-count_...`, etc.).
-   Image embeddings are calculated using the Image Encoder and stored in the cache files alongside the latents.

For VRAM savings during VAE decoding, consider using `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size`. If VRAM is overflowing and using shared memory, it is recommended to set `--vae_chunk_size` to 16 or 8, and `--vae_spatial_tile_sample_min_size` to 64 or 32.

<details>
<summary>日本語</summary>
FramePackのデフォルト解像度は640x640です。各バケットのデフォルト解像度については、[ソースコード](../frame_pack/bucket_tools.py)を参照してください。

画像データセットでの学習は行えません。また動画の長さによらず学習可能です。 `frame_extraction` に `full` を指定して、`max_frames` に十分に大きな値を指定してください。ただし、あまりにも長いとVAEのencodeでVRAMが不足する可能性があります。

latentの事前キャッシングはFramePack専用のスクリプトを使用します。画像エンコーダーモデルを指定する必要があります。

HunyuanVideoのキャッシングとの主な違いは次のとおりです。
-  `fpack_cache_latents.py`を使用します。
-  ダウンロードしたSigLIPモデルを指す`--image_encoder`引数が必要です。
-  `--latent_window_size`引数（デフォルト9）を指定できます（例では省略）。これは、FramePackが処理するlatentセクションのサイズを定義します。この値は、影響を理解していない限り、通常変更しないでください。
-  スクリプトは、各ビデオに対して複数のキャッシュファイルを生成します。各ファイルは異なるセクションに対応し、セクションインデックスがファイル名に追加されます（例：`..._frame_pos-0000-count_...`は`..._frame_pos-0000-0000-count_...`、`..._frame_pos-0000-0001-count_...`などになります）。
-  画像埋め込みは画像エンコーダーを使用して計算され、latentとともにキャッシュファイルに保存されます。

VAEのdecode時のVRAM節約のために、`--vae_chunk_size`と`--vae_spatial_tile_sample_min_size`を使用することを検討してください。VRAMがあふれて共有メモリを使用している場合には、`--vae_chunk_size`を16、8などに、`--vae_spatial_tile_sample_min_size`を64、32などに変更することをお勧めします。
</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python fpack_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --batch_size 16
```

Key differences from HunyuanVideo caching:
-   Uses `fpack_cache_text_encoder_outputs.py`.
-   Requires both `--text_encoder1` (LLaMA) and `--text_encoder2` (CLIP) arguments.
-   Uses `--fp8_llm` option to run the LLaMA Text Encoder 1 in fp8 mode for VRAM savings (similar to `--fp8_t5` in Wan2.1).
-   Saves LLaMA embeddings, attention mask, and CLIP pooler output to the cache file.

<details>
<summary>日本語</summary>
テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

HunyuanVideoのキャッシングとの主な違いは次のとおりです。
-  `fpack_cache_text_encoder_outputs.py`を使用します。
- LLaMAとCLIPの両方の引数が必要です。
-  LLaMAテキストエンコーダー1をfp8モードで実行するための`--fp8_llm`オプションを使用します（Wan2.1の`--fp8_t5`に似ています）。
-  LLaMAの埋め込み、アテンションマスク、CLIPのプーラー出力をキャッシュファイルに保存します。

</details>


## Training / 学習

### Training

Training uses a dedicated script `fpack_train_network.py`. Remember FramePack only supports I2V training.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 fpack_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model.safetensors \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder_model.safetensors \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

If you use the command prompt (Windows, not PowerShell), you may need to write them in a single line, or use `^` at the end of each line to continue the command.

The maximum value for `--blocks_to_swap` is 36. The default resolution for FramePack is 640x640, which requires around 17GB of VRAM. If you run out of VRAM, consider lowering the dataset resolution.

Key differences from HunyuanVideo training:
-   Uses `fpack_train_network.py`.
-   **Requires** specifying `--vae`, `--text_encoder1`, `--text_encoder2`, and `--image_encoder`.
-   **Requires** specifying `--network_module networks.lora_framepack`.
-  Optional `--latent_window_size` argument (default 9, should match caching).
-   Memory saving options like `--fp8_base` (for DiT) and `--fp8_llm` (for Text Encoder 1) are available. `--fp8_scaled` is recommended when using `--fp8_base` for DiT.
-   `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size` options are available for the VAE to prevent out-of-memory during sampling (similar to caching).
-  `--gradient_checkpointing` is available for memory savings.
<!-- -   Use `convert_lora.py` for converting the LoRA weights after training, similar to HunyuanVideo. -->

Training settings (learning rate, optimizers, etc.) are experimental. Feedback is welcome.

<details>
<summary>日本語</summary>
FramePackの学習は専用のスクリプト`fpack_train_network.py`を使用します。FramePackはI2V学習のみをサポートしています。

コマンド記述例は英語版を参考にしてください。WindowsでPowerShellではなくコマンドプロンプトを使用している場合、コマンドを1行で記述するか、各行の末尾に`^`を付けてコマンドを続ける必要があります。

`--blocks_to_swap`の最大値は36です。FramePackのデフォルト解像度（640x640）では、17GB程度のVRAMが必要です。VRAM容量が不足する場合は、データセットの解像度を下げてください。

HunyuanVideoの学習との主な違いは次のとおりです。
-  `fpack_train_network.py`を使用します。
-  `--vae`、`--text_encoder1`、`--text_encoder2`、`--image_encoder`を指定する必要があります。
-  `--network_module networks.lora_framepack`を指定する必要があります。
-  必要に応じて`--latent_window_size`引数（デフォルト9）を指定できます（キャッシング時と一致させる必要があります）。
-  `--fp8_base`（DiT用）や`--fp8_llm`（テキストエンコーダー1用）などのメモリ節約オプションが利用可能です。`--fp8_base`指定時は、`--fp8_scaled`を使用することをお勧めします。
-  サンプル生成時にメモリ不足を防ぐため、VAE用の`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`オプションが利用可能です（キャッシング時と同様）。
-  メモリ節約のために`--gradient_checkpointing`が利用可能です。

</details>

## Inference

Inference uses a dedicated script `fpack_generate_video.py`.

```bash
python fpack_generate_video.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model.safetensors \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder_model.safetensors \
    --image_path path/to/start_image.jpg \
    --prompt "A cat walks on the grass, realistic style." \
    --video_size 512 768 --video_seconds 5 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path path/to/save/dir --output_type both \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```
    <!-- --embedded_cfg_scale 10.0 --guidance_scale 1.0 \ -->

Key differences from HunyuanVideo inference:
-   Uses `fpack_generate_video.py`.
-   **Requires** specifying `--vae`, `--text_encoder1`, `--text_encoder2`, and `--image_encoder`.
-   **Requires** specifying `--image_path` for the starting frame.
-  **Requires** specifying `--video_seconds` (length of the video in seconds).
- `--video_size` is the size of the generated video, height and width are specified in that order. 
-  Optional `--latent_window_size` argument (default 9, should match caching and training).
-  `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower. `--fp8_llm` option is available to reduce memory usage of Text Encoder 1. `--fp8` alone is also an option for DiT but `--fp8_scaled` potentially offers better quality.
-   LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--lycoris` is also supported.
-   `--embedded_cfg_scale` (default 10.0) controls the distilled guidance scale.
-   `--guidance_scale` (default 1.0) controls the standard classifier-free guidance scale. **Changing this from 1.0 is generally not recommended for the base FramePack model.**
-   `--guidance_rescale` (default 0.0) is available but typically not needed.
-   `--bulk_decode` option can decode all frames at once, potentially faster but uses more VRAM during decoding. `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size` options are recommended to prevent out-of-memory errors.
-   `--sample_solver` (default `unipc`) is available but only `unipc` is implemented.
-   `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.
-   Batch and interactive modes (`--from_file`, `--interactive`) are **not yet implemented** for FramePack generation.

Other options like `--video_size`, `--fps`, `--infer_steps`, `--save_path`, `--output_type`, `--seed`, `--attn_mode`, `--blocks_to_swap`, `--vae_chunk_size`, `--vae_spatial_tile_sample_min_size` function similarly to HunyuanVideo/Wan2.1 where applicable.

The maximum value for `--blocks_to_swap` is 38.
<details>
<summary>日本語</summary>

FramePackの推論は専用のスクリプト`fpack_generate_video.py`を使用します。コマンド記述例は英語版を参考にしてください。

HunyuanVideoの推論との主な違いは次のとおりです。
-  `fpack_generate_video.py`を使用します。
-  `--vae`、`--text_encoder1`、`--text_encoder2`、`--image_encoder`を指定する必要があります。
-  `--image_path`を指定する必要があります。
-  `--video_seconds`を指定する必要があります（秒単位でのビデオの長さを指定）。
-  `--video_size`は生成するビデオのサイズで、高さと幅をその順番で指定します。
-  必要に応じて`--latent_window_size`引数（デフォルト9）を指定できます（キャッシング時、学習時と一致させる必要があります）。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。またText Encoder 1のメモリ使用量を削減するために、`--fp8_llm`オプションを指定可能です。DiT用に`--fp8`単独のオプションも用意されていますが、`--fp8_scaled`の方が品質が良い可能性があります。
-  LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
-  `--embedded_cfg_scale`（デフォルト10.0）は、蒸留されたガイダンススケールを制御します。通常は変更しないでください。
-  `--guidance_scale`（デフォルト1.0）は、標準の分類器フリーガイダンススケールを制御します。**FramePackモデルのベースモデルでは、通常1.0から変更しないことをお勧めします。**
-  `--guidance_rescale`（デフォルト0.0）も利用可能ですが、通常は必要ありません。
-  `--bulk_decode`オプションは、すべてのフレームを一度にデコードできるオプションです。高速ですが、デコード中にVRAMを多く使用します。VRAM不足エラーを防ぐために、`--vae_chunk_size`と`--vae_spatial_tile_sample_min_size`オプションを指定することをお勧めします。
-  `--sample_solver`（デフォルト`unipc`）は利用可能ですが、`unipc`のみが実装されています。
-  `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。
-  バッチモードとインタラクティブモード（`--from_file`、`--interactive`）はFramePack生成には**まだ実装されていません**。

`--video_size`、`--fps`、`--infer_steps`、`--save_path`、`--output_type`、`--seed`、`--attn_mode`、`--blocks_to_swap`、`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`などの他のオプションは、HunyuanVideo/Wan2.1と同様に機能します。

`--blocks_to_swap`の最大値は38です。
</details>