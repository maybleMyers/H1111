> 📝 Click on the language section to expand / 言語をクリックして展開

# Inference with WAN2.1 / Wan2.1の推論

## Overview / 概要

This is an unofficial inference script for [Wan2.1](https://github.com/Wan-Video/Wan2.1). The features are as follows.

- fp8 support and memory reduction by block swap: Inference of a 720x1280x81frames video is possible with 24GB VRAM
    
- Flash attention can be executed without installation (using PyTorch's scaled dot product attention)
- Supports xformers and Sage attention

This feature is experimental.

<details>
<summary>日本語</summary>
[Wan2.1](https://github.com/Wan-Video/Wan2.1) の非公式推論スクリプトを提供しています。

以下の特徴があります。

- fp8対応およびblock swapによる省メモリ化：720x1280x81framesの動画を24GB VRAMで推論可能
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- xformersおよびSage attention対応

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

Download the T5 `models_t5_umt5-xxl-enc-bf16.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` from the following page: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main

Download the VAE from the above page `Wan2.1_VAE.pth` or download `split_files/vae/wan_2.1_vae.safetensors` from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Please select the appropriate weights according to T2V, I2V, resolution, model size, etc. fp8 models can be used if `--fp8` is specified.

(Thanks to Comfy-Org for providing the repackaged weights.)
<details>
<summary>日本語</summary>
T5 `models_t5_umt5-xxl-enc-bf16.pth` およびCLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を、次のページからダウンロードしてください：https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main

VAEは上のページから `Wan2.1_VAE.pth` をダウンロードするか、次のページから `split_files/vae/wan_2.1_vae.safetensors` をダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

DiTの重みを次のページからダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

T2VやI2V、解像度、モデルサイズなどにより適切な重みを選択してください。`--fp8`指定時はfp8モデルも使用できます。

（repackaged版の重みを提供してくださっているComfy-Orgに感謝いたします。）
</details>

## Inference / 推論

### T2V Inference / T2V推論

The following is an example of T2V inference (input as a single line):

```bash
python wan_generate_video.py --fp8 --task t2v-1.3B --video_size  832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_t2v_1.3B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth 
--attn_mode torch
```

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B` and `t2i-14B`.

`--attn_mode` is `torch`, `sdpa` (same as `torch`), `xformers`, `sageattn`,`flash2`, `flash` (same as `flash2`) or `flash3`. `torch` is the default. Other options require the corresponding library to be installed. `flash3` (Flash attention 3) is not tested.

`--fp8_t5` can be used to specify the T5 model in fp8 format. This option reduces memory usage for the T5 model.  

`--negative_prompt` can be used to specify a negative prompt. If omitted, the default negative prompt is used.

` --flow_shift` can be used to specify the flow shift (default 3.0 for I2V with 480p, 5.0 for others).

`--guidance_scale` can be used to specify the guidance scale for classifier free guiance (default 5.0).

`--blocks_to_swap` is the number of blocks to swap during inference. The default value is None (no block swap). The maximum value is 39 for 14B model and 29 for 1.3B model.

Other options are same as `hv_generate_video.py` (some options are not supported, please check the help).

<details>
<summary>日本語</summary>
`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` のいずれかを指定します。

`--attn_mode` には `torch`, `sdpa`（`torch`と同じ）、`xformers`, `sageattn`, `flash2`, `flash`（`flash2`と同じ）, `flash3` のいずれかを指定します。デフォルトは `torch` です。その他のオプションを使用する場合は、対応するライブラリをインストールする必要があります。`flash3`（Flash attention 3）は未テストです。

`--fp8_t5` を指定するとT5モデルをfp8形式で実行します。T5モデル呼び出し時のメモリ使用量を削減します。

`--negative_prompt` でネガティブプロンプトを指定できます。省略した場合はデフォルトのネガティブプロンプトが使用されます。

`--flow_shift` でflow shiftを指定できます（480pのI2Vの場合はデフォルト3.0、それ以外は5.0）。

`--guidance_scale` でclassifier free guianceのガイダンススケールを指定できます（デフォルト5.0）。

`--blocks_to_swap` は推論時のblock swapの数です。デフォルト値はNone（block swapなし）です。最大値は14Bモデルの場合39、1.3Bモデルの場合29です。

その他のオプションは `hv_generate_video.py` と同じです（一部のオプションはサポートされていないため、ヘルプを確認してください）。
</details>

### I2V Inference / I2V推論

The following is an example of I2V inference (input as a single line):

```bash
python wan_generate_video.py --fp8 --task i2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_i2v_480p_14B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth 
--attn_mode torch --image_path path/to/image.jpg
```
