# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Adapted for H1111 from NVIDIA/cosmos-framework@058c8c0
# cosmos_framework/inference/prompt_upsampling.py (OpenMDW-1.1). The prompt
# templates under prompting_templates/ are vendored verbatim from the same
# revision. Trimmed to the client + message builders; the batch-file CLI was
# dropped and a UI-facing `upsample_prompt()` convenience wrapper was added.
#
# Cosmos3 checkpoints are trained on JSON-structured captions; this module
# turns a terse user prompt (plus, for i2v, the conditioning image) into that
# JSON via an external OpenAI-compatible VLM endpoint. Pure requests + stdlib —
# no torch — so the UI can call it without touching the generation stack.

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Cosmos3 was trained on `json.dumps(...)` captions (ASCII-escaped by default).
# Set JSON_ENSURE_ASCII=0 to keep non-ASCII characters verbatim instead.
JSON_ENSURE_ASCII = bool(int(os.environ.get("JSON_ENSURE_ASCII", "1")))

SYSTEM_MESSAGE: dict[str, Any] = {
    "role": "system",
    "content": [{"type": "text", "text": "You are a helpful assistant."}],
}
DEFAULT_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
log = logging.getLogger(__name__)

PROMPT_UPSAMPLER_MODES = ("text2image", "text2video", "image2video", "posttrain_image2video")

# Valid output geometry per (resolution tier, aspect ratio) — injected into the
# templates and used to pin the JSON output parameters post-hoc.
RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    "256": {
        "1,1": {"W": 256, "H": 256},
        "4,3": {"W": 320, "H": 256},
        "3,4": {"W": 256, "H": 320},
        "16,9": {"W": 320, "H": 192},
        "9,16": {"W": 192, "H": 320},
    },
    "480": {
        "1,1": {"W": 640, "H": 640},
        "4,3": {"W": 736, "H": 544},
        "3,4": {"W": 544, "H": 736},
        "16,9": {"W": 832, "H": 480},
        "9,16": {"W": 480, "H": 832},
    },
    "720": {
        "1,1": {"W": 960, "H": 960},
        "4,3": {"W": 1104, "H": 832},
        "3,4": {"W": 832, "H": 1104},
        "16,9": {"W": 1280, "H": 720},
        "9,16": {"W": 720, "H": 1280},
    },
    "768": {
        "1,1": {"W": 1024, "H": 1024},
        "4,3": {"W": 1184, "H": 880},
        "3,4": {"W": 880, "H": 1184},
        "16,9": {"W": 1360, "H": 768},
        "9,16": {"W": 768, "H": 1360},
    },
}

PROMPTING_TEMPLATES_DIR = Path(__file__).with_name("prompting_templates")


@lru_cache(maxsize=None)
def _load_prompting_template(filename: str) -> str:
    """Load a built-in template once and strip the trailing newline.

    Templates use ``string.Template`` placeholders (``$json_template``,
    ``$nl_description``, ``$resolution_ratio_dict``), not ``str.format`` braces.
    """
    return (PROMPTING_TEMPLATES_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


def _t2i_json_template() -> str:
    return _load_prompting_template("t2i_json_schema.json")


def _t2v_json_template() -> str:
    return _load_prompting_template("t2v_i2v_video_json_schema.json")


def _resolution_ratio_dict_text() -> str:
    """Return the valid output resolution table rendered for prompt injection."""
    return json.dumps(RESOLUTION_RATIO_DICT, indent=2)


def build_nl_description(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: Optional[str] = None,
    fps: Optional[int] = None,
) -> str:
    """Append literal output parameters to the user prompt.

    The external upsampler receives one text field, so generation metadata is
    made explicit in prose; the templates instruct the model to copy these
    values back into the JSON output.
    """
    params = [f"resolution {resolution}", f"aspect_ratio {aspect_ratio}"]
    if duration is not None:
        params.append(f"duration {duration}")
    if fps is not None:
        params.append(f"fps {fps}")
    return f"{prompt.strip()}\n\nOutput parameters: {', '.join(params)}."


def derive_duration_label(num_frames: int, fps: int) -> str:
    """Convert frame count and FPS to the compact duration label used by inference."""
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return f"{int(num_frames / fps)}s"


def build_t2i_prompt_text(prompt: str, *, resolution: str, aspect_ratio: str) -> str:
    """Render the complete text-to-image upsampler prompt."""
    nl_description = build_nl_description(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return Template(_load_prompting_template("t2i_prompt.txt")).substitute(
        json_template=_t2i_json_template(),
        nl_description=nl_description,
        resolution_ratio_dict=_resolution_ratio_dict_text(),
    )


def build_t2v_prompt_text(
    prompt: str,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_conditioned: bool = False,
) -> str:
    """Render the complete video upsampler prompt.

    ``image_conditioned`` keeps I2V on the same JSON schema as T2V while adding
    instructions that the attached image is visual ground truth for frame 0.
    """
    nl_description = build_nl_description(
        prompt, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps
    )
    intro = "Given the user's natural-language request below"
    image_note = ""
    if image_conditioned:
        intro = "Given the attached starting frame image and the user's natural-language request below"
        image_note = "\nIMPORTANT - IMAGE INPUT: The attached image is the first frame of the video. Use it as visual ground truth for subject appearance, setting, lighting, and colors. The natural-language request primarily describes temporal/action intent. Your JSON must be consistent with what is visible in the image.\n"
    return Template(_load_prompting_template("t2v_i2v_video_prompt.txt")).substitute(
        image_note=image_note,
        intro=intro,
        json_template=_t2v_json_template(),
        nl_description=nl_description,
        resolution_ratio_dict=_resolution_ratio_dict_text(),
    )


def build_posttrain_i2v_prompt_text(prompt: str) -> str:
    """Render the posttrained image-to-video upsampler prompt.

    The posttrained template fills the output-parameter fields post-hoc, so the
    raw user instruction is passed through verbatim.
    """
    return Template(_load_prompting_template("posttrained_i2v_prompt.txt")).substitute(
        json_template=_load_prompting_template("posttrained_i2v_json_schema.json"),
        nl_description=prompt.strip(),
    )


def build_t2i_messages(prompt: str, *, resolution: str, aspect_ratio: str) -> list[dict[str, Any]]:
    text = build_t2i_prompt_text(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
    return [SYSTEM_MESSAGE, {"role": "user", "content": [{"type": "text", "text": text}]}]


def build_t2v_messages(
    prompt: str, *, resolution: str, aspect_ratio: str, duration: str, fps: int
) -> list[dict[str, Any]]:
    text = build_t2v_prompt_text(prompt, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps)
    return [SYSTEM_MESSAGE, {"role": "user", "content": [{"type": "text", "text": text}]}]


def build_i2v_messages(
    prompt: str, *, image_url: str, resolution: str, aspect_ratio: str, duration: str, fps: int
) -> list[dict[str, Any]]:
    text = build_t2v_prompt_text(
        prompt, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps, image_conditioned=True
    )
    return [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ],
        },
    ]


def build_posttrain_i2v_messages(prompt: str, *, image_url: str) -> list[dict[str, Any]]:
    # The posttrained contract sends no system message (matches the model-card
    # proof-of-concept script and the framework client).
    text = build_posttrain_i2v_prompt_text(prompt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text},
            ],
        }
    ]


def _extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Return the stripped inner text of the first ``<tag>...</tag>`` block, if present."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, flags=re.DOTALL)
    if match is None:
        return None
    inner = match.group(1).strip()
    return inner or None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response (bare or ```json fenced)."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Upsampler response JSON must be an object.")
    return parsed


def image_path_to_data_url(path: str | Path) -> str:
    """Encode a local image path as a data URL for OpenAI-compatible VLM requests."""
    image_path = Path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_upsampled_prompt(prompt: Optional[str]) -> bool:
    """True iff ``prompt`` already looks like upsampler output (fenced or bare JSON object).

    Conservative on purpose: false negatives just redo the upsampling, false
    positives would silently skip it. Never raises.
    """
    s = (prompt or "").strip()
    if not s:
        return False
    if s.startswith("```json") or s.startswith("```\n{"):
        return True
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(obj, dict)
    return False


def _compact_json_object(data: dict[str, Any]) -> str:
    """Serialize a JSON object in the compact format expected by inference."""
    return json.dumps(data, ensure_ascii=JSON_ENSURE_ASCII)


def _apply_t2i_output_parameters(data: dict[str, Any], *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
    """Force canonical image metadata into an upsampled T2I JSON object."""
    if resolution not in RESOLUTION_RATIO_DICT:
        raise ValueError(f"Unsupported upsampler resolution {resolution!r}.")
    if aspect_ratio not in RESOLUTION_RATIO_DICT[resolution]:
        raise ValueError(f"Unsupported upsampler aspect_ratio {aspect_ratio!r} for resolution {resolution!r}.")
    resolution_pair = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    data["resolution"] = {"H": resolution_pair["H"], "W": resolution_pair["W"]}
    data["aspect_ratio"] = aspect_ratio
    return data


def _apply_t2v_output_parameters(
    data: dict[str, Any], *, resolution: str, aspect_ratio: str, duration: str, fps: int
) -> dict[str, Any]:
    """Force canonical video metadata into an upsampled T2V/I2V JSON object."""
    data = _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio)
    data["duration"] = duration
    data["fps"] = fps
    return data


@dataclass
class PromptUpsamplerConfig:
    """Connection and sampling settings for ``PromptUpsamplerClient``.

    ``endpoint_url`` may be a bare host, a ``/v1`` base URL, or a full
    ``/chat/completions`` URL; it is normalized by the client. Sampling
    parameters set to ``None`` are omitted from the request payload (several
    gateways reject unsupported keys instead of ignoring them — top_k in
    particular is not accepted by every OpenAI-compatible endpoint).
    """

    endpoint_url: str
    model: Optional[str] = None
    api_token: Optional[str] = None
    timeout_s: float = 300.0
    max_tokens: int = 8192
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    connection_max_retries: int = 2
    connection_pool_size: int = 4


class PromptUpsamplerClient:
    """Small OpenAI-compatible chat-completions client with explicit retries."""

    def __init__(
        self,
        config: PromptUpsamplerConfig,
        *,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._base_url = _normalize_openai_base_url(config.endpoint_url)
        self._session = _make_session(config) if session is None else session
        self._sleep = sleep

    def list_models(self) -> list[str]:
        """Fetch model ids from an OpenAI-compatible ``/models`` endpoint."""
        payload = self._with_retries("list models", lambda: self._request_json("GET", f"{self._base_url}/models"))
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Model list response missing 'data' list.")
        models = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if not models:
            raise ValueError("Model list response did not include any model ids.")
        return models

    def upsample_t2i(self, prompt: str, *, resolution: str, aspect_ratio: str) -> dict[str, Any]:
        messages = build_t2i_messages(prompt, resolution=resolution, aspect_ratio=aspect_ratio)
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2i_output_parameters(data, resolution=resolution, aspect_ratio=aspect_ratio),
        )

    def upsample_t2v(
        self, prompt: str, *, resolution: str, aspect_ratio: str, duration: str, fps: int
    ) -> dict[str, Any]:
        messages = build_t2v_messages(
            prompt, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2v_output_parameters(
                data, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps
            ),
        )

    def upsample_i2v(
        self, prompt: str, *, image_url: str, resolution: str, aspect_ratio: str, duration: str, fps: int
    ) -> dict[str, Any]:
        messages = build_i2v_messages(
            prompt, image_url=image_url, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps
        )
        return self._upsample_messages_with_parameters(
            messages,
            lambda data: _apply_t2v_output_parameters(
                data, resolution=resolution, aspect_ratio=aspect_ratio, duration=duration, fps=fps
            ),
        )

    def upsample_posttrain_i2v(
        self, prompt: str, *, image_url: str, resolution: str, aspect_ratio: str, duration: str, fps: int
    ) -> dict[str, Any]:
        """Upsample with the posttrained Cosmos3-Super-Image2Video contract.

        The response must carry a ``<final_prompt>`` JSON block and may carry a
        per-sample ``<negative_prompt>`` block; a response without the former
        violates the contract and is retried rather than parsed from arbitrary
        text.
        """
        messages = build_posttrain_i2v_messages(prompt, image_url=image_url)

        def _call() -> dict[str, Any]:
            content = self._chat_completion(messages)
            final_prompt = _extract_xml_tag(content, "final_prompt")
            if final_prompt is None:
                raise ValueError("Posttrained upsampler response missing <final_prompt> block.")
            data = _apply_t2v_output_parameters(
                extract_json_object(final_prompt),
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            )
            record: dict[str, Any] = {"prompt": _compact_json_object(data)}
            negative = _extract_xml_tag(content, "negative_prompt")
            if negative is not None:
                record["negative_prompt"] = negative
            return record

        return self._with_retries("upsample prompt", _call)

    def _upsample_messages_with_parameters(
        self,
        messages: list[dict[str, Any]],
        apply_parameters: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the model, parse the JSON object, and pin output parameters.

        Pinning metadata after the model response makes the output robust to
        small formatting/copy mistakes by the LLM and keeps inference metadata
        exactly aligned with the UI settings.
        """

        def _call() -> dict[str, Any]:
            content = self._chat_completion(messages)
            data = apply_parameters(extract_json_object(content))
            return {"prompt": _compact_json_object(data)}

        return self._with_retries("upsample prompt", _call)

    def _get_model(self) -> str:
        """Resolve the model name from config, env, or endpoint discovery."""
        if self.config.model:
            return self.config.model
        env_model = os.environ.get("PROMPT_UPSAMPLER_MODEL")
        if env_model:
            self.config.model = env_model
            return env_model
        self.config.model = self.list_models()[0]
        return self.config.model

    def _chat_completion(self, messages: list[dict[str, Any]]) -> str:
        """Send messages to the configured endpoint and return assistant text."""
        payload: dict[str, Any] = {
            "model": self._get_model(),
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k
        if self.config.min_p is not None:
            payload["min_p"] = self.config.min_p
        response = self._request_json("POST", f"{self._base_url}/chat/completions", payload=payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat completion response missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Chat completion choice must be an object.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat completion choice missing message.")
        return _message_content_to_text(message.get("content"))

    def _request_json(self, method: str, url: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Issue one HTTP request and parse a JSON object response."""
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        try:
            response = self._session.request(method, url, json=payload, headers=headers, timeout=self.config.timeout_s)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:1000]}")
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Response from {url} was not valid JSON: {response.text[:1000]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Response from {url} must be a JSON object.")
        return parsed

    def _with_retries(self, operation: str, fn: Callable[[], Any]) -> Any:
        """Retry transient endpoint/model failures with exponential backoff."""
        if self.config.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt == self.config.max_retries - 1:
                    break
                self._sleep(self.config.retry_base_delay_s * (2**attempt))
        raise RuntimeError(
            f"Prompt upsampler failed to {operation} after {self.config.max_retries} attempts: {last_exc}"
        ) from last_exc


def _normalize_openai_base_url(url: str) -> str:
    """Normalize user-provided endpoint strings to a request base URL."""
    normalized = url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url cannot be empty.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _make_session(config: PromptUpsamplerConfig) -> requests.Session:
    """Create a requests session with connection-level retry behavior."""
    session = requests.Session()
    retry = Retry(
        total=config.connection_max_retries,
        connect=config.connection_max_retries,
        read=0,
        status=0,
        backoff_factor=0.25,
        allowed_methods=None,
    )
    adapter = HTTPAdapter(
        pool_connections=config.connection_pool_size,
        pool_maxsize=config.connection_pool_size,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI message content variants into plain assistant text."""
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Chat completion message content is empty or unsupported.")


# ---------------------------------------------------------------------------
# High-level entry point for the H1111 UI
# ---------------------------------------------------------------------------


def resolve_mode(mode: str, *, has_image: bool, video_length: int) -> str:
    """Resolve the UI mode selector to a concrete upsampler mode.

    ``auto`` picks the posttrained i2v contract when a conditioning image is
    present (the Cosmos3-Super-Image2Video recipe, which also returns a
    tailored negative prompt), otherwise t2i for single-frame outputs and t2v
    for everything else.
    """
    normalized = (mode or "auto").strip().lower().replace("-", "_")
    if normalized == "auto":
        if has_image:
            return "posttrain_image2video"
        return "text2image" if int(video_length) == 1 else "text2video"
    if normalized not in PROMPT_UPSAMPLER_MODES:
        valid = ", ".join(PROMPT_UPSAMPLER_MODES)
        raise ValueError(f"Unsupported prompt upsampling mode {mode!r}. Valid modes: auto, {valid}.")
    return normalized


def _normalize_resolution_tier(resolution: str) -> str:
    """Map UI resolution tiers onto the upsampler metadata table.

    The 704 tier shares the 720 training regime and has no entry of its own in
    ``RESOLUTION_RATIO_DICT``.
    """
    tier = str(resolution).strip()
    if tier == "704":
        return "720"
    return tier


def upsample_prompt(
    prompt: str,
    *,
    endpoint_url: str,
    model: Optional[str] = None,
    api_token: Optional[str] = None,
    mode: str = "auto",
    image_path: Optional[str] = None,
    resolution: str = "720",
    aspect_ratio: str = "16:9",
    video_length: int = 189,
    fps: int = 24,
    timeout_s: float = 300.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Upsample one prompt and return ``{"prompt": <compact JSON str>, "negative_prompt"?: str}``.

    ``aspect_ratio`` accepts both UI colon form (``16:9``) and template comma
    form (``16,9``). ``api_token`` falls back to the PROMPT_UPSAMPLER_API_TOKEN
    / PROMPT_UPSAMPLER_API_KEY environment variables.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is empty.")

    resolved_mode = resolve_mode(mode, has_image=bool(image_path), video_length=video_length)
    if resolved_mode in ("image2video", "posttrain_image2video") and not image_path:
        raise ValueError(f"{resolved_mode} upsampling requires an input image.")

    token = api_token or os.environ.get("PROMPT_UPSAMPLER_API_TOKEN") or os.environ.get("PROMPT_UPSAMPLER_API_KEY")
    config = PromptUpsamplerConfig(
        endpoint_url=endpoint_url,
        model=(model or "").strip() or None,
        api_token=(token or "").strip() or None,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    client = PromptUpsamplerClient(config)

    tier = _normalize_resolution_tier(resolution)
    ratio = str(aspect_ratio).strip().replace(":", ",")
    fps = int(fps)
    duration = derive_duration_label(int(video_length), fps)

    if resolved_mode == "text2image":
        return client.upsample_t2i(prompt, resolution=tier, aspect_ratio=ratio)
    if resolved_mode == "text2video":
        return client.upsample_t2v(prompt, resolution=tier, aspect_ratio=ratio, duration=duration, fps=fps)
    image_url = image_path_to_data_url(image_path)
    if resolved_mode == "image2video":
        return client.upsample_i2v(
            prompt, image_url=image_url, resolution=tier, aspect_ratio=ratio, duration=duration, fps=fps
        )
    return client.upsample_posttrain_i2v(
        prompt, image_url=image_url, resolution=tier, aspect_ratio=ratio, duration=duration, fps=fps
    )
