import base64
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from PIL import Image, ImageOps
import os
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import re

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLMoeForConditionalGeneration = None

MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")

# --- Optional vLLM backend ------------------------------------------------------
# By default describe_images() runs the checkpoint locally via transformers .generate()
# (unchanged behavior). If QWEN_VLLM_BASE_URL is set, requests for QWEN_VLLM_MODEL_ID
# (default: the same checkpoint as QWEN_MODEL_ID) are instead sent as OpenAI-compatible
# chat-completion calls to a running `vllm serve` process -- see
# robot_code/start_vllm_server.sh. That process runs in its own conda env (vllm pins its
# own torch/CUDA build, which would otherwise clash with detectron2's compiled
# extensions here), so this side only needs an HTTP client (httpx, already a dependency).
# A persistent server avoids paying per-call model/KV-cache setup cost, uses vLLM's
# faster attention kernels, and -- because we fire one request per image concurrently
# below -- lets vLLM's continuous batching interleave a frame's objects server-side
# instead of this process building the batch itself.
_VLLM_BASE_URL = os.getenv("QWEN_VLLM_BASE_URL")
_VLLM_MODEL_ID = os.getenv("QWEN_VLLM_MODEL_ID") or MODEL_ID
_VLLM_TIMEOUT = float(os.getenv("QWEN_VLLM_TIMEOUT", "30"))
_VLLM_MAX_WORKERS = int(os.getenv("QWEN_VLLM_MAX_WORKERS", "8"))
_vllm_client = None


def _get_vllm_client():
    global _vllm_client
    if _vllm_client is None:
        import httpx
        _vllm_client = httpx.Client(base_url=_VLLM_BASE_URL, timeout=_VLLM_TIMEOUT)
    return _vllm_client


def _image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _describe_one_vllm(client, model_id, prompt, data_uri, max_new_tokens):
    resp = client.post("/chat/completions", json={
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _describe_images_vllm(images, prompt, max_new_tokens, model_id):
    """Same contract as describe_images() (list of images in -> list of labels out, same
    order), but against a running vLLM server instead of a local model. Requests are
    fired concurrently (bounded by QWEN_VLLM_MAX_WORKERS) rather than batched client-side
    -- vLLM's continuous batching does the batching server-side, and concurrent requests
    are what let it actually interleave them.

    A single request failing (a slow timeout, one bad image) degrades that one object to
    "unknown", same as any other stage that can't decide. But if the server is down
    entirely, every request in the batch fails the same way (connection refused) -- that's
    a setup problem, not an uncertain object, so it's raised loudly instead of quietly
    mislabeling everything "unknown"."""
    import httpx

    client = _get_vllm_client()
    prepared = [_prepare_image(img) for img in images]
    data_uris = [_image_to_data_uri(img) for img in prepared]
    results = [None] * len(images)
    connect_errors = 0
    with ThreadPoolExecutor(max_workers=min(_VLLM_MAX_WORKERS, len(images))) as pool:
        future_to_idx = {
            pool.submit(_describe_one_vllm, client, model_id, prompt, uri, max_new_tokens): i
            for i, uri in enumerate(data_uris)
        }
        for fut, i in future_to_idx.items():
            try:
                results[i] = _normalize_label(fut.result())
            except httpx.ConnectError:
                connect_errors += 1
                results[i] = "unknown"
            except Exception as e:
                print(f"[qwen3_generator] vLLM request failed for image {i}: {e}")
                results[i] = "unknown"

    if connect_errors == len(images):
        raise RuntimeError(
            f"Could not reach the vLLM server at {_VLLM_BASE_URL!r} (QWEN_VLLM_BASE_URL). "
            "Start it with robot_code/start_vllm_server.sh, or unset QWEN_VLLM_BASE_URL "
            "to fall back to loading the checkpoint locally."
        )
    return results


def _least_loaded_cuda_device() -> str:
    """Returns 'cuda:N' for the visible GPU with the most free memory right now."""
    if not torch.cuda.is_available():
        return "cpu"
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_idx, best_free = i, free
    return f"cuda:{best_idx}"


@dataclass
class _ModelBundle:
    model: object
    processor: object
    device: object
    gen_cfg: object


# Keyed by model_id so a caller can request a second (e.g. bigger, verification)
# checkpoint via describe_images(..., model_id=...) without evicting the default
# one — both stay resident for the life of the process once loaded.
_MODEL_CACHE: dict = {}


def _build_bundle(model_id: str) -> _ModelBundle:
    # dtype: bf16 if available, else fp16
    bundle_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    # Load model & processor (trust remote code; ensure versions are compatible)
    is_moe_model = ("moe" in model_id.lower()) or ("a3b" in model_id.lower())
    if is_moe_model:
        if Qwen3VLMoeForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLMoeForConditionalGeneration is not available in this Transformers version. "
                "Upgrade transformers to use MoE checkpoints."
            )
        model_cls = Qwen3VLMoeForConditionalGeneration
    else:
        model_cls = Qwen3VLForConditionalGeneration

    device_map_env = os.getenv("QWEN_DEVICE_MAP", "auto")
    if device_map_env == "auto":
        device_map = "auto"
    elif device_map_env == "cpu":
        device_map = {"": "cpu"}
    elif device_map_env == "single":
        # Pin the *whole* model onto whichever single GPU currently has the most free
        # memory, instead of accelerate's "auto" placement. "auto" (really "balanced")
        # happily splits a model across every visible GPU even when it would fit on
        # one card alone -- fine for throughput/training, but disastrous for
        # interactive token-by-token generation, because every decode step then has to
        # move activations across the PCIe/NVLink boundary for the split layers. That
        # turns a <30ms single-GPU forward pass into hundreds of ms. Checked dynamically
        # (not hardcoded to cuda:0) so a second bundle (e.g. a bigger verification model
        # loaded lazily after the first) naturally lands on whichever GPU is now freest,
        # spreading multiple resident models across cards without ever splitting one.
        device_map = {"": _least_loaded_cuda_device()}
    else:
        # e.g. cuda:0 / cuda:1
        device_map = {"": device_map_env}

    bundle_model = model_cls.from_pretrained(
        model_id,
        torch_dtype=bundle_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor_kwargs = {"trust_remote_code": True}
    max_pixels_env = os.getenv("QWEN_MAX_PIXELS")
    min_pixels_env = os.getenv("QWEN_MIN_PIXELS")
    if max_pixels_env:
        processor_kwargs["max_pixels"] = int(max_pixels_env)
    if min_pixels_env:
        processor_kwargs["min_pixels"] = int(min_pixels_env)
    bundle_processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    bundle_device = next(bundle_model.parameters()).device

    # Decoder-only models should use left padding for batched generation.
    if hasattr(bundle_processor, "tokenizer") and bundle_processor.tokenizer is not None:
        bundle_processor.tokenizer.padding_side = "left"
        if bundle_processor.tokenizer.pad_token_id is None and bundle_processor.tokenizer.eos_token_id is not None:
            bundle_processor.tokenizer.pad_token = bundle_processor.tokenizer.eos_token

    # Force deterministic greedy decoding and clear sampling knobs to avoid warnings.
    bundle_gen_cfg = bundle_model.generation_config
    bundle_gen_cfg.do_sample = False
    for _attr in ("temperature", "top_p", "top_k"):
        if hasattr(bundle_gen_cfg, _attr):
            setattr(bundle_gen_cfg, _attr, None)

    return _ModelBundle(model=bundle_model, processor=bundle_processor, device=bundle_device, gen_cfg=bundle_gen_cfg)


def _get_bundle(model_id: str) -> _ModelBundle:
    bundle = _MODEL_CACHE.get(model_id)
    if bundle is None:
        bundle = _build_bundle(model_id)
        _MODEL_CACHE[model_id] = bundle
    return bundle


# Eagerly load the default model at import time (unchanged behavior for existing
# callers) -- unless the default checkpoint is itself routed to the vLLM server (see
# above), in which case loading it locally too would be pure waste: double the GPU
# memory for the same weights, and minutes of startup time nothing will ever call.
# `model`/`processor`/`device`/`gen_cfg` stay as module-level aliases for backward
# compatibility (None when there's no local default bundle); describe_images()/
# describe_image() default to `_default_bundle` and only touch `_get_bundle` for an
# explicitly requested model_id.
_default_routed_to_vllm = bool(_VLLM_BASE_URL) and MODEL_ID == _VLLM_MODEL_ID
_default_bundle = None if _default_routed_to_vllm else _get_bundle(MODEL_ID)
model = _default_bundle.model if _default_bundle else None
processor = _default_bundle.processor if _default_bundle else None
device = _default_bundle.device if _default_bundle else None
gen_cfg = _default_bundle.gen_cfg if _default_bundle else None

DEFAULT_PROMPT = (
    "You are a vision classifier. Look at the image carefully and identify the complete "
    "object shown, even if the crop is tight or cuts off part of it -- infer the whole "
    "object's identity from what is visible, not just the visible fragment. "
    "Reply with exactly one of the following: "
    "(1) the object's general category name as a single word or short phrase "
    "(e.g. 'banana', 'traffic cone'), if you can identify a distinct object; "
    "(2) the single word 'background' if the image shows only scene background or surface "
    "(e.g. floor, wall, table, sky, blurred clutter) with no distinct object; "
    "(3) the single word 'unknown' if a distinct object is present but you cannot tell what it is. "
    "Do not add explanations or punctuation."
)

MAX_ABS_ASPECT_RATIO = float(os.getenv("QWEN_MAX_ABS_ASPECT_RATIO", "180"))

def extract_noun_from_caption(text: str) -> str:
    """
    Extracts the noun (LVIS label) from the generated caption text.
    Example input: 'user\nClassify the object. Reply with exactly one LVIS label.\nassistant\nbanana'
    Returns: 'banana'
    """
    match = re.search(r"assistant\s*([\w-]+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""

def _stabilize_aspect_ratio(img: Image.Image, max_ratio: float = MAX_ABS_ASPECT_RATIO) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (2, 2))

    # Ensure the short side is at least 2px for processor robustness.
    if w == 1:
        img = ImageOps.expand(img, border=(0, 0, 1, 0), fill=0)
        w, h = img.size
    if h == 1:
        img = ImageOps.expand(img, border=(0, 0, 0, 1), fill=0)
        w, h = img.size

    ratio = max(w / h, h / w)
    if ratio <= max_ratio:
        return img

    # Pad only the short side (minimal distortion, low memory overhead).
    if w > h:
        target_h = max(2, int((w / max_ratio) + 0.999))
        pad_total = max(0, target_h - h)
        top = pad_total // 2
        bottom = pad_total - top
        img = ImageOps.expand(img, border=(0, top, 0, bottom), fill=0)
    else:
        target_w = max(2, int((h / max_ratio) + 0.999))
        pad_total = max(0, target_w - w)
        left = pad_total // 2
        right = pad_total - left
        img = ImageOps.expand(img, border=(left, 0, right, 0), fill=0)

    return img

def _prepare_image(img):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
        return _stabilize_aspect_ratio(img)
    if isinstance(img, Image.Image):
        return _stabilize_aspect_ratio(img.convert("RGB"))
    if hasattr(img, "convert"):
        return _stabilize_aspect_ratio(img.convert("RGB"))
    raise TypeError(f"Unsupported image type for Qwen preprocessing: {type(img)}")

def _normalize_label(generated_text: str) -> str:
    label = extract_noun_from_caption(generated_text).strip().lower()
    if label:
        return label
    # Fallback: use first token from last non-empty line.
    for line in reversed(generated_text.splitlines()):
        line = line.strip().lower()
        if not line or line in {"assistant", "user"}:
            continue
        token = re.split(r"\s+", line, maxsplit=1)[0]
        token = re.sub(r"[^a-z0-9_-]", "", token)
        if token:
            return token
    return "unknown"

def describe_images(
    images,
    prompt=DEFAULT_PROMPT,
    max_new_tokens=6,
    batch_size=4,
    model_id=None,
):
    """model_id: optional checkpoint to use instead of the default (QWEN_MODEL_ID) one,
    e.g. a bigger model for a second-opinion/verification pass. The checkpoint is loaded
    on first use and then cached for the rest of the process."""
    if not images:
        return []

    resolved_model_id = model_id or MODEL_ID
    prompt = prompt or DEFAULT_PROMPT
    if _VLLM_BASE_URL and resolved_model_id == _VLLM_MODEL_ID:
        return _describe_images_vllm(images, prompt, max_new_tokens, resolved_model_id)

    bundle = _get_bundle(model_id) if model_id else _default_bundle
    prepared_images = [_prepare_image(img) for img in images]
    labels = []
    template_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": prepared_images[0]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_template_text = bundle.processor.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    for start_idx in range(0, len(prepared_images), batch_size):
        image_batch = prepared_images[start_idx:start_idx + batch_size]
        text_batch = [chat_template_text] * len(image_batch)

        inputs = bundle.processor(
            text=text_batch,
            images=image_batch,
            return_tensors="pt",
            padding=True,
        ).to(bundle.device)

        with torch.inference_mode():
            out = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_config=bundle.gen_cfg,
            )

        generated_batch = bundle.processor.batch_decode(out, skip_special_tokens=True)
        labels.extend(_normalize_label(text) for text in generated_batch)

    return labels

def describe_image(
    img,
    prompt=DEFAULT_PROMPT,
    max_new_tokens=6,
    model_id=None,
):
    labels = describe_images(
        [img],
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        model_id=model_id,
    )
    return labels[0] if labels else "unknown"

if __name__ == "__main__":
    img_path = "/home/seher/Projects/object_detection/SAMOOD/OOD_object_name_generation/filtered/3661_329.jpg"
    print(describe_image(img_path))
