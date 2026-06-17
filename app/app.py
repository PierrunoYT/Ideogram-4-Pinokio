import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# outlines_core ships an @torch.compile bitmask kernel dynamo can't trace (torch.device const) -> noisy
# WON'T CONVERT spam on every local upsample. We never use torch.compile at runtime, so disable dynamo.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import base64
import io
import json
import math
import random
import time
from pathlib import Path

import requests
import torch
import transformers
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import get_token, login, whoami
from huggingface_hub.utils import enable_progress_bars

import diffusers
from diffusers import Ideogram4Pipeline

enable_progress_bars()
diffusers.utils.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity_info()

# Runtime shim (keeps the bundled diffusers pristine): cu130-era bitsandbytes returns Params4bit.shape as a
# plain tuple, but diffusers' check_quantized_param_shape calls .numel() on it. math.prod handles both.
from diffusers.quantizers.bitsandbytes.bnb_quantizer import BnB4BitDiffusersQuantizer  # noqa: E402


def _check_quantized_param_shape(self, param_name, current_param, loaded_param):
    n = math.prod(tuple(current_param.shape))
    inferred_shape = (n,) if "bias" in param_name else ((n + 1) // 2, 1)
    if tuple(loaded_param.shape) != tuple(inferred_shape):
        raise ValueError(f"Expected flattened shape of {param_name} to be {inferred_shape}, got {tuple(loaded_param.shape)}.")
    return True


BnB4BitDiffusersQuantizer.check_quantized_param_shape = _check_quantized_param_shape

MODEL_ID = "ideogram-ai/ideogram-4-nf4"
LM_HEAD_REPO = "multimodalart/qwen3-vl-8b-instruct-lm-head"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_SEED = 2**31 - 1
STATIC_DIR = Path(__file__).parent / "static"

IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"
IDEOGRAM_API_KEY = os.environ.get("IDEOGRAM_API_KEY")

# V4 presets (forward step-order: main CFG 7.0 -> polish 3.0).
MODES = {
    "Turbo · 12 steps": dict(num_inference_steps=12, guidance_schedule=(7.0,) * 11 + (3.0,) * 1, mu=0.5, std=1.75),
    "Default · 20 steps": dict(num_inference_steps=20, guidance_schedule=(7.0,) * 18 + (3.0,) * 2, mu=0.0, std=1.75),
    "Quality · 48 steps": dict(num_inference_steps=48, guidance_schedule=(7.0,) * 45 + (3.0,) * 3, mu=0.0, std=1.5),
}

pipe = None  # set by /api/load_model


# --- Model helpers ----------------------------------------------------------------------------------------
def _local_upsample(prompt, width, height):
    t = time.perf_counter()
    raw = pipe.upsample_prompt(prompt, height=int(height), width=int(width), lm_head_repo_id=LM_HEAD_REPO)[0]
    print(f"[timing] upsample local: {time.perf_counter() - t:.2f}s", flush=True)
    try:
        return json.loads(raw)
    except Exception:
        return {"high_level_description": raw}


def _remote_upsample(prompt, width, height):
    d = math.gcd(int(width), int(height)) or 1
    aspect_ratio = f"{int(width) // d}x{int(height) // d}"
    resp = requests.post(
        IDEOGRAM_MAGIC_PROMPT_URL,
        headers={"Api-Key": IDEOGRAM_API_KEY, "Content-Type": "application/json"},
        json={"text_prompt": prompt, "aspect_ratio": aspect_ratio},
        timeout=120,
    )
    resp.raise_for_status()
    jp = resp.json().get("json_prompt")
    if not jp:
        raise RuntimeError("Ideogram API returned no json_prompt")
    jp.pop("aspect_ratio", None)
    return jp


def _img_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --- API --------------------------------------------------------------------------------------------------
app = FastAPI(title="Ideogram 4 Studio")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"loaded": pipe is not None, "device": DEVICE, "has_ideogram_key": bool(IDEOGRAM_API_KEY)}


@app.post("/api/signin")
def signin(payload: dict = Body(...)):
    token = (payload.get("token") or "").strip()
    if not token:
        return JSONResponse({"ok": False, "message": "Enter a Hugging Face token first."}, status_code=400)
    try:
        login(token=token)
        name = whoami(token=token).get("name", "user")
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Sign-in failed: {e}"}, status_code=400)
    return {"ok": True, "message": f"Signed in as {name}. Now click Download / Load model."}


@app.post("/api/save_key")
def save_key(payload: dict = Body(...)):
    global IDEOGRAM_API_KEY
    IDEOGRAM_API_KEY = (payload.get("key") or "").strip() or None
    if IDEOGRAM_API_KEY:
        return {"ok": True, "message": "Ideogram API key saved — pick 'Ideogram (remote)' to use it."}
    return {"ok": True, "message": "Ideogram key cleared — using the local Qwen upsampler."}


@app.post("/api/load_model")
def load_model_endpoint():
    global pipe
    if pipe is not None:
        return {"ok": True, "message": "Model already loaded."}
    token = get_token()
    if not token:
        return JSONResponse({"ok": False, "message": "Sign in first."}, status_code=400)
    print("=" * 70, flush=True)
    print(f"[model] downloading + loading {MODEL_ID} (cached files load without a progress bar)", flush=True)
    t = time.perf_counter()
    try:
        loaded = Ideogram4Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, token=token)
        print(f"[model] from_pretrained done in {time.perf_counter() - t:.1f}s — moving to {DEVICE}…", flush=True)
        loaded.to(DEVICE)
    except Exception as e:
        print(f"[model] load failed: {e!r}", flush=True)
        return JSONResponse({"ok": False, "message": f"Load failed: {e}"}, status_code=500)
    pipe = loaded
    dt = time.perf_counter() - t
    print(f"[model] ✅ ready on {DEVICE} in {dt:.1f}s", flush=True)
    print("=" * 70, flush=True)
    return {"ok": True, "message": f"Model loaded on {DEVICE} in {dt:.0f}s — generate away."}


@app.post("/api/upsample")
def upsample(payload: dict = Body(...)):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "message": "Type a prompt first."}, status_code=400)
    upsampler = payload.get("upsampler", "local")
    width, height = payload.get("width", 1024), payload.get("height", 1024)
    if upsampler == "remote":
        if not IDEOGRAM_API_KEY:
            return JSONResponse({"ok": False, "message": "No Ideogram API key set — save one or use local."}, status_code=400)
        try:
            return {"ok": True, "json": _remote_upsample(prompt, width, height)}
        except Exception as e:
            print(f"[upsample] remote failed, falling back to local: {e!r}", flush=True)
    if pipe is None:
        return JSONResponse({"ok": False, "message": "Load the model first (local upsampler needs it)."}, status_code=400)
    return {"ok": True, "json": _local_upsample(prompt, width, height)}


@app.post("/api/generate")
def generate(payload: dict = Body(...)):
    if pipe is None:
        return JSONResponse({"ok": False, "message": "Load the model first."}, status_code=400)
    studio = payload.get("studio")
    if isinstance(studio, str):
        try:
            studio = json.loads(studio)
        except Exception as e:
            return JSONResponse({"ok": False, "message": f"Invalid JSON: {e}"}, status_code=400)
    cd = (studio or {}).get("compositional_deconstruction") or {}
    if not isinstance(studio, dict) or not (studio.get("high_level_description") or cd.get("elements") or cd.get("background")):
        return JSONResponse({"ok": False, "message": "The Studio JSON is empty — generate a prompt or fill the editor."}, status_code=400)

    mode = payload.get("mode", "Default · 20 steps")
    width, height = int(payload.get("width", 1024)), int(payload.get("height", 1024))
    seed = payload.get("seed", 0)
    if payload.get("randomize", True) or seed is None or int(seed) < 0:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    final_prompt = json.dumps(studio, ensure_ascii=False, separators=(",", ":"))
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    preset = MODES.get(mode, MODES["Default · 20 steps"])
    t = time.perf_counter()
    image = pipe(prompt=final_prompt, width=width, height=height, generator=generator, **preset).images[0]
    print(f"[timing] diffusion ({mode}): {time.perf_counter() - t:.2f}s", flush=True)
    return {"ok": True, "image": _img_to_data_url(image), "seed": seed, "width": width, "height": height}


if __name__ == "__main__":
    print(f"[startup] device={DEVICE} — open the printed URL once it appears.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=7860)
