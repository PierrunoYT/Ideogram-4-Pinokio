import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# outlines_core ships an @torch.compile bitmask kernel dynamo can't trace (torch.device const) -> noisy
# WON'T CONVERT spam on every local upsample. We never use torch.compile at runtime, so disable dynamo.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Show download progress bars + verbose loading logs in the terminal.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import base64
import io
import json
import math
import random
import time

import gradio as gr
import requests
import torch
import transformers
from huggingface_hub import get_token, login, whoami
from huggingface_hub.utils import enable_progress_bars

import diffusers
from diffusers import Ideogram4Pipeline

from studio_editor import StudioEditor

# Surface download + component-loading progress in the terminal.
enable_progress_bars()
diffusers.utils.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity_info()

# Runtime shim (keeps the bundled diffusers pristine): cu130-era bitsandbytes returns Params4bit.shape as a
# plain tuple, but diffusers' check_quantized_param_shape calls .numel() on it. math.prod handles both, so
# this is a no-op once diffusers/bnb fix it upstream.
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
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_SEED = 2**31 - 1

# Prompt upsampling: Ideogram's hosted magic-prompt (needs an API key) or the local Qwen graft (default).
IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"
IDEOGRAM_API_KEY = os.environ.get("IDEOGRAM_API_KEY")
UPSAMPLER_REMOTE = "Ideogram (remote)"
UPSAMPLER_LOCAL = "Qwen (local)"
UPSAMPLERS = [UPSAMPLER_LOCAL, UPSAMPLER_REMOTE]

# V4 presets (forward step-order: main CFG 7.0 -> polish 3.0).
MODES = {
    "Turbo · 12 steps": dict(num_inference_steps=12, guidance_schedule=(7.0,) * 11 + (3.0,) * 1, mu=0.5, std=1.75),
    "Default · 20 steps": dict(num_inference_steps=20, guidance_schedule=(7.0,) * 18 + (3.0,) * 2, mu=0.0, std=1.75),
    "Quality · 48 steps": dict(num_inference_steps=48, guidance_schedule=(7.0,) * 45 + (3.0,) * 3, mu=0.0, std=1.5),
}

DEFAULT_PROMPT = "a ginger cat wearing a tiny wizard hat reading a spellbook"

# --- Lazy model: nothing loads at startup. Sign in, then Download to load the nf4 pipeline (kept quantized
# to save VRAM). The model runs eager — no ZeroGPU / AOTI fast-path locally. ---
pipe = None  # set by load_model()


def sign_in(token):
    """Persist the entered HF token (huggingface_hub login cache) so the gated download authenticates."""
    token = (token or "").strip()
    if not token:
        return "⚠️ Enter a Hugging Face token first — get one at https://huggingface.co/settings/tokens"
    try:
        login(token=token)
    except Exception as e:
        return f"❌ Sign-in failed: {e}"
    try:
        name = whoami(token=token).get("name", "user")
    except Exception:
        name = "user"
    return f"✅ Signed in as **{name}**. Now click **⬇️ Download / Load model**."


def set_ideogram_key(key):
    """Store the Ideogram API key so the remote magic-prompt upsampler can be used."""
    global IDEOGRAM_API_KEY
    IDEOGRAM_API_KEY = (key or "").strip() or None
    if IDEOGRAM_API_KEY:
        return "✅ Ideogram API key saved — select **Ideogram (remote)** as the prompt enhancer to use it."
    return "Ideogram key cleared — using the local Qwen upsampler."


def load_model(progress=gr.Progress(track_tqdm=True)):
    global pipe
    if pipe is not None:
        return "✅ Model already loaded."
    token = get_token()
    if not token:
        return "⚠️ Sign in first — enter your token and click **Sign in**."
    progress(0.0, desc="Downloading / loading Ideogram 4 (nf4)…")
    print("=" * 70, flush=True)
    print(f"[model] downloading + loading {MODEL_ID}", flush=True)
    print("[model] already-cached files load instantly (no download bar); new files show a progress bar.", flush=True)
    t = time.perf_counter()
    try:
        loaded = Ideogram4Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, token=token)
        print(f"[model] from_pretrained done in {time.perf_counter() - t:.1f}s — moving to {DEVICE}…", flush=True)
        progress(0.9, desc=f"Moving model to {DEVICE}…")
        loaded.to(DEVICE)
    except Exception as e:
        print(f"[model] load failed: {e!r}", flush=True)
        return f"❌ Load failed: {e}"
    pipe = loaded
    dt = time.perf_counter() - t
    print(f"[model] ✅ ready on {DEVICE} in {dt:.1f}s", flush=True)
    print("=" * 70, flush=True)
    return f"✅ Model loaded on **{DEVICE}** in {dt:.0f}s — enter a prompt and generate."


def _require_pipe():
    if pipe is None:
        raise gr.Error("Load the model first — enter your HF token, click Sign in, then ⬇️ Download / Load model.")


# --- Prompt upsampling --------------------------------------------------------------------------------------
def remote_upsample(prompt, width, height):
    """Rewrite the prompt into Ideogram's native JSON caption via the hosted magic-prompt API."""
    d = math.gcd(width, height) or 1
    aspect_ratio = f"{width // d}x{height // d}"
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


def local_upsample(prompt, width, height):
    """Expand the prompt into Ideogram's JSON caption with the local Qwen graft (no diffusion)."""
    _require_pipe()
    t = time.perf_counter()
    raw = pipe.upsample_prompt(prompt, height=int(height), width=int(width), lm_head_repo_id=LM_HEAD_REPO)[0]
    print(f"[timing] upsample local: {time.perf_counter() - t:.2f}s", flush=True)
    try:
        return json.loads(raw)
    except Exception:
        return {"high_level_description": raw}


def generate_prompt(prompt, upsampler, width, height, progress=gr.Progress(track_tqdm=True)):
    """✨ Generate prompt → fill the JSON editor with a structured caption."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a text prompt first — the upsampler expands it into Ideogram's JSON caption.")
    if upsampler == UPSAMPLER_REMOTE:
        if not IDEOGRAM_API_KEY:
            raise gr.Error("No Ideogram API key set — add one above (Save key) or switch to Qwen (local).")
        progress(0.0, desc="✍️ Upsampling (Ideogram)…")
        try:
            jp = remote_upsample(prompt, int(width), int(height))
        except Exception as e:
            print(f"[upsample] remote failed, falling back to local: {e!r}", flush=True)
            gr.Warning("Ideogram API unavailable — using the local Qwen upsampler.")
            jp = local_upsample(prompt, width, height)
    else:
        progress(0.0, desc="✍️ Upsampling (local Qwen)…")
        jp = local_upsample(prompt, width, height)
    return gr.update(value=jp)


# --- Image generation ---------------------------------------------------------------------------------------
def _img_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate(editor_value, mode, width, height, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    """🎨 Generate image → render the editor's JSON exactly, and drop the result onto the canvas.

    The editor keeps its JSON in props.value (synced from the canvas), so we read it as a normal input
    instead of the trigger payload (gr.EventData isn't delivered on this gradio build)."""
    _require_pipe()
    try:
        studio = json.loads(editor_value) if isinstance(editor_value, str) else editor_value
    except Exception as e:
        raise gr.Error(f"The JSON is invalid: {e}. Fix it in the editor or re-run ✨ Generate prompt.")
    if not isinstance(studio, dict) or not (
        studio.get("high_level_description")
        or (studio.get("compositional_deconstruction") or {}).get("elements")
        or (studio.get("compositional_deconstruction") or {}).get("background")
    ):
        raise gr.Error("The Studio JSON is empty — draft it with ✨ Generate prompt or fill in the editor first.")

    if randomize_seed or seed is None or seed < 0:
        seed = random.randint(0, MAX_SEED)

    final_prompt = json.dumps(studio, ensure_ascii=False, separators=(",", ":"))
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    preset = MODES.get(mode, MODES["Default · 20 steps"])
    progress(0.0, desc="🎨 Generating image…")
    t = time.perf_counter()
    image = pipe(prompt=final_prompt, width=int(width), height=int(height), generator=generator, **preset).images[0]
    print(f"[timing] diffusion ({mode}): {time.perf_counter() - t:.2f}s", flush=True)

    editor_update = gr.update(
        value=studio,
        image_url=_img_to_data_url(image),
        img_width=int(width),
        img_height=int(height),
    )
    return editor_update, image, int(seed)


CSS = """
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title="Ideogram 4 Studio", css=CSS, theme=gr.themes.Citrus()) as demo:
    gr.Markdown(
        "# Ideogram 4 Studio\n"
        "Run Ideogram's open-weights model locally. Draft a structured JSON prompt from text "
        "(**✨ Generate prompt**), refine it in the visual editor — drag boxes on the canvas to place "
        "elements and text, set color palettes, or edit the JSON directly — then render it "
        "(**🎨 Generate image**).\n\n"
        "[Model](https://huggingface.co/ideogram-ai/ideogram-4-nf4) · "
        "[Prompting guide](https://huggingface.co/ideogram-ai/ideogram-4-nf4#prompting-guide) · "
        "[Blog](https://ideogram.ai/blog/ideogram-4.0/)"
    )

    with gr.Accordion("🔑 Hugging Face access & model", open=True):
        with gr.Row():
            hf_token_box = gr.Textbox(
                label="Hugging Face token",
                value=HF_TOKEN or "",
                type="password",
                placeholder="hf_...",
                scale=4,
                info="Gated model — accept the license at huggingface.co/ideogram-ai/ideogram-4-nf4 first.",
            )
            sign_in_btn = gr.Button("Sign in", scale=1)
            load_btn = gr.Button("⬇️ Download / Load model", variant="primary", scale=1)
        with gr.Row():
            ideogram_key_box = gr.Textbox(
                label="Ideogram API key (optional)",
                value=IDEOGRAM_API_KEY or "",
                type="password",
                placeholder="leave blank to use the local Qwen upsampler",
                scale=4,
                info="Only needed for the 'Ideogram (remote)' prompt enhancer — get one at developer.ideogram.ai",
            )
            save_key_btn = gr.Button("Save key", scale=1)
        access_status = gr.Markdown(
            "Enter your token and click **Sign in**, then **⬇️ Download / Load model** before generating."
        )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=3,
                info="✨ Generate prompt expands this into Ideogram's JSON caption.",
            )
            upsampler = gr.Radio(
                choices=UPSAMPLERS,
                value=UPSAMPLER_LOCAL,
                label="Prompt enhancement",
                info="Which upsampler drafts the JSON caption. 'Ideogram (remote)' needs an API key above.",
            )
            gen_prompt_btn = gr.Button("✨ Generate prompt", variant="primary")
            gr.Examples(
                examples=[
                    ["a ginger cat wearing a tiny wizard hat reading a spellbook"],
                    ["an isometric illustration of a tiny city floating in the clouds"],
                    ["a movie poster for 'THE LAST SUMMER' with dramatic golden-hour lighting"],
                ],
                inputs=[prompt],
            )
            with gr.Accordion("Advanced", open=False):
                width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                seed = gr.Number(label="Seed", value=0, precision=0)
                randomize = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Column(scale=2):
            editor = StudioEditor()
            with gr.Row():
                mode = gr.Radio(
                    choices=list(MODES.keys()), value="Default · 20 steps", label="Mode (speed ↔ quality)", scale=2
                )
                gen_image_btn = gr.Button("🎨 Generate image", variant="primary", scale=1)
            out_image = gr.Image(label="Generated image", type="pil", interactive=False)
            used_seed = gr.Number(label="Seed used", interactive=False)

    sign_in_btn.click(sign_in, inputs=[hf_token_box], outputs=[access_status])
    load_btn.click(load_model, inputs=None, outputs=[access_status])
    save_key_btn.click(set_ideogram_key, inputs=[ideogram_key_box], outputs=[access_status])

    gen_prompt_btn.click(
        generate_prompt,
        inputs=[prompt, upsampler, width, height],
        outputs=[editor],
    )
    # Both the canvas toolbar button (editor.generate_image custom event) and the standalone button
    # run generate, reading the editor's synced JSON value from inputs (no EventData payload needed).
    gen_args = dict(
        fn=generate,
        inputs=[editor, mode, width, height, seed, randomize],
        outputs=[editor, out_image, used_seed],
    )
    editor.generate_image(**gen_args)
    gen_image_btn.click(**gen_args)
    width.change(lambda v: gr.update(img_width=int(v)), width, editor, show_progress="hidden")
    height.change(lambda v: gr.update(img_height=int(v)), height, editor, show_progress="hidden")

demo.launch()
