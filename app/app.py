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
# Prefer an explicit env var, but fall back to a token saved by `hf auth login` so running
# `python app.py` by hand (outside Pinokio, which injects the ENVIRONMENT file) still authenticates.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAX_SEED = 2**31 - 1

# Prompt upsampling: Ideogram's hosted magic-prompt (default) with the local Qwen graft as fallback,
# plus "None" — the Studio JSON (or raw text) goes to the model verbatim.
IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"
IDEOGRAM_API_KEY = os.environ.get("IDEOGRAM_API_KEY")
UPSAMPLER_REMOTE = "Ideogram (remote)"
UPSAMPLER_LOCAL = "Qwen (local)"
UPSAMPLERS = [UPSAMPLER_REMOTE, UPSAMPLER_LOCAL]

# V4 presets (forward step-order: main CFG 7.0 -> polish 3.0).
MODES = {
    "Turbo · 12 steps": dict(num_inference_steps=12, guidance_schedule=(7.0,) * 11 + (3.0,) * 1, mu=0.5, std=1.75),
    "Default · 20 steps": dict(num_inference_steps=20, guidance_schedule=(7.0,) * 18 + (3.0,) * 2, mu=0.0, std=1.75),
    "Quality · 48 steps": dict(num_inference_steps=48, guidance_schedule=(7.0,) * 45 + (3.0,) * 3, mu=0.0, std=1.5),
}

# --- Lazy model: nothing loads at startup. The user signs in with an HF token, then clicks Download to
# load the nf4-quantized pipeline onto the GPU (kept quantized — no dequant to bf16 — to save VRAM). ---
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


def remote_upsample(prompt, width, height):
    """Rewrite the prompt into Ideogram's native JSON caption via the hosted magic-prompt API.

    Unlike the plain demo, bbox entries are KEPT — the Studio canvas editor uses them for layout control."""
    d = math.gcd(width, height) or 1
    aspect_ratio = f"{width // d}x{height // d}"  # Ideogram's WxH form
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


def _gpu_generate(final_prompt, mode, width, height, seed, do_local, progress=gr.Progress(track_tqdm=True)):
    _require_pipe()
    if do_local:
        progress(0.0, desc="✍️ Upsampling (local Qwen)…")
        t = time.perf_counter()
        try:
            final_prompt = pipe.upsample_prompt(
                final_prompt, height=int(height), width=int(width), lm_head_repo_id=LM_HEAD_REPO
            )[0]
            print(f"[timing] upsample local: {time.perf_counter() - t:.2f}s", flush=True)
        except Exception as e:
            print(f"[upsample] local failed: {e!r}", flush=True)
            gr.Warning("Local upsampler unavailable — generating from the raw prompt.")

    progress(0.0, desc="🎨 Generating image…")
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    preset = MODES.get(mode, MODES["Default · 20 steps"])
    t = time.perf_counter()
    image = pipe(prompt=final_prompt, width=int(width), height=int(height), generator=generator, **preset).images[0]
    print(f"[timing] diffusion ({mode}): {time.perf_counter() - t:.2f}s", flush=True)

    try:
        caption = json.loads(final_prompt)
    except Exception:
        caption = {"high_level_description": final_prompt}
    return image, int(seed), caption


def _gpu_upsample(prompt, width, height, progress=gr.Progress(track_tqdm=True)):
    """Prompt-only drafting with the local Qwen graft (no diffusion)."""
    _require_pipe()
    progress(0.0, desc="✍️ Upsampling (local Qwen)…")
    t = time.perf_counter()
    out = pipe.upsample_prompt(prompt, height=int(height), width=int(width), lm_head_repo_id=LM_HEAD_REPO)[0]
    print(f"[timing] upsample-only local: {time.perf_counter() - t:.2f}s", flush=True)
    return out


# --- Studio glue -------------------------------------------------------------------------------------------
def _img_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _studio_has_content(j):
    if not isinstance(j, dict):
        return False
    cd = j.get("compositional_deconstruction") or {}
    return bool(j.get("high_level_description") or cd.get("elements") or cd.get("background"))


def generate(
    mode,
    width,
    height,
    seed,
    randomize_seed,
    evt: gr.EventData = None,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed or seed is None or seed < 0:
        seed = random.randint(0, MAX_SEED)

    studio = None
    raw_state = getattr(evt, "state_json", None) if evt is not None else None
    print(f"[generate] received state_json: {raw_state!r}", flush=True)
    if evt is not None:
        try:
            studio = json.loads(evt.state_json)
        except Exception as e:
            print(f"[generate] failed to parse state_json: {e!r}", flush=True)
            studio = None

    if not _studio_has_content(studio):
        raise gr.Error("The Studio JSON is empty — draft it with ✨ Generate prompt or fill in the editor first.")
    final_prompt = json.dumps(studio, ensure_ascii=False, separators=(",", ":"))

    image, seed, caption = _gpu_generate(final_prompt, mode, width, height, seed, False)
    editor_update = gr.update(
        value=caption,
        image_url=_img_to_data_url(image),
        img_width=int(width),
        img_height=int(height),
    )
    return editor_update, image, seed


def generate_prompt(
    prompt,
    upsampler,
    width,
    height,
    progress=gr.Progress(track_tqdm=True),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a text prompt first — the upsampler expands it into Ideogram's JSON caption.")
    if upsampler == UPSAMPLER_REMOTE and IDEOGRAM_API_KEY:
        progress(0.0, desc="✍️ Upsampling (Ideogram)…")
        try:
            return gr.update(value=remote_upsample(prompt, int(width), int(height)))
        except Exception as e:
            print(f"[upsample] remote failed, falling back to local: {e!r}", flush=True)
            gr.Warning("Ideogram API unavailable — using the local Qwen upsampler.")
    raw = _gpu_upsample(prompt, width, height)
    try:
        jp = json.loads(raw)
    except Exception:
        jp = {"high_level_description": raw}
    print(f"[generate_prompt] drafted JSON -> editor: {json.dumps(jp, ensure_ascii=False)[:500]}", flush=True)
    return gr.update(value=jp)


# --- Studio editor: custom gr.HTML component ---------------------------------------------------------------
DEFAULT_PROMPT_JSON = {
    "high_level_description": "",
    "style_description": {"aesthetics": "", "lighting": "", "medium": "", "art_style": "", "color_palette": []},
    "compositional_deconstruction": {"background": "", "elements": []},
}

EDITOR_CSS = """
.i4-root { display:flex; flex-direction:column; gap:10px; }
.i4-toolbar { display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
.i4-tabs { display:flex; gap:4px; }
.i4-tab { padding:6px 14px; border-radius:8px; border:1px solid var(--border-color-primary); background:var(--block-background-fill); color:var(--body-text-color); cursor:pointer; font-size:13px; }
.i4-tab.on { border-color:var(--color-accent, #6366f1); color:var(--color-accent, #6366f1); font-weight:700; }
.i4-actions { display:flex; gap:8px; }
.i4-btn { padding:8px 18px; border-radius:8px; border:1px solid var(--border-color-primary); cursor:pointer; background:var(--block-background-fill); color:var(--body-text-color); font-weight:600; font-size:13px; }
.i4-btn.primary { background:var(--button-primary-background-fill, #ea580c); color:var(--button-primary-text-color, #fff); border-color:transparent; }
.i4-btn:hover, .i4-tab:hover, .i4-mini:hover { filter:brightness(1.08); }
.i4-cols { display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap; }
.i4-left { flex:1.5; min-width:320px; display:flex; flex-direction:column; gap:8px; }
.i4-right { flex:1; min-width:260px; display:flex; flex-direction:column; gap:8px; }
.i4-canvas { position:relative; width:100%; border:1.5px dashed var(--border-color-primary); border-radius:8px; overflow:hidden; touch-action:none; user-select:none; -webkit-user-select:none; cursor:crosshair; background:repeating-conic-gradient(var(--background-fill-secondary) 0% 25%, transparent 0% 50%) 50% / 24px 24px; }
.i4-box { position:absolute; border:2px solid #ff3366; background:rgba(255,51,102,.12); cursor:grab; box-sizing:border-box; border-radius:2px; }
.i4-box.text { border-color:#06b6d4; background:rgba(6,182,212,.12); }
.i4-box.sel { border-color:#6366f1; background:rgba(99,102,241,.2); z-index:5; }
.i4-boxlab { position:absolute; top:0; left:0; font-size:10px; line-height:1.2; background:rgba(0,0,0,.65); color:#fff; padding:1px 5px; border-radius:0 0 4px 0; white-space:nowrap; pointer-events:none; max-width:95%; overflow:hidden; text-overflow:ellipsis; }
.i4-handle { position:absolute; width:12px; height:12px; right:-6px; bottom:-6px; background:#fff; border:1px solid #333; cursor:nwse-resize; border-radius:3px; }
.i4-boxedit { position:absolute; top:2px; left:2px; width:calc(100% - 26px); min-width:70px; max-width:calc(100% - 4px); box-sizing:border-box; font-size:11px; line-height:1.3; padding:2px 5px; border:none; border-radius:3px; background:rgba(0,0,0,.7); color:#fff; outline:none; font-family:inherit; cursor:text; }
.i4-boxedit::placeholder { color:rgba(255,255,255,.55); }
.i4-boxtype { position:absolute; top:2px; right:2px; width:18px; height:18px; font-size:11px; line-height:1; border:none; border-radius:3px; background:rgba(0,0,0,.7); color:#fff; cursor:pointer; padding:0; }
.i4-boxtype:hover { background:rgba(0,0,0,.9); }
.i4-panel { border:1px solid var(--border-color-primary); border-radius:8px; padding:12px; background:var(--block-background-fill); display:flex; flex-direction:column; gap:5px; }
.i4-panel-title { font-weight:700; font-size:13px; margin-bottom:4px; }
.i4-panel label { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; opacity:.7; margin-top:6px; }
.i4-panel input[type=text], .i4-panel textarea, .i4-panel select { width:100%; box-sizing:border-box; padding:6px 8px; border:1px solid var(--border-color-primary); border-radius:6px; background:var(--input-background-fill, var(--background-fill-secondary)); color:var(--body-text-color); font-size:13px; font-family:inherit; }
.i4-pills { display:flex; gap:6px; }
.i4-pill { padding:3px 14px; border-radius:20px; border:1px solid var(--border-color-primary); cursor:pointer; background:transparent; color:var(--body-text-color); font-size:12px; }
.i4-pill.on { background:var(--color-accent, #6366f1); color:#fff; border-color:transparent; }
.i4-palrow { display:flex; gap:6px; align-items:center; }
.i4-color { width:38px; height:28px; padding:1px; border:1px solid var(--border-color-primary); border-radius:6px; background:none; cursor:pointer; }
.i4-mini { padding:4px 12px; font-size:12px; border-radius:6px; border:1px solid var(--border-color-primary); background:var(--block-background-fill); color:var(--body-text-color); cursor:pointer; }
.i4-mini.danger { color:#ff3366; border-color:#ff3366; margin-top:10px; }
.i4-ellist { display:flex; flex-direction:column; gap:2px; max-height:380px; overflow-y:auto; border:1px solid var(--border-color-primary); border-radius:8px; padding:8px 10px; background:var(--block-background-fill); }
.i4-eledit { margin:4px 0 10px 18px; border-left:3px solid var(--color-accent, #6366f1); background:var(--background-fill-secondary); }
.i4-elrow { display:flex; align-items:center; gap:8px; padding:3px 6px; border-radius:6px; cursor:pointer; font-size:12.5px; color:var(--body-text-color); }
.i4-elrow:hover { background:var(--background-fill-secondary); }
.i4-elrow.sel { outline:1.5px solid var(--color-accent, #6366f1); }
.i4-eltype { font-size:11px; color:#ff3366; flex-shrink:0; width:12px; text-align:center; }
.i4-eltype.text { color:#06b6d4; }
.i4-elname { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; }
.i4-elrow .i4-mini { padding:1px 8px; font-size:11px; flex-shrink:0; }
.i4-swatches { display:flex; flex-wrap:wrap; gap:4px; min-height:4px; }
.i4-swatch { width:26px; height:26px; border-radius:5px; border:1px solid rgba(0,0,0,.35); cursor:pointer; color:transparent; display:flex; align-items:center; justify-content:center; font-size:13px; }
.i4-swatch:hover { color:#fff; text-shadow:0 0 3px #000; }
.i4-json { width:100%; min-height:440px; font-family:var(--font-mono, ui-monospace, monospace); font-size:12px; box-sizing:border-box; padding:10px; border:1px solid var(--border-color-primary); border-radius:8px; background:var(--input-background-fill, var(--background-fill-secondary)); color:var(--body-text-color); }
.i4-jsonbar { display:flex; gap:8px; align-items:center; margin-top:6px; }
.i4-jsonmsg { font-size:12px; opacity:.85; }
.i4-hint { font-size:12px; opacity:.6; }
"""

EDITOR_JS = """
const root = element.querySelector('.i4-root');
const deep = (o) => JSON.parse(JSON.stringify(o));

function normalize(v){
  let s;
  try { s = (v && typeof v === 'object') ? deep(v) : (typeof v === 'string' && v.trim() ? JSON.parse(v) : {}); }
  catch (e) { s = { high_level_description: String(v || '') }; }
  if (!s || typeof s !== 'object' || Array.isArray(s)) s = { high_level_description: String(v || '') };
  s.high_level_description = s.high_level_description || '';
  s.style_description = (s.style_description && typeof s.style_description === 'object') ? s.style_description : {};
  s.compositional_deconstruction = (s.compositional_deconstruction && typeof s.compositional_deconstruction === 'object') ? s.compositional_deconstruction : {};
  if (s.compositional_deconstruction.background === undefined) s.compositional_deconstruction.background = '';
  if (!Array.isArray(s.compositional_deconstruction.elements)) s.compositional_deconstruction.elements = [];
  return s;
}

let state = normalize(props.value);
let lastPushed = JSON.stringify(props.value ?? null);
let selIdx = null;
let tab = 'visual';
let drag = null;
let jsonMsg = '';

const sd = () => { state.style_description = state.style_description || {}; return state.style_description; };
const cd = () => {
  state.compositional_deconstruction = state.compositional_deconstruction || {};
  const c = state.compositional_deconstruction;
  if (!Array.isArray(c.elements)) c.elements = [];
  return c;
};
const els = () => cd().elements;
const isPhoto = () => Object.prototype.hasOwnProperty.call(sd(), 'photo');

function esc(s){ return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function getField(f){
  if (f === 'hld') return state.high_level_description || '';
  if (f === 'bg') return cd().background || '';
  if (f === 'stylefield') return isPhoto() ? (sd().photo || '') : (sd().art_style || '');
  if (f.indexOf('sd.') === 0) return sd()[f.slice(3)] || '';
  return '';
}
function setField(f, v){
  if (f === 'hld') state.high_level_description = v;
  else if (f === 'bg') cd().background = v;
  else if (f === 'stylefield') { if (isPhoto()) sd().photo = v; else sd().art_style = v; }
  else if (f.indexOf('sd.') === 0) sd()[f.slice(3)] = v;
}

// Ideogram key-order convention: photo mode -> photo before medium; art mode -> medium before art_style.
function orderSD(x){
  const o = {};
  const put = (k) => { const v = x[k]; if (v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && !v.length)) o[k] = v; };
  put('aesthetics'); put('lighting');
  if (Object.prototype.hasOwnProperty.call(x, 'photo')) { put('photo'); put('medium'); }
  else { put('medium'); put('art_style'); }
  put('color_palette');
  for (const k of Object.keys(x)) {
    const v = x[k];
    if (!(k in o) && v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && !v.length)) o[k] = v;
  }
  return o;
}

function clean(){
  const s = deep(state);
  const out = {};
  if (s.high_level_description) out.high_level_description = s.high_level_description;
  const sdo = orderSD(s.style_description || {});
  if (Object.keys(sdo).length) out.style_description = sdo;
  const cdo = {};
  const scd = s.compositional_deconstruction || {};
  if (scd.background) cdo.background = scd.background;
  if (Array.isArray(scd.elements) && scd.elements.length) cdo.elements = scd.elements.map(el => { const c = { ...el }; if (c.type !== 'text') delete c.text; return c; });
  if (Object.keys(cdo).length) out.compositional_deconstruction = cdo;
  for (const k of Object.keys(s)) {
    if (!(k in out) && k !== 'high_level_description' && k !== 'style_description' && k !== 'compositional_deconstruction') out[k] = s[k];
  }
  return out;
}
const serialize = (indent) => JSON.stringify(clean(), null, indent);

function fieldHTML(label, f, kind){
  const v = esc(getField(f));
  if (kind === 'ta') return `<label>${label}</label><textarea data-f="${f}" rows="2">${v}</textarea>`;
  return `<label>${label}</label><input type="text" data-f="${f}" value="${v}">`;
}

function swatchesHTML(list, pal){
  return (list || []).map(h => `<span class="i4-swatch" data-act="rmcolor" data-pal="${pal}" data-hex="${esc(h)}" style="background:${esc(h)}" title="${esc(h)}">×</span>`).join('');
}

function visualHTML(){
  const photo = isPhoto();
  return `<div class="i4-cols">
    <div class="i4-left">
      <div class="i4-canvas"></div>
      <div class="i4-hint">Drag on the canvas to add an element box, then type its description right on it · the ▢/T button toggles object vs text · drag to move · corner handle to resize. Elements without a box render freely — 📍 place them to control their layout.</div>
      <div class="i4-ellist" id="i4-ellist"></div>
    </div>
    <div class="i4-right">
      <div class="i4-panel">
        <div class="i4-panel-title">Prompt structure</div>
        ${fieldHTML('High-level description', 'hld', 'ta')}
        ${fieldHTML('Background', 'bg', 'ta')}
        ${fieldHTML('Aesthetics', 'sd.aesthetics')}
        ${fieldHTML('Lighting', 'sd.lighting')}
        <label>Style mode</label>
        <div class="i4-pills">
          <button class="i4-pill ${photo ? 'on' : ''}" data-act="stylemode" data-m="photo">photo</button>
          <button class="i4-pill ${photo ? '' : 'on'}" data-act="stylemode" data-m="art">art style</button>
        </div>
        ${photo ? fieldHTML('Photo', 'stylefield') + fieldHTML('Medium', 'sd.medium') : fieldHTML('Medium', 'sd.medium') + fieldHTML('Art style', 'stylefield')}
        <label>Color palette (max 16)</label>
        <div class="i4-palrow"><input type="color" class="i4-color" data-pal="global"><button class="i4-mini" data-act="addcolor" data-pal="global">Add</button></div>
        <div class="i4-swatches">${swatchesHTML(sd().color_palette, 'global')}</div>
      </div>
    </div>
  </div>`;
}

function jsonHTML(){
  return `<div>
    <textarea class="i4-json" spellcheck="false">${esc(serialize(2))}</textarea>
    <div class="i4-jsonbar">
      <button class="i4-mini" data-act="applyjson">Apply JSON</button>
      <button class="i4-mini" data-act="copyjson">Copy</button>
      <span class="i4-jsonmsg">${esc(jsonMsg)}</span>
    </div>
  </div>`;
}

function renderAll(){
  root.innerHTML = `
    <div class="i4-toolbar">
      <div class="i4-tabs">
        <button class="i4-tab ${tab === 'visual' ? 'on' : ''}" data-act="tab" data-tab="visual">Visual editor</button>
        <button class="i4-tab ${tab === 'json' ? 'on' : ''}" data-act="tab" data-tab="json">JSON</button>
      </div>
      <div class="i4-actions">
        <button class="i4-btn primary" data-act="gen-image">🎨 Generate image</button>
      </div>
    </div>
    <div class="i4-body">${tab === 'visual' ? visualHTML() : jsonHTML()}</div>`;
  if (tab === 'visual') { renderCanvas(); renderElList(); }
}

const hasBbox = (el) => Array.isArray(el.bbox) && el.bbox.length === 4;
const elLabel = (el, i) => el.type === 'text' ? '“' + (el.text || '') + '”' : (el.desc || 'object ' + (i + 1));

function renderCanvas(){
  const c = root.querySelector('.i4-canvas');
  if (!c) return;
  const W = props.img_width || 1024, H = props.img_height || 1024;
  c.style.aspectRatio = W + ' / ' + H;
  if (props.image_url) { c.style.backgroundImage = "url('" + props.image_url + "')"; c.style.backgroundSize = '100% 100%'; }
  else { c.style.backgroundImage = ''; c.style.backgroundSize = ''; }
  c.innerHTML = els().map((el, i) => {
    if (!hasBbox(el)) return '';
    const b = el.bbox;
    const isText = el.type === 'text';
    const inner = i === selIdx
      ? `<input class="i4-boxedit" data-bef="${isText ? 'text' : 'desc'}" data-idx="${i}" value="${esc(isText ? (el.text || '') : (el.desc || ''))}" placeholder="${isText ? 'text to render…' : 'describe this element…'}" spellcheck="false"><button class="i4-boxtype" data-act="boxtype" data-idx="${i}" title="Toggle object / text">${isText ? 'T' : '▢'}</button>`
      : `<span class="i4-boxlab">${esc(String(elLabel(el, i)).slice(0, 60))}</span>`;
    return `<div class="i4-box ${isText ? 'text' : ''} ${i === selIdx ? 'sel' : ''}" data-idx="${i}"
      style="top:${b[0] / 10}%;left:${b[1] / 10}%;height:${(b[2] - b[0]) / 10}%;width:${(b[3] - b[1]) / 10}%">
      ${inner}<span class="i4-handle" data-idx="${i}"></span></div>`;
  }).join('');
}

function elEditorHTML(el){
  return `<div class="i4-panel i4-eledit">
    <label>Type</label>
    <select data-ef="type"><option value="obj" ${el.type !== 'text' ? 'selected' : ''}>Object</option><option value="text" ${el.type === 'text' ? 'selected' : ''}>Text</option></select>
    ${el.type === 'text' ? `<label>Text content</label><input type="text" data-ef="text" value="${esc(el.text || '')}">` : ''}
    <label>Description</label><textarea data-ef="desc" rows="2">${esc(el.desc || '')}</textarea>
    <label>Color palette (max 5)</label>
    <div class="i4-palrow"><input type="color" class="i4-color" data-pal="box"><button class="i4-mini" data-act="addcolor" data-pal="box">Add</button></div>
    <div class="i4-swatches">${swatchesHTML(el.color_palette, 'box')}</div>
    <button class="i4-mini danger" data-act="delbox">Delete element</button>
  </div>`;
}

function renderElList(){
  const l = root.querySelector('#i4-ellist');
  if (!l) return;
  if (!els().length) { l.innerHTML = ''; l.style.display = 'none'; return; }
  l.style.display = 'flex';
  l.innerHTML = '<div class="i4-panel-title">Elements</div>' + els().map((el, i) => `
    <div class="i4-elrow ${i === selIdx ? 'sel' : ''}" data-act="selel" data-idx="${i}">
      <span class="i4-eltype ${el.type === 'text' ? 'text' : ''}">${el.type === 'text' ? 'T' : '▢'}</span>
      <span class="i4-elname">${esc(String(elLabel(el, i)).slice(0, 70))}</span>
      ${hasBbox(el) ? '' : `<button class="i4-mini" data-act="place" data-idx="${i}">📍 place</button>`}
    </div>${i === selIdx ? elEditorHTML(el) : ''}`).join('');
}

function positionBox(i){
  const d = root.querySelector(`.i4-box[data-idx="${i}"]`);
  const el = els()[i];
  if (!d || !el) return;
  const b = el.bbox;
  d.style.top = b[0] / 10 + '%'; d.style.left = b[1] / 10 + '%';
  d.style.height = (b[2] - b[0]) / 10 + '%'; d.style.width = (b[3] - b[1]) / 10 + '%';
}

function select(i){
  selIdx = i;
  renderCanvas();
  renderElList();
}

element.addEventListener('click', (e) => {
  const t = e.target.closest('[data-act]');
  if (!t) return;
  const act = t.dataset.act;
  if (act === 'tab') { tab = t.dataset.tab; jsonMsg = ''; renderAll(); }
  else if (act === 'gen-image') { trigger('generate_image', { state_json: JSON.stringify(clean()) }); }
  else if (act === 'stylemode') {
    const m = t.dataset.m, s = sd();
    if (m === 'photo' && !isPhoto()) { s.photo = s.photo || s.art_style || ''; delete s.art_style; s.medium = 'photograph'; }
    else if (m === 'art' && isPhoto()) { s.art_style = s.art_style || s.photo || ''; delete s.photo; if (s.medium === 'photograph') s.medium = ''; }
    renderAll();
  }
  else if (act === 'addcolor') {
    const pal = t.dataset.pal;
    const picker = root.querySelector(`.i4-color[data-pal="${pal}"]`);
    if (!picker) return;
    const hex = picker.value.toUpperCase();
    if (pal === 'global') {
      const arr = sd().color_palette = sd().color_palette || [];
      if (arr.length < 16 && !arr.includes(hex)) arr.push(hex);
      const sw = picker.closest('.i4-panel').querySelector('.i4-swatches');
      if (sw) sw.innerHTML = swatchesHTML(arr, 'global');
    } else if (selIdx !== null && els()[selIdx]) {
      const el = els()[selIdx];
      const arr = el.color_palette = el.color_palette || [];
      if (arr.length < 5 && !arr.includes(hex)) arr.push(hex);
      renderElList();
    }
  }
  else if (act === 'rmcolor') {
    const pal = t.dataset.pal, hex = t.dataset.hex;
    if (pal === 'global') {
      sd().color_palette = (sd().color_palette || []).filter(c => c !== hex);
      t.parentElement.innerHTML = swatchesHTML(sd().color_palette, 'global');
    } else if (selIdx !== null && els()[selIdx]) {
      const el = els()[selIdx];
      el.color_palette = (el.color_palette || []).filter(c => c !== hex);
      renderElList();
    }
  }
  else if (act === 'selel') {
    const i = Number(t.dataset.idx);
    if (i === selIdx) { selIdx = null; renderCanvas(); renderElList(); }
    else select(i);
  }
  else if (act === 'boxtype') {
    const i = Number(t.dataset.idx);
    const el = els()[i];
    if (el) {
      el.type = el.type === 'text' ? 'obj' : 'text';
      renderCanvas();
      renderElList();
      const inp = root.querySelector(`.i4-box[data-idx="${i}"] .i4-boxedit`);
      if (inp) inp.focus();
    }
  }
  else if (act === 'place') {
    const i = Number(t.dataset.idx);
    const el = els()[i];
    if (el) {
      const o = (i % 5) * 40;
      el.bbox = [200 + o, 200 + o, 650 + o, 650 + o];
      renderCanvas();
      renderElList();
      select(i);
    }
  }
  else if (act === 'delbox') {
    if (selIdx !== null) { els().splice(selIdx, 1); selIdx = null; renderCanvas(); renderElList(); }
  }
  else if (act === 'applyjson') {
    const ta = root.querySelector('.i4-json');
    try { state = normalize(JSON.parse(ta.value)); selIdx = null; jsonMsg = '✓ applied'; }
    catch (err) { jsonMsg = '✗ ' + err.message; }
    renderAll();
  }
  else if (act === 'copyjson') {
    navigator.clipboard && navigator.clipboard.writeText(serialize(2));
    jsonMsg = '✓ copied';
    const msg = root.querySelector('.i4-jsonmsg');
    if (msg) msg.textContent = jsonMsg;
  }
});

element.addEventListener('input', (e) => {
  const ds = e.target.dataset || {};
  if (ds.f) { setField(ds.f, e.target.value); return; }
  const syncRow = (i, el) => {
    const row = root.querySelector(`.i4-elrow[data-idx="${i}"] .i4-elname`);
    if (row) row.textContent = String(elLabel(el, i)).slice(0, 70);
  };
  if (ds.bef !== undefined) {
    const i = Number(ds.idx);
    const el = els()[i];
    if (!el) return;
    el[ds.bef] = e.target.value;
    syncRow(i, el);
    const lf = root.querySelector(`.i4-eledit [data-ef="${ds.bef}"]`);
    if (lf) lf.value = e.target.value;
    return;
  }
  const ef = ds.ef;
  if (ef && selIdx !== null && els()[selIdx]) {
    const el = els()[selIdx];
    if (ef === 'type') { el.type = e.target.value; renderCanvas(); renderElList(); }
    else {
      el[ef] = e.target.value;
      syncRow(selIdx, el);
      const inp = root.querySelector(`.i4-box[data-idx="${selIdx}"] .i4-boxedit[data-bef="${ef}"]`);
      if (inp) inp.value = e.target.value;
    }
  }
});

element.addEventListener('pointerdown', (e) => {
  const canvas = root.querySelector('.i4-canvas');
  if (!canvas || !canvas.contains(e.target)) return;
  if (e.target.closest('.i4-boxedit, .i4-boxtype')) return;
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width * 1000;
  const py = (e.clientY - r.top) / r.height * 1000;
  const handle = e.target.closest('.i4-handle');
  const boxEl = e.target.closest('.i4-box');
  if (handle) {
    const i = Number(handle.dataset.idx);
    select(i);
    drag = { kind: 'resize', i, px, py, b: els()[i].bbox.slice() };
  } else if (boxEl) {
    const i = Number(boxEl.dataset.idx);
    select(i);
    drag = { kind: 'move', i, px, py, b: els()[i].bbox.slice() };
  } else {
    const el = { type: 'obj', bbox: [Math.round(py), Math.round(px), Math.round(py), Math.round(px)], desc: '' };
    els().push(el);
    select(els().length - 1);
    drag = { kind: 'draw', i: els().length - 1, px, py, b: el.bbox.slice() };
  }
});

window.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const canvas = root.querySelector('.i4-canvas');
  if (!canvas) return;
  const r = canvas.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width * 1000;
  const py = (e.clientY - r.top) / r.height * 1000;
  const clamp = (v) => Math.max(0, Math.min(1000, Math.round(v)));
  const el = els()[drag.i];
  if (!el) { drag = null; return; }
  const b = drag.b;
  if (drag.kind === 'draw') {
    el.bbox = [clamp(Math.min(drag.py, py)), clamp(Math.min(drag.px, px)), clamp(Math.max(drag.py, py)), clamp(Math.max(drag.px, px))];
  } else if (drag.kind === 'move') {
    const h = b[2] - b[0], w = b[3] - b[1];
    const ny = Math.max(0, Math.min(1000 - h, Math.round(b[0] + (py - drag.py))));
    const nx = Math.max(0, Math.min(1000 - w, Math.round(b[1] + (px - drag.px))));
    el.bbox = [ny, nx, ny + h, nx + w];
  } else {
    el.bbox = [b[0], b[1], clamp(Math.max(b[0] + 10, b[2] + (py - drag.py))), clamp(Math.max(b[1] + 10, b[3] + (px - drag.px)))];
  }
  positionBox(drag.i);
});

window.addEventListener('pointerup', () => {
  if (!drag) return;
  const el = els()[drag.i];
  if (drag.kind === 'draw' && el && ((el.bbox[2] - el.bbox[0]) < 15 || (el.bbox[3] - el.bbox[1]) < 15)) {
    els().splice(drag.i, 1);
    selIdx = null;
    renderCanvas();
    renderElList();
  } else if (drag.kind === 'draw') {
    const inp = root.querySelector(`.i4-box[data-idx="${drag.i}"] .i4-boxedit`);
    if (inp) inp.focus();
  }
  drag = null;
});

watch('value', () => {
  const s = JSON.stringify(props.value ?? null);
  if (s === lastPushed) return;
  lastPushed = s;
  state = normalize(props.value);
  selIdx = null;
  tab = 'visual';
  renderAll();
});

watch(['image_url', 'img_width', 'img_height'], () => { if (tab === 'visual') renderCanvas(); });

renderAll();
"""


class StudioEditor(gr.HTML):
    def __init__(self, value=None, **kwargs):
        # setdefault (not hardcode): gradio re-instantiates the class with updated props as kwargs
        # when an event returns gr.update(image_url=...), which would otherwise collide.
        kwargs.setdefault("image_url", "")
        kwargs.setdefault("img_width", 1024)
        kwargs.setdefault("img_height", 1024)
        kwargs.setdefault("html_template", '<div class="i4-root"></div>')
        kwargs.setdefault("css_template", EDITOR_CSS)
        kwargs.setdefault("js_on_load", EDITOR_JS)
        if value is None:
            value = json.loads(json.dumps(DEFAULT_PROMPT_JSON))
        super().__init__(value=value, **kwargs)

    def api_info(self):
        return {"type": "object", "description": "Ideogram 4 structured JSON prompt"}


CSS = """
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title="Ideogram 4 Studio") as demo:
    gr.Markdown(
        "# Ideogram 4 Studio\n"
        "A studio workspace for Ideogram's open-weights model: draft a structured JSON prompt from text "
        "(✨ Generate prompt), refine it in the visual or JSON editor, then render it (🎨 Generate image) — "
        "the image always renders the Studio JSON exactly as edited, and lands on the canvas so you can "
        "move, resize and re-describe each element and regenerate.\n\n"
        "[Model](https://huggingface.co/ideogram-ai/ideogram-4-nf4) · "
        "[Plain demo](https://huggingface.co/spaces/ideogram-ai/ideogram4) · "
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
        access_status = gr.Markdown(
            "Enter your token and click **Sign in**, then **⬇️ Download / Load model** before generating."
        )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="a ginger cat wearing a tiny wizard hat reading a spellbook",
                lines=3,
                info="✨ Generate prompt expands this into Ideogram's JSON caption and fills the editor.",
            )
            upsampler = gr.Radio(
                choices=UPSAMPLERS,
                value=UPSAMPLER_REMOTE,
                label="Prompt enhancement",
                info="Which upsampler drafts the JSON caption that populates the editor.",
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
        with gr.Column(scale=2):
            editor = StudioEditor()
            with gr.Row():
                mode = gr.Radio(
                    choices=list(MODES.keys()), value="Default · 20 steps", label="Mode (speed ↔ quality)", scale=2
                )
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
                        height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=0, precision=0)
                        randomize = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Accordion("Generated image (full resolution)", open=False):
                out_image = gr.Image(label="Generated image", type="pil", interactive=False, show_label=False)

    sign_in_btn.click(sign_in, inputs=[hf_token_box], outputs=[access_status])
    load_btn.click(load_model, inputs=None, outputs=[access_status])

    gen_prompt_btn.click(
        generate_prompt,
        inputs=[prompt, upsampler, width, height],
        outputs=[editor],
    )
    editor.generate_image(
        generate,
        inputs=[mode, width, height, seed, randomize],
        outputs=[editor, out_image, seed],
    )
    width.change(lambda v: gr.update(img_width=int(v)), width, editor, show_progress="hidden")
    height.change(lambda v: gr.update(img_height=int(v)), height, editor, show_progress="hidden")

demo.launch(theme=gr.themes.Citrus(), css=CSS)
