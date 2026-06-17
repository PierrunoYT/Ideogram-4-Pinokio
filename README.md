# Ideogram 4 Studio Launcher

Local Pinokio launcher for [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4), Ideogram's open-weights text-to-image model, wrapped in a Studio UI for structured JSON prompting.

## Model Details

- **Params**: 9.3B (nf4 quantized)
- **Architecture**: Single-stream Diffusion Transformer (DiT), flow-matching, 34 layers
- **Text encoder**: Qwen3-VL-8B-Instruct (vision-language model)
- **Resolution**: any multiple-of-16 from 256 to 2048 per side (native 2k)
- **License**: Ideogram 4 Non-Commercial

## Features

- Structured JSON prompting (the format the model was trained on) via a visual Studio editor
- Bounding-box layout control — drag boxes on the canvas to place objects and text
- Color-palette conditioning (global + per-element)
- Prompt upsampling: Ideogram's free hosted magic-prompt (remote) or local Qwen graft
- Best-in-class in-image text rendering
- Speed/quality modes: Turbo (12 steps), Default (20 steps), Quality (48 steps)

## Requirements

- NVIDIA GPU with CUDA support
- Gated model — accept the license at the [model page](https://huggingface.co/ideogram-ai/ideogram-4-nf4) and provide an HF token

## Installation

1. Accept the gate at [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4)
2. Click **Install** in the Pinokio UI
3. Enter your **HF token** (required — the weights are gated)
4. Optionally enter an **Ideogram API key** ([developer.ideogram.ai](https://developer.ideogram.ai/)) to enable remote prompt upsampling

## Usage

1. Click **Start** to launch the Gradio web UI
2. Type a prompt and click **✨ Generate prompt** to draft a structured JSON caption
3. Refine it in the visual or JSON editor (move/resize/describe elements, set colors)
4. Click **🎨 Generate image** — the result lands on the canvas for further editing

## API Access

### Python (diffusers)
```python
import os, torch
from diffusers import Ideogram4Pipeline

pipe = Ideogram4Pipeline.from_pretrained(
    "ideogram-ai/ideogram-4-nf4",
    torch_dtype=torch.bfloat16,
    token=os.environ["HF_TOKEN"],
).to("cuda")

image = pipe(
    "a ginger cat wearing a tiny wizard hat reading a spellbook",
    height=1024, width=1024,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("ideogram4.png")
```

## License

Ideogram 4 Non-Commercial — see [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4) for details.
