module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "input",
      params: {
        title: "Hugging Face Token (required)",
        description: "The Ideogram 4 weights are gated on Hugging Face. Accept the license at https://huggingface.co/ideogram-ai/ideogram-4-nf4, then enter a token (https://huggingface.co/settings/tokens) so the download is authenticated.",
        form: [{
          type: "password",
          key: "hf_token",
          title: "HF_TOKEN",
          placeholder: "hf_..."
        }]
      }
    },
    {
      method: "input",
      params: {
        title: "Ideogram API Key (optional)",
        description: "Used for the free hosted magic-prompt (remote prompt upsampling). Get one at https://developer.ideogram.ai/ — leave blank to use the local Qwen upsampler instead.",
        form: [{
          type: "password",
          key: "ideogram_api_key",
          title: "IDEOGRAM_API_KEY",
          placeholder: "(optional)"
        }]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "ENVIRONMENT",
        content: "HF_TOKEN={{input.hf_token}}\nIDEOGRAM_API_KEY={{input.ideogram_api_key}}"
      }
    }
  ]
}
