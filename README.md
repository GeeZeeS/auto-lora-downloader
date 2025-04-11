# Civitai API Extension for Stable Diffusion WebUI

A minimal API extension that adds `/civitai/exists` and `/civitai/download` endpoints to Stable Diffusion WebUI, with documentation available at `/docs`.

## Requirements

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- WebUI started with the `--api` flag

## Installation

1. Clone this repository into your `extensions` folder:
   ```
   cd stable-diffusion-webui/extensions
   git clone https://github.com/yourusername/sd-webui-civitai-api
   ```

2. Restart the WebUI with the `--api` flag:
   ```
   python launch.py --api
   ```

## Features

This extension provides just two essential API endpoints:

1. **Check if a model exists**: `POST /civitai/exists`
2. **Download a model**: `POST /civitai/download`

Both endpoints have full documentation available in the Swagger UI at `/docs`.

## Example Usage

### Check if a model exists
```
POST /civitai/exists
Content-Type: application/json

{
    "model_id": 12345,
    "model_type": "lora"
}
```

Response:
```json
{
    "exists": true,
    "model_id": 12345,
    "version_id": 67890,
    "filename": "example_model.safetensors",
    "path": "/workspace/stable-diffusion-webui/models/Lora/example_model.safetensors"
}
```

### Download a model
```
POST /civitai/download
Content-Type: application/json

{
    "model_id": 12345,
    "model_type": "lora",
    "force": false
}
```

Response:
```json
{
    "success": true,
    "message": "Model downloaded successfully",
    "file_path": "/workspace/stable-diffusion-webui/models/Lora/example_model.safetensors",
    "model_id": 12345,
    "version_id": 67890
}
```

## Supported Model Types

- `checkpoint` or `ckpt`: Stable Diffusion checkpoints
- `lora` or `locon`: LoRA models
- `lycoris` or `lyco`: LyCORIS models
- `embedding`, `textualinversion`, or `ti`: Textual Inversion embeddings
- `hypernetwork`: Hypernetworks
- `vae`: VAE models

## License

MIT
