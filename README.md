# SD-WebUI Civitai Browser API Extension

This extension adds API endpoints with documentation (`/docs/`) to the [SD-WebUI Civitai Browser](https://github.com/SignalFlagZ/sd-webui-civbrowser) extension.

## Requirements

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [SD-WebUI Civitai Browser](https://github.com/SignalFlagZ/sd-webui-civbrowser) extension installed
- WebUI started with the `--api` flag

## Installation

1. Clone this repository into your `extensions` folder:
   ```
   cd stable-diffusion-webui/extensions
   git clone https://github.com/yourusername/sd-webui-civbrowser-api
   ```

2. Restart the WebUI with the `--api` flag:
   ```
   python launch.py --api
   ```

## Features

- Adds comprehensive API endpoints for Civitai Browser functionality
- Full OpenAPI documentation at `/docs`
- Integrates with the existing Civitai Browser extension

## Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/civitai/folders` | List available model folders |
| GET | `/civitai/models` | Search Civitai models |
| GET | `/civitai/models/{model_id}` | Get specific model details |
| GET | `/civitai/models/{model_id}/versions` | Get model versions |
| POST | `/civitai/download` | Download a model |
| GET | `/civitai/docs` | Redirects to API documentation |

## Example Usage

### Search for models
```
GET /civitai/models?limit=10&page=1&query=realistic&type=LORA
```

### Get model details
```
GET /civitai/models/12345
```

### Download a model
```
POST /civitai/download
Content-Type: application/json

{
    "model_id": 12345,
    "model_type": "lora"
}
```

## Documentation

Access the full API documentation by opening your browser to:
```
http://localhost:7860/docs
```

Navigate to the "Civitai Browser" section to see all available endpoints.

## License

Same as the original Civitai Browser extension (MIT)
