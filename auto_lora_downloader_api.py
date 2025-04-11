# Create a separate script file to register the API directly
# File: auto_lora_downloader_api.py
"""
This file should be placed in the extensions/auto-lora-downloader directory
alongside the scripts directory.
"""

import sys
import os
import importlib
from modules import script_callbacks, shared
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import copy

# Ensure our extension directory is in the path
extension_dir = os.path.dirname(os.path.realpath(__file__))
if extension_dir not in sys.path:
    sys.path.append(extension_dir)

# Import our main script
try:
    from scripts import auto_lora_downloader
except ImportError:
    print("Failed to import auto_lora_downloader script")

# Define API models for documentation
class LoRADownloadResult(BaseModel):
    success: bool = Field(description="Whether the download was successful")
    status: str = Field(description="Status: 'downloaded', 'already_exists', etc.")
    path: Optional[str] = Field(None, description="Path to the downloaded file")
    message: Optional[str] = Field(None, description="Error message if any")

# Add our API endpoints to the app
def add_api_endpoints(app, fastapi_args={}):
    """
    Add our custom API endpoints to the main FastAPI app
    """
    try:
        # Import necessary components only when registering the endpoints
        from modules.api.models import StableDiffusionTxt2ImgProcessingAPI
        from modules import api
        from fastapi import Body
        
        # Create a dynamic model for the request
        class Txt2ImgWithLoRARequest(StableDiffusionTxt2ImgProcessingAPI):
            lora_model_ids: List[str] = Field(default=[], description="CivitAI model IDs to download")
        
        # Register the endpoint
        @app.post("/sdapi/v1/txt2img-with-lora", 
                 tags=["Generation"], 
                 summary="Generate images with automatic LoRA downloading",
                 description="This endpoint combines txt2img generation with automatic downloading of missing LoRAs from CivitAI")
        def api_txt2img_with_lora(params: Txt2ImgWithLoRARequest = Body(...)):
            """Generate images with automatic LoRA downloading"""
            # Extract model IDs
            model_ids = params.lora_model_ids
            lora_results = {}
            
            # Download missing LoRAs
            if model_ids and auto_lora_downloader.config["enabled"]:
                lora_results = auto_lora_downloader.download_loras_by_ids(model_ids)
            
            # Process with standard API
            standard_params = params.dict()
            if "lora_model_ids" in standard_params:
                standard_params.pop("lora_model_ids")
                
            # Call the standard txt2img API
            txt2img_result = api.api_txt2img(StableDiffusionTxt2ImgProcessingAPI(**standard_params))
            
            # Add LoRA results to the response
            result = txt2img_result.copy()
            result["lora_downloads"] = lora_results
            
            return result
            
        print("✅ Successfully registered txt2img-with-lora API endpoint")
    except Exception as e:
        print(f"❌ Failed to register API endpoints: {e}")

# Register our callback to add the API endpoints
def on_app_started(demo, app):
    """
    Callback that will be called when the Gradio app is started
    """
    add_api_endpoints(app)

# Register the callback
script_callbacks.on_app_started(on_app_started)

# Optional: Force reload our extension's script to pick up changes during development
importlib.reload(auto_lora_downloader)
