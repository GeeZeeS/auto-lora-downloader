from modules.api.api import Api
from modules import shared
import gradio as gr
import os
import json
import requests
import time
from pathlib import Path
from modules.processing import StableDiffusionProcessingTxt2Img
from fastapi import Body
from modules.api.models import *
from typing import List, Dict, Any

# Configuration
config = {
    "enabled": True,
    "models_path": "",  # Will be set during initialization
    "civitai_api_key": "",  # Optional: your CivitAI API key if you have one
    "download_timeout": 300,  # 5 minutes maximum download time
    "auto_download": True
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "auto_lora_downloader_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                for key, value in loaded_config.items():
                    if key in config:
                        config[key] = value
        except Exception as e:
            print(f"Error loading Auto LoRA Downloader config: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving Auto LoRA Downloader config: {e}")

# Initialize models path
def init_models_path():
    if not config["models_path"]:
        # Try to get the path from the shared settings
        lora_dir = shared.cmd_opts.lora_dir
        if lora_dir:
            config["models_path"] = lora_dir
        else:
            # Default path in Automatic1111
            config["models_path"] = os.path.join(shared.models_path, "Lora")
        
        # Create directory if it doesn't exist
        os.makedirs(config["models_path"], exist_ok=True)
        save_config()

# CivitAI API functions
def search_lora_on_civitai(lora_name):
    """Search for a LoRA model on CivitAI by name"""
    print(f"Searching for LoRA: {lora_name}")
    
    headers = {}
    if config["civitai_api_key"]:
        headers["Authorization"] = f"Bearer {config['civitai_api_key']}"
    
    params = {
        "limit": 5,
        "query": lora_name,
        "types": "LORA",
        "sort": "downloadCount"
    }
    
    try:
        response = requests.get(
            "https://civitai.com/api/v1/models", 
            params=params,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if "items" in data and len(data["items"]) > 0:
            # Find the best match by looking for exact name match first
            for item in data["items"]:
                if item["name"].lower() == lora_name.lower():
                    return item
            # If no exact match, return the most downloaded one
            return data["items"][0]
        
        return None
    except Exception as e:
        print(f"Error searching CivitAI: {e}")
        return None

def download_lora(model_info):
    """Download a LoRA model from CivitAI"""
    if not model_info or "modelVersions" not in model_info or not model_info["modelVersions"]:
        print("No model versions found")
        return None
    
    # Get the latest version
    latest_version = model_info["modelVersions"][0]
    
    # Find the primary file download (usually the safetensors file)
    download_url = None
    filename = None
    
    for file in latest_version.get("files", []):
        if file.get("primary", False):
            download_url = file.get("downloadUrl")
            filename = file.get("name")
            break
    
    if not download_url or not filename:
        print("No download URL found")
        return None
    
    # Sanitize filename to be safe for the file system
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    
    # Create the full path
    save_path = os.path.join(config["models_path"], safe_filename)
    
    print(f"Downloading {model_info['name']} to {save_path}")
    
    headers = {}
    if config["civitai_api_key"]:
        headers["Authorization"] = f"Bearer {config['civitai_api_key']}"
    
    try:
        # Stream the download to handle large files
        with requests.get(download_url, stream=True, headers=headers) as r:
            r.raise_for_status()
            
            # Get file size for progress reporting
            total_size = int(r.headers.get('content-length', 0))
            
            # Download with progress tracking
            with open(save_path, 'wb') as f:
                start_time = time.time()
                downloaded = 0
                
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        if time.time() - start_time > config["download_timeout"]:
                            raise TimeoutError("Download took too long")
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 5% or so
                        if total_size > 0 and downloaded % (total_size // 20) < 8192:
                            percent = (downloaded / total_size) * 100
                            print(f"Downloaded {percent:.1f}% of {model_info['name']}")
        
        print(f"Successfully downloaded {model_info['name']}")
        return safe_filename
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up partial download
        if os.path.exists(save_path):
            os.remove(save_path)
        return None

def download_loras_by_ids(model_ids):
    """Download LoRAs from CivitAI by their model IDs"""
    results = {}
    
    for model_id in model_ids:
        try:
            # Check if already exists by ID (filename might contain the ID)
            model_exists = False
            model_dir = Path(config["models_path"])
            for file in model_dir.glob(f"*{model_id}*"):
                model_exists = True
                results[model_id] = {"success": True, "status": "already_exists", "path": str(file)}
                break
                
            if model_exists:
                continue
                
            # Fetch model details from CivitAI API
            headers = {}
            if config["civitai_api_key"]:
                headers["Authorization"] = f"Bearer {config['civitai_api_key']}"
                
            response = requests.get(
                f"https://civitai.com/api/v1/models/{model_id}",
                headers=headers
            )
            response.raise_for_status()
            model_info = response.json()
            
            # Check if it's a LoRA
            if model_info.get("type") != "LORA":
                results[model_id] = {"success": False, "status": "not_lora"}
                continue
                
            # Download the LoRA
            downloaded_filename = download_lora(model_info)
            
            if downloaded_filename:
                results[model_id] = {
                    "success": True, 
                    "status": "downloaded", 
                    "path": os.path.join(config["models_path"], downloaded_filename)
                }
            else:
                results[model_id] = {"success": False, "status": "download_failed"}
                
        except Exception as e:
            results[model_id] = {"success": False, "status": "error", "message": str(e)}
            
    return results

# Parse the prompt to extract LoRA models
def extract_lora_from_prompt(prompt):
    """Extract LoRA model names from a prompt"""
    lora_models = []
    
    # Look for <lora:name:weight> format
    import re
    lora_pattern = re.compile(r'<lora:(.*?)(?::|>)')
    matches = lora_pattern.findall(prompt)
    
    for match in matches:
        # Remove any weight value if present
        lora_name = match.split(':')[0] if ':' in match else match
        lora_models.append(lora_name)
    
    return lora_models

# Check if a LoRA model exists locally
def check_lora_exists(lora_name):
    """Check if a LoRA model exists in the models directory"""
    model_dir = Path(config["models_path"])
    
    # Check for common file extensions
    for ext in ['.safetensors', '.pt', '.ckpt']:
        if list(model_dir.glob(f"{lora_name}{ext}")):
            return True
        
        # Also check for files that might have the lora name as part of their filename
        for file in model_dir.glob(f"*{lora_name}*{ext}"):
            return True
    
    return False

# Custom API endpoint model
class Txt2ImgWithLoRARequest(StableDiffusionTxt2ImgProcessingAPI):
    lora_model_ids: List[str] = []

# Setup API
def setup_api(api: Api):
    init_models_path()
    load_config()

    @api.post("/txt2img-with-lora")
    def txt2img_with_lora(txt2img_with_lora_request: Txt2ImgWithLoRARequest = Body(...)):
        # Extract model IDs
        model_ids = txt2img_with_lora_request.lora_model_ids
        lora_results = {}
        
        # Download missing LoRAs
        if model_ids and config["enabled"]:
            lora_results = download_loras_by_ids(model_ids)
        
        # Process standard txt2img
        txt2img_request = txt2img_with_lora_request.dict()
        txt2img_request.pop("lora_model_ids", None)
        
        # Call the standard API
        result = api.txt2img(StableDiffusionTxt2ImgProcessingAPI(**txt2img_request))
        
        # Add LoRA download results
        result_dict = result
        if isinstance(result, dict):  # It should be a dict already
            result_dict["lora_downloads"] = lora_results
        
        return result_dict

# UI components
def on_ui_tabs():
    with gr.Blocks() as ui_component:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h2>Auto LoRA Downloader</h2>")
                
                enabled = gr.Checkbox(
                    label="Enable Auto LoRA Downloader",
                    value=config["enabled"],
                    info="Toggle the extension on/off"
                )
                
                auto_download = gr.Checkbox(
                    label="Auto-download missing LoRAs",
                    value=config["auto_download"],
                    info="Automatically download missing LoRAs from CivitAI"
                )
                
                models_path = gr.Textbox(
                    label="LoRA Models Path",
                    value=config["models_path"],
                    info="Directory where LoRA models will be saved"
                )
                
                civitai_api_key = gr.Textbox(
                    label="CivitAI API Key (Optional)",
                    value=config["civitai_api_key"],
                    type="password",
                    info="Your CivitAI API key for higher rate limits"
                )
                
                download_timeout = gr.Number(
                    label="Download Timeout (seconds)",
                    value=config["download_timeout"],
                    info="Maximum time allowed for downloads"
                )
                
                save_button = gr.Button(value="Save Settings")
                
                # Status message
                status_text = gr.HTML("<p>Auto LoRA Downloader is ready.</p>")
                
        def save_settings():
            config["enabled"] = enabled.value
            config["auto_download"] = auto_download.value
            config["models_path"] = models_path.value
            config["civitai_api_key"] = civitai_api_key.value
            config["download_timeout"] = int(download_timeout.value)
            
            save_config()
            return "<p style='color: green'>Settings saved successfully!</p>"
        
        save_button.click(fn=save_settings, outputs=[status_text])
                
    return [(ui_component, "Auto LoRA Downloader", "auto_lora_downloader")]

# Main hook for processing before generation starts
def process_before_generation(p: StableDiffusionProcessingTxt2Img):
    """Process prompts before generation to download any missing LoRAs"""
    if not config["enabled"] or not config["auto_download"]:
        return
    
    # Get the prompt based on processing type
    prompt = p.prompt
    
    # Extract LoRA models from the prompt
    lora_models = extract_lora_from_prompt(prompt)
    
    if not lora_models:
        return
    
    print(f"Found LoRA models in prompt: {lora_models}")
    
    # Check each LoRA model
    for lora_name in lora_models:
        if not check_lora_exists(lora_name):
            print(f"LoRA model not found locally: {lora_name}")
            
            # Search on CivitAI
            model_info = search_lora_on_civitai(lora_name)
            
            if model_info:
                print(f"Found model on CivitAI: {model_info['name']}")
                downloaded_filename = download_lora(model_info)
                
                if downloaded_filename:
                    print(f"Successfully downloaded LoRA: {downloaded_filename}")
                else:
                    print(f"Failed to download LoRA: {lora_name}")
            else:
                print(f"Could not find LoRA on CivitAI: {lora_name}")
