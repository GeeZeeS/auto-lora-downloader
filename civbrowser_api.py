import os
import sys
import json
import requests
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Import modules from webui for direct access
from modules import script_callbacks, shared

# Add the API route manually
def add_api_routes(app: FastAPI):
    """Add the Civitai Browser API routes"""
    
    # Model request validation
    class ModelCheckRequest(BaseModel):
        model_id: int = Field(..., description="Civitai Model ID")
        model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")
        version_id: Optional[int] = Field(None, description="Specific version ID (defaults to latest)")

    class ModelDownloadRequest(BaseModel):
        model_id: int = Field(..., description="Civitai Model ID")
        model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")
        version_id: Optional[int] = Field(None, description="Specific version ID (defaults to latest)")
        force: Optional[bool] = Field(False, description="Force download even if exists")

    class ModelResponse(BaseModel):
        exists: bool = Field(..., description="Whether the model exists locally")
        model_id: int = Field(..., description="Civitai Model ID")
        version_id: Optional[int] = Field(None, description="Version ID")
        filename: Optional[str] = Field(None, description="Filename if exists")
        path: Optional[str] = Field(None, description="Full path if exists")

    class DownloadResponse(BaseModel):
        success: bool = Field(..., description="Download success status")
        message: str = Field(..., description="Status message")
        file_path: Optional[str] = Field(None, description="Path to downloaded file")
        model_id: int = Field(..., description="Civitai Model ID")
        version_id: Optional[int] = Field(None, description="Version ID")

    # Helper functions
    def get_civitai_api():
        """Get the CivitaiAPI instance"""
        try:
            # First try to import from original extension
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-webui-civbrowser"))
            from scripts.civitai_api import CivitaiAPI
            return CivitaiAPI()
        except ImportError:
            # If not available, create minimal version
            class MinimalCivitaiAPI:
                def __init__(self):
                    self.base_url = "https://civitai.com/api/v1"
                    self.headers = {"Content-Type": "application/json"}
                    
                def get_model(self, model_id):
                    url = f"{self.base_url}/models/{model_id}"
                    response = requests.get(url, headers=self.headers)
                    return response.json()
            
            return MinimalCivitaiAPI()

    def get_model_folder(model_type):
        """Get the folder for a model type"""
        # Define folder mapping
        folders = {
            "checkpoint": os.path.join(shared.models_path, "Stable-diffusion"),
            "ckpt": os.path.join(shared.models_path, "Stable-diffusion"),
            "lora": os.path.join(shared.models_path, "Lora"),
            "locon": os.path.join(shared.models_path, "Lora"),
            "lycoris": os.path.join(shared.models_path, "LyCORIS"),
            "lyco": os.path.join(shared.models_path, "LyCORIS"),
            "embedding": shared.cmd_opts.embeddings_dir,
            "textualinversion": shared.cmd_opts.embeddings_dir,
            "ti": shared.cmd_opts.embeddings_dir,
            "hypernetwork": os.path.join(shared.models_path, "hypernetworks"),
            "vae": os.path.join(shared.models_path, "VAE"),
        }
        
        # Normalize model type
        model_type = model_type.lower()
        
        if model_type in folders:
            folder = folders[model_type]
            os.makedirs(folder, exist_ok=True)
            return folder
        
        # If not found, return None
        return None

    def find_model_file(model_filename, model_type):
        """Find a model file in the appropriate folder"""
        folder = get_model_folder(model_type)
        if not folder:
            return None
        
        # Check if file exists
        full_path = os.path.join(folder, model_filename)
        if os.path.exists(full_path):
            return full_path
        
        # Try common extensions
        extensions = [".safetensors", ".ckpt", ".pt"]
        name_without_ext = os.path.splitext(model_filename)[0]
        
        for ext in extensions:
            test_path = os.path.join(folder, name_without_ext + ext)
            if os.path.exists(test_path):
                return test_path
        
        return None

    def download_model_file(url, dest_path):
        """Download a file with progress reporting"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Download the file
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Print progress
                            if total > 0:
                                percent = (downloaded / total) * 100
                                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB)")
                                sys.stdout.flush()
            
            print(f"\nDownload completed: {dest_path}")
            return True
        except Exception as e:
            print(f"Download error: {str(e)}")
            return False

    # Check if model exists endpoint
    @app.post("/civitai/exists", response_model=ModelResponse, tags=["Civitai Browser"])
    async def check_model_exists(request: ModelCheckRequest):
        """Check if a model is already downloaded"""
        try:
            # Get CivitaiAPI instance
            civitai_api = get_civitai_api()
            
            # Get model details from Civitai
            model_info = civitai_api.get_model(request.model_id)
            
            # Get or select version
            version = None
            version_id = request.version_id
            
            if version_id:
                # Find specific version
                for v in model_info.get("modelVersions", []):
                    if v.get("id") == version_id:
                        version = v
                        break
            else:
                # Use latest version
                if model_info.get("modelVersions"):
                    version = model_info["modelVersions"][0]
                    version_id = version.get("id")
            
            if not version:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Find primary file
            file_info = None
            for file in version.get("files", []):
                if file.get("primary", False):
                    file_info = file
                    break
            
            # If no primary file, use first file
            if not file_info and version.get("files"):
                file_info = version["files"][0]
            
            if not file_info:
                raise HTTPException(status_code=404, detail="No files found for this model version")
            
            # Get filename
            filename = file_info.get("name")
            
            # Check if file exists
            file_path = find_model_file(filename, request.model_type)
            
            return {
                "exists": file_path is not None,
                "model_id": request.model_id,
                "version_id": version_id,
                "filename": filename,
                "path": file_path
            }
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error checking model: {str(e)}")

    # Download model endpoint
    @app.post("/civitai/download", response_model=DownloadResponse, tags=["Civitai Browser"])
    async def download_model(request: ModelDownloadRequest):
        """Download a model from Civitai if not already downloaded"""
        try:
            # First check if model exists
            check_request = ModelCheckRequest(
                model_id=request.model_id,
                model_type=request.model_type,
                version_id=request.version_id
            )
            
            # Get model existence info
            model_exists = await check_model_exists(check_request)
            
            # If model exists and not forcing download, return early
            if model_exists["exists"] and not request.force:
                return {
                    "success": True,
                    "message": "Model already exists",
                    "file_path": model_exists["path"],
                    "model_id": request.model_id,
                    "version_id": model_exists["version_id"]
                }
            
            # Get model folder
            folder = get_model_folder(request.model_type)
            if not folder:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model type: {request.model_type}. Available types: checkpoint, lora, lycoris, embedding, hypernetwork, vae"
                )
            
            # Get CivitaiAPI instance
            civitai_api = get_civitai_api()
            
            # Get model details
            model_info = civitai_api.get_model(request.model_id)
            
            # Select version
            version = None
            if request.version_id:
                for v in model_info.get("modelVersions", []):
                    if v.get("id") == request.version_id:
                        version = v
                        break
            else:
                if model_info.get("modelVersions"):
                    version = model_info["modelVersions"][0]
            
            if not version:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Find primary file
            file_info = None
            for file in version.get("files", []):
                if file.get("primary", False):
                    file_info = file
                    break
            
            # If no primary file, use first file
            if not file_info and version.get("files"):
                file_info = version["files"][0]
            
            if not file_info:
                raise HTTPException(status_code=404, detail="No files found for this model version")
            
            # Get filename and download URL
            filename = file_info.get("name")
            download_url = file_info.get("downloadUrl")
            
            if not download_url:
                raise HTTPException(status_code=404, detail="Download URL not found")
            
            # Create full path
            dest_path = os.path.join(folder, filename)
            
            # Download the file
            print(f"Downloading {filename} to {dest_path}")
            success = download_model_file(download_url, dest_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="Download failed")
            
            # Refresh model list if needed (for checkpoints and VAEs)
            if request.model_type.lower() in ["checkpoint", "ckpt", "vae"]:
                from modules import sd_models
                sd_models.list_models()
            
            return {
                "success": True,
                "message": "Model downloaded successfully",
                "file_path": dest_path,
                "model_id": request.model_id,
                "version_id": version.get("id")
            }
        
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error downloading model: {error_details}")
            raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

    # Return our app with routes added
    return app

# Function to register with the webui
def on_app_started(demo, app):
    """Register API routes when the webui starts"""
    try:
        add_api_routes(app)
        print("Civitai Browser API endpoints registered successfully")
        print("API documentation available at: /docs")
    except Exception as e:
        print(f"Error registering Civitai Browser API endpoints: {e}")

# Register our callback
script_callbacks.on_app_started(on_app_started)

# Check if the API is enabled
if not shared.cmd_opts.api:
    print("=" * 80)
    print("WARNING: API access is not enabled! To use the Civitai Browser API endpoints,")
    print("you need to start the webui with the '--api' flag:")
    print("    python launch.py --api")
    print("=" * 80)
