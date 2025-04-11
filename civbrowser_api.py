import os
import sys
import json
import requests
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Import original extension modules
# Update these paths as needed based on where the original extension is installed
original_extension_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sd-webui-civbrowser")
if original_extension_path not in sys.path:
    sys.path.append(original_extension_path)

try:
    from scripts.civitai_api import CivitaiAPI
    from scripts.civbrowser_lib import get_model_folders, get_image_dir, get_preview_dir, get_temp_dir, parse_generation_parameters
except ImportError:
    # If original modules can't be imported, define minimum necessary functionality
    class CivitaiAPI:
        def __init__(self):
            self.base_url = "https://civitai.com/api/v1"
            self.headers = {
                "Content-Type": "application/json"
            }
            
        def get_models(self, query_params=None):
            url = f"{self.base_url}/models"
            if query_params:
                url += "?" + "&".join([f"{k}={v}" for k, v in query_params.items()])
            response = requests.get(url, headers=self.headers)
            return response.json()
        
        def get_model(self, model_id):
            url = f"{self.base_url}/models/{model_id}"
            response = requests.get(url, headers=self.headers)
            return response.json()
            
        def get_model_versions(self, model_id):
            url = f"{self.base_url}/models/{model_id}/versions"
            response = requests.get(url, headers=self.headers)
            return response.json()
            
        def get_images(self, query_params=None):
            url = f"{self.base_url}/images"
            if query_params:
                url += "?" + "&".join([f"{k}={v}" for k, v in query_params.items()])
            response = requests.get(url, headers=self.headers)
            return response.json()
    
    def get_model_folders():
        """Returns a list of model folders"""
        from modules.paths_internal import models_path
        from modules import shared
        
        model_folders = {}
        for key, path in {
            'ckpt': os.path.join(models_path, "Stable-diffusion"),
            'lora': os.path.join(models_path, "Lora"),
            'lyco': os.path.join(models_path, "LyCORIS"),
            'vae': os.path.join(models_path, "VAE"),
            'hypernetwork': os.path.join(models_path, "hypernetwork"),
            'embedding': shared.cmd_opts.embeddings_dir,
            'controlnet': shared.cmd_opts.controlnet_dir,
        }.items():
            if os.path.isdir(path):
                model_folders[key] = path
        return model_folders

    def get_image_dir():
        """Returns the image directory"""
        from modules import shared
        return os.path.join(shared.data_path, "extensions", "sd-webui-civbrowser", "images")
        
    def get_preview_dir():
        """Returns the preview directory"""
        from modules import shared
        return os.path.join(shared.data_path, "extensions", "sd-webui-civbrowser", "preview")
        
    def get_temp_dir():
        """Returns the temp directory"""
        from modules import shared
        return os.path.join(shared.data_path, "extensions", "sd-webui-civbrowser", "temp")
        
    def parse_generation_parameters(x):
        """Parses generation parameters"""
        return {}

# Create the FastAPI app
civitai_api = CivitaiAPI()

# Define Pydantic models for API documentation
class ModelType(BaseModel):
    name: str = Field(..., description="Model type name")
    path: str = Field(..., description="Path to model directory")

class ModelFolder(BaseModel):
    folders: Dict[str, str] = Field(..., description="Available model folders by type")

class ModelSearchParams(BaseModel):
    limit: Optional[int] = Field(20, description="Number of items to return per page")
    page: Optional[int] = Field(1, description="Page number")
    query: Optional[str] = Field(None, description="Search query")
    tag: Optional[str] = Field(None, description="Filter by tag")
    type: Optional[str] = Field(None, description="Filter by model type")
    sort: Optional[str] = Field("Most Downloaded", description="Sort field")
    period: Optional[str] = Field("AllTime", description="Time period filter")
    nsfw: Optional[bool] = Field(False, description="Include NSFW content")

class ModelResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(..., description="List of models")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class ModelDetail(BaseModel):
    id: int = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    type: str = Field(..., description="Model type")
    nsfw: bool = Field(..., description="NSFW flag")
    tags: List[str] = Field(..., description="Model tags")
    modelVersions: List[Dict[str, Any]] = Field(..., description="Model versions")

class DownloadRequest(BaseModel):
    model_id: int = Field(..., description="Civitai Model ID")
    version_id: Optional[int] = Field(None, description="Specific version ID to download (defaults to latest)")
    model_type: str = Field(..., description="Model type (ckpt, lora, etc.)")

class DownloadResponse(BaseModel):
    success: bool = Field(..., description="Download success status")
    message: str = Field(..., description="Status message")
    file_path: Optional[str] = Field(None, description="Path to downloaded file")

# Function to register our API routes with FastAPI app
def register_civbrowser_api(app: FastAPI):
    """Register Civitai Browser API routes with the main FastAPI app"""
    
    # Get model folders
    @app.get("/civitai/folders", response_model=ModelFolder, tags=["Civitai Browser"])
    async def get_folders():
        """Get available model folders"""
        return {"folders": get_model_folders()}
    
    # Search models
    @app.get("/civitai/models", response_model=ModelResponse, tags=["Civitai Browser"])
    async def search_models(
        limit: int = Query(20, description="Number of items per page"),
        page: int = Query(1, description="Page number"),
        query: Optional[str] = Query(None, description="Search query"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        type: Optional[str] = Query(None, description="Filter by model type"),
        sort: str = Query("Most Downloaded", description="Sort field"),
        period: str = Query("AllTime", description="Time period filter"),
        nsfw: bool = Query(False, description="Include NSFW content")
    ):
        """Search for models on Civitai"""
        params = {
            "limit": limit,
            "page": page,
            "sort": sort,
            "period": period,
            "nsfw": str(nsfw).lower()
        }
        
        if query:
            params["query"] = query
        if tag:
            params["tag"] = tag
        if type:
            params["type"] = type
            
        result = civitai_api.get_models(params)
        return result
    
    # Get model details
    @app.get("/civitai/models/{model_id}", response_model=ModelDetail, tags=["Civitai Browser"])
    async def get_model(model_id: int):
        """Get details for a specific model"""
        try:
            result = civitai_api.get_model(model_id)
            return result
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    
    # Get model versions
    @app.get("/civitai/models/{model_id}/versions", tags=["Civitai Browser"])
    async def get_model_versions(model_id: int):
        """Get all versions for a specific model"""
        try:
            result = civitai_api.get_model_versions(model_id)
            return result
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model versions not found: {str(e)}")
    
    # Download model
    @app.post("/civitai/download", response_model=DownloadResponse, tags=["Civitai Browser"])
    async def download_model(request: DownloadRequest):
        """Download a model from Civitai"""
        try:
            model_id = request.model_id
            version_id = request.version_id
            model_type = request.model_type
            
            # Get model details
            model_info = civitai_api.get_model(model_id)
            
            # Find the version to download
            version = None
            if version_id:
                for v in model_info.get("modelVersions", []):
                    if v.get("id") == version_id:
                        version = v
                        break
            else:
                # Use latest version
                if model_info.get("modelVersions"):
                    version = model_info["modelVersions"][0]
            
            if not version:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Find the file to download
            file_to_download = None
            for file in version.get("files", []):
                if file.get("primary", False):
                    file_to_download = file
                    break
            
            if not file_to_download:
                if version.get("files"):
                    file_to_download = version["files"][0]
                else:
                    raise HTTPException(status_code=404, detail="No files found for this model version")
            
            # Get model folders
            model_folders = get_model_folders()
            
            if model_type not in model_folders:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
            
            # Download destination
            dest_dir = model_folders[model_type]
            filename = file_to_download.get("name")
            dest_path = os.path.join(dest_dir, filename)
            
            # Check if already downloaded
            if os.path.exists(dest_path):
                return {
                    "success": True,
                    "message": "File already exists",
                    "file_path": dest_path
                }
            
            # Download file
            download_url = file_to_download.get("downloadUrl")
            if not download_url:
                raise HTTPException(status_code=404, detail="Download URL not found")
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return {
                "success": True,
                "message": "File downloaded successfully",
                "file_path": dest_path
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
    
    # Return our API documentation
    @app.get("/civitai/docs", include_in_schema=False)
    async def get_civitai_docs():
        """Redirect to the API documentation"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs#/Civitai%20Browser")
    
    return app

# For testing as a standalone script
if __name__ == "__main__":
    print("This module is designed to be imported by a FastAPI application.")
    print("To test it, you should run this through the webui with the --api flag enabled.")
