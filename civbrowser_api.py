import os
import sys
import json
import requests
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Add path to Civitai Browser extension
civbrowser_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-webui-civbrowser")
if civbrowser_path not in sys.path:
    sys.path.append(civbrowser_path)

# Import from Civitai Browser extension
from scripts.civitai_api import CivitaiAPI
from scripts.civbrowser_lib import (get_model_folders, get_image_dir, 
                                    get_preview_dir, get_temp_dir, 
                                    download_file_with_progressbar, 
                                    find_available_file_name)

# Create the FastAPI app
civitai_api = CivitaiAPI()

# Define Pydantic models for API documentation
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
    model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")

class DownloadResponse(BaseModel):
    success: bool = Field(..., description="Download success status")
    message: str = Field(..., description="Status message")
    file_path: Optional[str] = Field(None, description="Path to downloaded file")

# Debug helper
def debug_api_request(request_data, endpoint="download"):
    """Helper function to log API request data for debugging"""
    import json
    import datetime
    import os
    
    # Get the log directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"api_{endpoint}_{timestamp}.json")
    
    # Log the request data
    with open(log_file, "w") as f:
        json.dump(request_data, f, indent=2)
    
    print(f"API request logged to {log_file}")
    return log_file

# Function to register our API routes with FastAPI app
def register_civbrowser_api(app: FastAPI):
    """Register Civitai Browser API routes with the main FastAPI app"""
    
    # Get model folders
    @app.get("/civitai/folders", tags=["Civitai Browser"])
    async def get_folders():
        """Get available model folders"""
        folders = get_model_folders()
        # Convert to regular dictionary with path strings
        folder_dict = {k: str(v) for k, v in folders.items()}
        return {"folders": folder_dict}
    
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
            # Log the request for debugging
            debug_api_request(request.dict())
            
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
            
            # Get model folders (using Civitai Browser's function)
            model_folders = get_model_folders()
            
            # Map model_type to the correct folder key
            folder_key = model_type.lower()
            
            # Map common terms to Civitai Browser's keys
            if folder_key == "checkpoint":
                folder_key = "ckpt"
            elif folder_key == "textualinversion" or folder_key == "ti":
                folder_key = "embedding"
            elif folder_key == "hypernetworks":
                folder_key = "hypernetwork"
                
            if folder_key not in model_folders:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model type: {model_type}. Valid types: {list(model_folders.keys())}"
                )
            
            # Get destination directory
            dest_dir = model_folders[folder_key]
            
            # Get filename and sanitize if needed
            filename = file_to_download.get("name")
            if not filename:
                raise HTTPException(status_code=404, detail="Filename not found in model data")
            
            # Use Civitai Browser's find_available_file_name to avoid conflicts
            full_path = find_available_file_name(os.path.join(dest_dir, filename))
            
            # Check if already downloaded
            if os.path.exists(full_path):
                return {
                    "success": True,
                    "message": "File already exists",
                    "file_path": full_path
                }
            
            # Get download URL
            download_url = file_to_download.get("downloadUrl")
            if not download_url:
                raise HTTPException(status_code=404, detail="Download URL not found")
            
            print(f"Downloading {filename} to {full_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Use Civitai Browser's download function
            success = download_file_with_progressbar(download_url, full_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="Download failed")
            
            return {
                "success": True,
                "message": "File downloaded successfully",
                "file_path": full_path
            }
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error downloading model: {error_details}")
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
