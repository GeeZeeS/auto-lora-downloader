import os
import json
import gradio as gr
from modules import script_callbacks
from modules import shared
from fastapi import FastAPI

# Import our API module
from civbrowser_api import register_civbrowser_api

def on_app_started(demo: gr.Blocks, app: FastAPI):
    """Called when the Gradio app is started"""
    try:
        # Register our API routes
        register_civbrowser_api(app)
        
        # Add API info to UI
        print("Civitai Browser API endpoints registered successfully")
        print("API documentation available at: /docs")
    except Exception as e:
        print(f"Error registering Civitai Browser API endpoints: {e}")

# Register our callback
script_callbacks.on_app_started(on_app_started)

# Check if the main API is enabled
if not shared.cmd_opts.api:
    print("=" * 80)
    print("WARNING: API access is not enabled! To use the Civitai Browser API endpoints,")
    print("you need to start the webui with the '--api' flag:")
    print("    python launch.py --api")
    print("=" * 80)
