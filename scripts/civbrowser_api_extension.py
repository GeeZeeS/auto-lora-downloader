import os
import sys
import gradio as gr
from modules import script_callbacks
from modules import shared

# Simple function to check if the Civitai Browser extension is installed
def is_civbrowser_installed():
    extension_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-webui-civbrowser")
    return os.path.exists(extension_path)

# Import the API module (defined in the main script)
def on_app_started(demo: gr.Blocks, app):
    """Called when the Gradio app is started"""
    try:
        from civbrowser_api import on_app_started as register_api
        register_api(demo, app)
    except Exception as e:
        print(f"Error registering Civitai Browser API endpoints: {e}")

# Register our callback
script_callbacks.on_app_started(on_app_started)

# Display warnings if needed
if not shared.cmd_opts.api:
    print("=" * 80)
    print("WARNING: API access is not enabled! To use the Civitai Browser API endpoints,")
    print("you need to start the webui with the '--api' flag:")
    print("    python launch.py --api")
    print("=" * 80)

if not is_civbrowser_installed():
    print("=" * 80)
    print("NOTE: SD-WebUI Civitai Browser extension is not detected.")
    print("This API extension can work independently, but you may want to install")
    print("the full browser extension from: https://github.com/SignalFlagZ/sd-webui-civbrowser")
    print("=" * 80)
