from modules import script_callbacks, shared
from modules.api.api import Api

from auto_lora_api import setup_api, on_ui_tabs, process_before_generation

def on_app_started(demo, app):
    setup_api(Api(app, demo))

script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_before_image_generated(process_before_generation)
