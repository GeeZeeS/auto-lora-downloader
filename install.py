import launch
import os
import sys

# Check if requirements are met
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if lib and not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"sd-webui-civbrowser-api requirement: {lib}")

# Add this extension to Python path
extension_path = os.path.dirname(os.path.realpath(__file__))
if extension_path not in sys.path:
    sys.path.append(extension_path)

# Check if original Civitai Browser extension is installed
original_ext_path = os.path.join(os.path.dirname(extension_path), "sd-webui-civbrowser")
if not os.path.exists(original_ext_path):
    print("=" * 80)
    print("WARNING: SD-WebUI Civitai Browser extension (sd-webui-civbrowser) not found!")
    print("This extension adds API endpoints to the original extension.")
    print("Please make sure you have installed the original extension from:")
    print("https://github.com/SignalFlagZ/sd-webui-civbrowser")
    print("=" * 80)
