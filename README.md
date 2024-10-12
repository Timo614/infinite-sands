# infinite-sands
Python project utilizing a sandbox for displaying standard diffusion depth mapped art

See the Hackster article for more information: https://www.hackster.io/sand-command/infinite-sands-df675a

Utilizes the OpenCV FullScreen logic from https://github.com/elerac/fullscreen

## Setup

Prerequisite: The code here has been updated to use ROCM 6.2 while the initial Hackster project used 6.1. Utilize 6.2 when installing this version for maximum compatibility otherwise scripts and logic will need to be updated to reference the earlier ROCM version.

Download the repository. You may need to modify the paths of files if you have a setup that is different from my own. 

`infinite-sands.sh` - I placed this file in my home directory to start the stable diffusion web UI as a background task in my terminal. `chmod +x infinite-sands.sh` prepares the file for execution.

`infinite-sands.py`, `infinite-sands-api.json`, and `cv_fullscreen.py` should be placed alongside each other.

Updates to ready the machine for the device:
- Set Screen Sleep to Never from Settings -> Power
- Disable popup for unresponsive script (may not trigger but may cause overlay above screen if it does)

### Disable Popup
Install dconf-editor:
`sudo apt install dconf-editor`

Run: `dconf-editor`

Go to section: `/org/gnome/mutter/`
Modify key `check-alive-timeout`
Override default with a large number

### Note About ControlNet

After installing the ControlNet extension to use this project's existing json without modifications you will need to download the model diff_control_sd15_depth_fp16 which can be found [here](https://huggingface.co/kohya-ss/ControlNet-diff-modules/blob/main/diff_control_sd15_depth_fp16.safetensors). Alternatively a separate depth ControlNet model can be used but it will require an update to the json.

### Install requirements
Installed requirements (there may be others depending upon your environment but these were needed for me):
```sh
sudo apt-get install libgtk2.0-dev
sudo apt-get install portaudio19-dev
pip install screeninfo
sudo apt install screen
sudo apt install git

python3 -m venv venv
source venv/bin/activate

pip install openai-whisper sounddevice numpy
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2/
pip install transformers
pip install mediapipe opencv-python 
```

When you want to run the logic from the root (assuming you have stable diffusion web UI there for the shell script):
```sh
nohup ./infinite-sands.sh &
screen -S display_session
source venv/bin/activate
export DISPLAY=:0
python3 infinite-sands.py
```

I start the script from terminal so I use screen for the display session. The `DISPLAY` variable is set to 0 here for my projector.

## Test Files

The `test-files` directory contains several test related pieces of code used during development. These can be ignored for a deployment but may help to debug issues if you're experiencing problems with your setup.

`board-test.py` - A file used to create a CharucoBoard utilized for viewport mapping. This logic is utilized in the infinite-sands.py script file as part of the calibration but the `sands-improved.ipynb` file provides a jupyter notebook to step through the calibration process so it's useful to leave this board on, run a capture from the notebook, and walk through the calibration.

`blank-test.py` - Once the calibration was completed I needed a way to make the sand well lit so we could take a photo for use with [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) utilized by the [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet). The aforementioned jupyter notebook utilized this to take a capture of the state of the board, we warped it using the earlier calibration, and tested it with the web UI to test our setup.

`sands.ipynb` - An initial jupyter notebook testing utilizing colors for determining the outline of the sandbox. Experimented with photos taken directly from the hung webcam. Time of day and general lighting conditions affected this approach heavily.

`sands-improved.ipynb` - A jupyter notebook utilized for additional experimentation. The color based approach was dropped here in favor of using a projected CharucoBoard layout and mapping back to those values.

# Licenses for Used Elements

- Mediapipe: [Apache License](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- Whisper: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)
- Stable Diffusion 1.5:  [CreativeML Open RAIL-M License](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE)
- DepthAnythingV2 Large: [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md)
- ControlNet: [Apache License](https://github.com/lllyasviel/ControlNet/blob/main/LICENSE)