import whisper
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import string
import cv2
import cv2_fullscreen
import mediapipe as mp
import json
import base64
import requests
from datetime import datetime, timedelta
import argparse 

# Initialize Whisper model
model = whisper.load_model("base")

audio_queue = queue.Queue()
prompt_queue = queue.Queue()
render_queue = queue.Queue()

# Wake word
wake_word = "hello"

# Initial prompt
prompt = "mountains"
lora_enabled = False

# Used for Whisper
samplerate = 16000
channels = 1
blocksize = int(samplerate * 1)
silence_threshold = 0.1
silence_duration = 0.2

# Event to kill threads
stop_event = threading.Event()

# Variables to store calibration
H = None
height = None
width = None
board_projection = None


def check_for_hands(image, mp_hands) -> bool:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)

    if results.multi_hand_landmarks:
        print("Hands found")
        return True
    else:
        return False


def audio_callback(indata, frames, time, status):
    """Callback function to capture audio and put it in the queue."""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())


def process_audio_and_detect_wake_word(stop_event):
    """Detect the wake word and process the audio after detecting it."""
    print("Listening for the wake word and processing audio...")
    silence_start_time = None
    captured_audio = []

    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        result = model.transcribe(audio_data, language='en', condition_on_previous_text=False, no_speech_threshold=0.8,
                                  prepend_punctuations="", append_punctuations="")
        text = result['text'].strip().lower()

        # Check for wake word
        if wake_word in text:
            print(f"Wake word '{wake_word}' detected!")
            captured_audio = []
            silence_start_time = None

            # Capture subsequent audio until silence
            while not stop_event.is_set():
                try:
                    audio_chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)

                # Check if audio is silent
                if np.max(np.abs(audio_data)) >= silence_threshold:
                    captured_audio.append(audio_data)
                    silence_start_time = None
                else:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= silence_duration:
                        if captured_audio:
                            captured_audio = np.concatenate(captured_audio)
                            prompt_message = send_prompt(captured_audio)
                            prompt_queue.put(prompt_message)
                        break


def send_prompt(captured_audio, language='en', condition_on_previous_text=False, no_speech_threshold=0.8,
                prepend_punctuations="", append_punctuations=""):
    """Send the prompt or perform an action."""
    result = model.transcribe(captured_audio, language=language, condition_on_previous_text=condition_on_previous_text,
                              no_speech_threshold=no_speech_threshold, prepend_punctuations=prepend_punctuations,
                              append_punctuations=append_punctuations)
    message = result['text'].strip().lower()

    # Remove punctuation
    message = message.translate(str.maketrans('', '', string.punctuation))

    # Filter out the words "thank" and "you" that are often hallucinated and not needed here
    filtered_words = [word for word in message.split() if word not in ["thank", "you"]]
    filtered_message = " ".join(filtered_words)

    print(f"Prompt: {filtered_message}")
    return filtered_message


def calibrate():
    """Display the calibration screen and perform calibration."""
    global H, height, width, board_projection

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((12, 8), 0.015, 0.012, aruco_dict)
    board_image = board.generateImage((1920, 1080), marginSize=0)
    board_projection = board_image

    render_queue.put(board_image)

    # Allow some time for the board to be displayed before calibration
    print("Displaying calibration board. Waiting for 5 seconds...")
    time.sleep(5)

    # Perform calibration
    print("Starting calibration process...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    detection_params = cv2.aruco.DetectorParameters()
    detection_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

    image_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charucodetector = cv2.aruco.CharucoDetector(board, detectorParams=detection_params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(image_grayscale)

    if charuco_corners is None or len(charuco_corners) == 0:
        print("No Charuco corners detected. Calibration failed.")
        return False

    base_charuco_corners, base_charuco_ids, base_marker_corners, base_marker_ids = charucodetector.detectBoard(
        board_image)

    if base_charuco_corners is None or len(base_charuco_corners) == 0:
        print("No base Charuco corners detected. Calibration failed.")
        return False

    # Filter to set of charuco recognized
    if base_charuco_ids is not None and charuco_ids is not None:
        filtered_corners = []
        filtered_ids = []
        original_ids_set = set(charuco_ids.flatten())

        for i, base_id in enumerate(base_charuco_ids.flatten()):
            if base_id in original_ids_set:
                filtered_corners.append(base_charuco_corners[i])
                filtered_ids.append(base_id)

        filtered_corners = np.array(filtered_corners)
        filtered_ids = np.array(filtered_ids)

    src_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    dst_pts = filtered_corners.reshape(-1, 2).astype(np.float32)

    # Find the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography calculation failed.")
        return False
    height, width = board_image.shape[:2]
    print("Calibration complete. Homography matrix: ", H)
    return True


def render_queued_image():
    """Render the queued image."""

    screen = cv2_fullscreen.FullScreen()

    while not stop_event.is_set():
        try:
            image = render_queue.get(timeout=0.1)
            screen.imshow(image)
        except queue.Empty:
            continue

    screen.destroyWindow()


def process_and_render():
    """Process the captured frame and send it to render queue."""
    global should_render
    global prompt
    while not stop_event.is_set():
        if should_render:
            print(f"Enter command (/render, /art, /normal, or a new prompt): Rendering: {prompt}")
            # Set display to white screen
            white_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
            render_queue.put(white_image)
            time.sleep(1)

            print("Rendering: Capturing frame from webcam...")
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                print("Rendering: Failed to capture frame from webcam.")
                should_render = False
                continue
            cap.release()

            warped_image = cv2.warpPerspective(frame, H, (width, height))
            retval, buffer = cv2.imencode('.jpg', warped_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            with open('infinite-sands-api.json') as infile:
                data = json.load(infile)

            prompt_data = prompt
            # Optionally enable art LoRA
            if lora_enabled:
                prompt_data += " <lora:kerin-lovett:1>"
            data['prompt'] = prompt_data
            # ControlNet depth image
            data['alwayson_scripts']['ControlNet']['args'][0]['image']['image'] = base64_image

            api_url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
            response = requests.post(api_url, json=data)

            if response.status_code == 200:
                print('Request successful.')
                rendered_image = response.json()['images'][0]

                image_data = base64.b64decode(rendered_image)
                np_arr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                scaling_factor = 3.0
                image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
                render_queue.put(image)
            else:
                print('Request failed with status code:', response.status_code)
                print('Response:', response.text)

            should_render = False

        time.sleep(1)


def run_loop(mp_hands, mode):
    """Unified loop for both CLI and Voice mode, with hand detection and rerendering."""
    global should_render

    if mode == 'cli':
        # Run CLI mode where we listen for user input and handle commands, while also detecting hands
        while not stop_event.is_set():
            try:
                if not prompt_queue.empty():
                    # Process any queued prompts from CLI input or other sources
                    temp_prompt = prompt_queue.get()
                    handle_prompt(temp_prompt)

                # Handle hand detection for rerendering
                handle_hand_detection(mp_hands)

                # Non-blocking CLI input
                user_input = input("Enter command (/render, /art, /normal, or a new prompt): ").strip()
                if user_input:
                    prompt_queue.put(user_input)  # Queue the user input to be processed

            except (KeyboardInterrupt, EOFError):
                print("Exiting CLI mode...")
                stop_event.set()  # Stop the event to safely exit all threads
                break

        time.sleep(0.1)

    else:
        # In voice mode, we detect hand gestures for rerendering
        while not stop_event.is_set():
            if not prompt_queue.empty():
                temp_prompt = prompt_queue.get()
                handle_prompt(temp_prompt)
            else:
                handle_hand_detection(mp_hands)

            time.sleep(1)


def handle_hand_detection(mp_hands):
    """Detect hands and trigger rerendering if hands are detected or leave, without printing any messages."""
    global last_hands_seen
    global render_triggered
    global should_render

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return

    hand_frame = cv2.warpPerspective(frame, H, (width, height))
    hands_visible = check_for_hands(hand_frame, mp_hands)

    # If hands are detected, update `last_hands_seen` and reset rendering trigger
    if hands_visible:
        last_hands_seen = datetime.now()
        render_triggered = False
    else:
        # If it's been more than 2 seconds since hands were last seen and rerender hasn't happened yet
        if last_hands_seen and (datetime.now() - last_hands_seen) > timedelta(seconds=2) and not render_triggered:
            should_render = True
            render_triggered = True  # Prevent further rerender until hands reappear
        else:
            should_render = False  # No rerender if hands haven't been gone for 2 seconds yet


def main():
    global should_render
    global prompt
    global lora_enabled
    global last_hands_seen
    global render_triggered

    last_hands_seen = None
    render_triggered = False

    # Add command-line argument parsing for mode selection
    parser = argparse.ArgumentParser(description="Choose between CLI and Voice Command Mode")
    parser.add_argument('--mode', type=str, choices=['cli', 'voice'], default='voice',
                        help="Choose CLI or Voice Command mode.")
    args = parser.parse_args()

    mp_hands = mp.solutions.hands.Hands()
    rendering_thread = threading.Thread(target=render_queued_image)
    rendering_thread.start()

    if calibrate():
        print("Calibration successful.")
        should_render = True
    else:
        print("Calibration failed. Exiting...")
        return

    process_thread = threading.Thread(target=process_and_render)
    process_thread.start()

    stream = None
    if args.mode == 'voice':
        stream = sd.InputStream(callback=audio_callback, channels=channels, samplerate=samplerate, blocksize=blocksize)

    try:
        if stream:
            with stream:
                print("Listening for the wake word...")

                processing_thread = threading.Thread(target=process_audio_and_detect_wake_word, args=(stop_event,))
                processing_thread.daemon = True
                processing_thread.start()

                run_loop(mp_hands, mode=args.mode)

        elif args.mode == 'cli':
            run_loop(mp_hands, mode=args.mode)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
        stop_event.set()  # Trigger the stop event to cleanly exit all threads
    except Exception as e:
        print(f"An error occurred: {e}")

    if stream:
        stream.close()


def handle_prompt(temp_prompt):
    global should_render
    global lora_enabled
    global prompt

    if temp_prompt == "/render":
        should_render = True
    elif temp_prompt == "/art":
        lora_enabled = True
        should_render = True
    elif temp_prompt == "/normal":
        lora_enabled = False
        should_render = True
    else:
        prompt = temp_prompt
        should_render = True


if __name__ == "__main__":
    main()
