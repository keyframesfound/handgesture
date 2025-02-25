# HandGesture

A Python-based hand gesture recognition system using OpenCV and MediaPipe.  
This project detects various hand gestures (e.g. thumbs up, peace sign, and closed fist) and triggers associated actions like opening Spotlight with voice recognition or controlling screen recording.

## Features

- **Thumbs Up:** Opens Spotlight with voice recognition.
- **Peace Sign:** Toggles the screen recording menu.
- **Closed Fist:** Starts or stops screen recording.

## Prerequisites

- Python 3.x
- OpenCV
- [MediaPipe](https://google.github.io/mediapipe/)
- pyautogui, pyperclip, and other dependencies

See [requirements.txt](requirements.txt) for the complete list.

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/keyframesfound/handgesture.git
    cd handgesture
    ```
2. Create a virtual environment and install dependencies:
    ```sh
    python -m venv venv
    source venv/bin/activate  # (Linux/macOS) or venv\Scripts\activate (Windows)
    pip install -r requirements.txt
    ```

## Usage

Run the main script:
```sh
python [main.py](main.py)
