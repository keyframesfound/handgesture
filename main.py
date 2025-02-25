import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from datetime import datetime
import os
import subprocess
import time
import speech_recognition as sr
import threading
import pyperclip
from pynput.keyboard import Key, Controller

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize speech recognizer
recognizer = sr.Recognizer()
keyboard = Controller()

# Start video capture
cap = cv2.VideoCapture(0)

# Variables to track the last action time to prevent multiple triggers
last_action_time = 0
ACTION_COOLDOWN = 2  # Cooldown in seconds

# Variable to track recording state
is_recording = False
is_listening = False

def listen_and_type():
    """Listen for speech and type it into Spotlight"""
    global is_listening
    
    try:
        with sr.Microphone() as source:
            print("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                
                # Copy text to clipboard
                pyperclip.copy(text)
                
                # Paste text (Command + V)
                time.sleep(0.5)  # Wait for Spotlight to be ready
                pyautogui.hotkey('command', 'v')
                
                # Press return to execute
                time.sleep(0.2)
                pyautogui.press('return')
                
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
    
    except Exception as e:
        print(f"Error in speech recognition: {e}")
    
    finally:
        is_listening = False

def calculate_thumb_up(hand_landmarks):
    """Check if gesture is thumbs up"""
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    other_fingers_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    other_fingers_bases = [6, 10, 14, 18]  # Corresponding bases
    
    # Check if thumb is pointing up
    thumb_is_up = thumb_tip.y < thumb_base.y
    
    # Check if other fingers are closed
    other_fingers_closed = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y
        for tip, base in zip(other_fingers_tips, other_fingers_bases)
    )
    
    return thumb_is_up and other_fingers_closed

def calculate_peace_sign(hand_landmarks):
    """Check if gesture is peace sign (V sign)"""
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Check if index and middle fingers are up
    index_up = index_tip.y < hand_landmarks.landmark[6].y
    middle_up = middle_tip.y < hand_landmarks.landmark[10].y
    
    # Check if other fingers are down
    ring_down = ring_tip.y > hand_landmarks.landmark[14].y
    pinky_down = pinky_tip.y > hand_landmarks.landmark[18].y
    
    return index_up and middle_up and ring_down and pinky_down

def calculate_closed_fist(hand_landmarks):
    """Check if all fingers are closed (fist)"""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_bases = [6, 10, 14, 18]  # Corresponding bases
    thumb_tip = 4
    thumb_base = 2
    
    # Check if all fingers are closed
    fingers_closed = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y
        for tip, base in zip(finger_tips, finger_bases)
    )
    
    # Check thumb position
    thumb_closed = hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_base].x
    
    return fingers_closed and thumb_closed

def open_spotlight_with_voice():
    """Open Spotlight and start voice recognition"""
    global is_listening
    
    if not is_listening:
        # Open Spotlight
        pyautogui.hotkey('command', 'space')
        print("Opened Spotlight with voice recognition")
        
        # Start voice recognition in a separate thread
        is_listening = True
        threading.Thread(target=listen_and_type).start()

def toggle_screen_recording():
    """Simulate Command + Shift + 5 to toggle screen recording"""
    pyautogui.hotkey('command', 'shift', '5')
    print("Toggled screen recording menu")

def start_stop_recording():
    """Start/Stop screen recording using Command + Shift + 5"""
    global is_recording
    if not is_recording:
        # Start recording
        pyautogui.hotkey('command', 'shift', '5')
        time.sleep(1)  # Wait for menu to appear
        pyautogui.press('space')  # Press space to start recording
        is_recording = True
        print("Started recording")
    else:
        # Stop recording
        pyautogui.hotkey('command', 'control', 'esc')  # Stop recording
        is_recording = False
        print("Stopped recording")

def main():
    global last_action_time
    
    print("Gesture Controls:")
    print("- Thumbs Up: Open Spotlight with Voice Recognition")
    print("- Peace Sign: Toggle Recording Menu")
    print("- Closed Fist: Start/Stop Recording")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                current_time = time.time()
                if current_time - last_action_time >= ACTION_COOLDOWN:
                    # Check for thumbs up
                    if calculate_thumb_up(hand_landmarks):
                        open_spotlight_with_voice()
                        last_action_time = current_time
                    
                    # Check for peace sign
                    elif calculate_peace_sign(hand_landmarks):
                        toggle_screen_recording()
                        last_action_time = current_time
                    
                    # Check for closed fist
                    elif calculate_closed_fist(hand_landmarks):
                        start_stop_recording()
                        last_action_time = current_time

        # Display the image
        cv2.imshow('Hand Gesture Recognition', image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()