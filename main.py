import cv2
import mediapipe as mp
import pyautogui
import time
from datetime import datetime
import numpy as np
import threading
import queue

# Initialize MediaPipe Hand solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
# Set a lower resolution to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Cooldown for screenshots (3 seconds)
SCREENSHOT_COOLDOWN = 3
last_screenshot_time = 0

# Add new global variables after the last_screenshot_time
is_recording = False
video_writer = None
recording_start_time = 0

# Flag to track thumbs-up gesture state across frames.
thumbs_up_active = False

# New globals for multithreading recording
record_queue = None
record_thread = None
record_thread_running = False

def is_fist_closed(hand_landmarks):
    # Get finger tip and pip landmarks
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_pips = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky pips
    
    # Check if fingers are closed
    closed_fingers = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            closed_fingers += 1
    
    # If all fingers are closed (below their pips), it's a fist
    return closed_fingers >= 3

def is_thumbs_up(hand_landmarks):
    # Thumb tip and IP (Inter Phalangeal Joint)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    
    # Other finger landmarks
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_pips = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky pips
    
    # Check if thumb is pointing up
    thumb_up = thumb_tip.y < thumb_ip.y
    
    # Check if other fingers are closed
    other_fingers_closed = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )
    
    return thumb_up and other_fingers_closed

def take_screenshot():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'screenshot_{timestamp}.png'
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"Screenshot saved as {filename}")

def record_video_worker():
    global record_thread_running, video_writer, record_queue
    while record_thread_running or (record_queue and not record_queue.empty()):
        try:
            frame = record_queue.get(timeout=0.1)
            if video_writer is not None:
                video_writer.write(frame)
        except queue.Empty:
            continue

def start_recording():
    global video_writer, is_recording, recording_start_time, record_queue, record_thread, record_thread_running
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'recording_{timestamp}.mp4'
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    video_writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        frame_size
    )
    is_recording = True
    recording_start_time = time.time()
    # Initialize the queue and start the recorder thread
    record_queue = queue.Queue()
    record_thread_running = True
    record_thread = threading.Thread(target=record_video_worker)
    record_thread.daemon = True
    record_thread.start()
    print(f"Started recording: {filename}")

def stop_recording():
    global video_writer, is_recording, record_thread_running, record_thread
    record_thread_running = False
    if record_thread is not None:
        record_thread.join()
    if video_writer is not None:
        video_writer.release()
    is_recording = False
    print("Recording stopped")

def main():
    global last_screenshot_time, is_recording, thumbs_up_active, record_queue
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image once for selfie-view
        flipped_image = cv2.flip(image, 1)
        # Use the flipped image for both display and landmark processing
        image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Reset thumbs_up_active if no hand is detected.
        current_thumbs = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    flipped_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Check for fist gesture (screenshot)
                if is_fist_closed(hand_landmarks):
                    current_time = time.time()
                    if current_time - last_screenshot_time >= SCREENSHOT_COOLDOWN:
                        take_screenshot()
                        last_screenshot_time = current_time
                
                # Use edge detection for thumbs-up to toggle recording.
                if is_thumbs_up(hand_landmarks):
                    current_thumbs = True
                    if not thumbs_up_active:
                        if not is_recording:
                            start_recording()
                        else:
                            stop_recording()
        
        # Update state at end of frame.
        thumbs_up_active = current_thumbs

        # Recording: write the already flipped image without re-flipping
        if is_recording:
            cv2.circle(flipped_image, (30, 30), 10, (0, 0, 255), -1)
            # Enqueue frame for recording; avoid adding if queue is too large
            if record_queue is not None and record_queue.qsize() < 10:
                record_queue.put(flipped_image.copy())
        
        # Display the image
        cv2.imshow('Hand Gesture Recognition', flipped_image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if is_recording:
                stop_recording()
            break

    # Clean up
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()