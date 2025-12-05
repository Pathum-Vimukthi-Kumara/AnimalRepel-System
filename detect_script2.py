import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import json
import threading
import subprocess
import time
import os
import csv
from datetime import datetime
import platform

# Try to import GPIO for hardware buzzer
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    BUZZER_PIN = 17  # GPIO pin 17 (physical pin 11) - change if needed
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    print(f"✓ GPIO initialized - Buzzer on GPIO{BUZZER_PIN}")
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    print("⚠ GPIO not available - using audio fallback")

# Load YOLO model (downloads yolov8n.pt on first run if not present)
print("Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')

# Load TFLite model
tflite_path = 'object_identifier_model.tflite'
if not os.path.exists(tflite_path):
    print(f"Error: {tflite_path} not found.")
    exit()
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # e.g., [1, height, width, 3]

# Load class indices from JSON and invert for lookup
json_path = 'class_indices.json'
if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit()
with open(json_path, 'r') as f:
    class_indices = json.load(f)
classes = {int(v): k for k, v in class_indices.items()}  # Invert {str: int} to {int: str}

# Dangerous classes for buzzer (only those with '_dang' suffix)
dangerous_classes = set()
for cls_id, name in classes.items():
    if name.endswith('_dang'):
        dangerous_classes.add(cls_id)
print(f"Loaded {len(classes)} classes, {len(dangerous_classes)} marked as dangerous")

# Potential animal classes from YOLO (COCO names to check for TFLite override)
potential_animals = {'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# Confidence thresholds
yolo_conf_threshold = 0.4
tflite_conf_threshold = 0.5

# Buzzer cooldown (seconds)
last_buzzer_time = 0
cooldown = 1

# CSV file for logging _dang detections
csv_path = 'detections.csv'
# Create header if file doesn't exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Animal', 'Timestamp'])

# Hardware buzzer function
def play_hardware_buzzer(duration=0.5, beeps=3):
    """Play buzzer using GPIO pin"""
    if not GPIO_AVAILABLE:
        return
    
    try:
        for _ in range(beeps):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.1)
    except Exception as e:
        print(f"Buzzer error: {e}")

# Audio fallback function
def play_audio_beep():
    """Play beep using audio - fallback when GPIO not available"""
    system = platform.system()
    
    if system == 'Windows':
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except ImportError:
            print("\a")
    else:
        # Linux/Raspberry Pi
        try:
            # Try using sox (play command)
            subprocess.Popen(['play', '-n', 'synth', '0.2', 'sine', '1000'], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            try:
                # Try playing buzzer.wav if it exists
                if os.path.exists('buzzer.wav'):
                    subprocess.Popen(['aplay', 'buzzer.wav'], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                else:
                    # Fallback to beep command
                    subprocess.Popen(['beep', '-f', '1000', '-l', '200'], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                # Last resort - terminal bell
                print("\a")

# Combined beep function
def play_beep():
    """Play beep - tries hardware buzzer first, then audio fallback"""
    if GPIO_AVAILABLE:
        play_hardware_buzzer()
    else:
        play_audio_beep()

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Troubleshooting tips:")
    print("1. Check if camera is connected: ls /dev/video*")
    print("2. For Pi Camera: sudo raspi-config -> Interface Options -> Legacy Camera")
    print("3. Try different camera index: VideoCapture(1) or VideoCapture(2)")
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    exit()

# Optimize for Raspberry Pi - reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
print("Starting detection... Press 'q' to quit")
print("-" * 50)

frame_count = 0
detection_count = 0

# FPS calculation variables
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS every second
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Detect with YOLO
        yolo_results = yolo_model(frame, verbose=False)[0]  # verbose=False reduces console spam
        annotated_frame = frame.copy()

        detected_dangerous = False

        for box, cls, conf in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
            if conf < yolo_conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box)
            yolo_class_id = int(cls)
            yolo_label = yolo_results.names[yolo_class_id]
            override_label = None
            override_conf = conf

            # Relabel specific common classes
            if yolo_label == 'person':
                yolo_label = 'person1'
            elif yolo_label == 'cell phone':
                yolo_label = 'phone'

            # If potential animal, crop and classify with TFLite
            if yolo_label in potential_animals:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Preprocess for TFLite (resize, normalize)
                resized = cv2.resize(crop, (input_shape[2], input_shape[1]))
                input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                pred_class = np.argmax(output)
                pred_conf = output[pred_class]
                if pred_conf > tflite_conf_threshold:
                    custom_label = classes.get(pred_class, 'Unknown')
                    override_label = custom_label
                    override_conf = pred_conf
                    if pred_class in dangerous_classes and custom_label.endswith('_dang'):
                        detected_dangerous = True
                        detection_count += 1
                        # Log to CSV
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([custom_label, timestamp])
                        print(f"🚨 ALERT #{detection_count}: {custom_label} detected at {timestamp}")

            # Final label and color
            final_label = override_label if override_label else yolo_label
            color = (0, 255, 0) if override_label else (255, 0, 0)  # Green for custom, blue for YOLO
            full_label = f"{final_label} {override_conf:.2f}"

            # Draw box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add status info to frame
        status_text = f"FPS: {current_fps:.1f} | Frame: {frame_count} | Alerts: {detection_count}"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add buzzer status
        buzzer_status = f"Buzzer: {'GPIO' if GPIO_AVAILABLE else 'Audio'}"
        cv2.putText(annotated_frame, buzzer_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow('Live Detection (YOLO + TFLite)', annotated_frame)

        # Play beep if dangerous animal detected (with cooldown)
        current_time = time.time()
        if detected_dangerous and (current_time - last_buzzer_time > cooldown):
            threading.Thread(target=play_beep, daemon=True).start()
            last_buzzer_time = current_time

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⚠ Interrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if GPIO_AVAILABLE:
        GPIO.cleanup()
        print("GPIO cleaned up")
    
    print("-" * 50)
    print(f"Session Summary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Dangerous animals detected: {detection_count}")
    print(f"  Detections saved to: {csv_path}")
    print("Detection stopped.")