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

# Load YOLO model (downloads yolov8n.pt on first run if not present)
yolo_model = YOLO('yolov8n.pt')

# Load TFLite model
tflite_path = 'object_identifier_model.tflite'
if not os.path.exists(tflite_path):
    print(f"Error: {tflite_path} not found.")
    exit()
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

# Function to play beep on Raspberry Pi/Linux
def play_beep():
    """Play beep using system command - works on Raspberry Pi"""
    try:
        # Try using sox (play command)
        subprocess.Popen(['play', '-n', 'synth', '0.2', 'sine', '1000'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        try:
            # Fallback to system beep
            subprocess.Popen(['beep', '-f', '1000', '-l', '200'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            # If neither works, print to console
            print("\a")  # Terminal bell

# Open webcam (use 0 for USB camera, or adjust for Pi Camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("For Raspberry Pi Camera, you may need to enable legacy camera support")
    exit()

# Optimize for Raspberry Pi - reduce resolution if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting detection... Press 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    
    # Detect with YOLO
    yolo_results = yolo_model(frame)[0]
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
                    # Log to CSV
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([custom_label, timestamp])
                    print(f"⚠️  DANGEROUS: {custom_label} detected at {timestamp}")

        # Final label and color
        final_label = override_label if override_label else yolo_label
        color = (0, 255, 0) if override_label else (255, 0, 0)  # Green for custom, blue for YOLO
        full_label = f"{final_label} {override_conf:.2f}"

        # Draw box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"\nProcessed {frame_count} frames. Detections saved to {csv_path}")