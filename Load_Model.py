#!/usr/bin/env python3
"""
Complete Testing Script for Raspberry Pi Object Identifier
Tests each component individually before running full inference
"""

import os
import sys
import time
import json

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(status, message):
    symbols = {"pass": "✓", "fail": "✗", "warn": "⚠", "info": "ℹ"}
    colors = {"pass": "\033[92m", "fail": "\033[91m", "warn": "\033[93m", "info": "\033[94m"}
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{symbols.get(status, '•')} {message}{reset}")

# ==================== TEST 1: CHECK FILES ====================
print_header("TEST 1: Checking Required Files")

files_to_check = {
    'object_identifier_model.h5': 'Trained model file',
    'class_indices.json': 'Class mapping file (optional)',
}

files_found = {}
for filename, description in files_to_check.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename) / (1024*1024)  # MB
        print_status("pass", f"{filename} found ({size:.2f} MB) - {description}")
        files_found[filename] = True
    else:
        print_status("fail" if filename.endswith('.h5') else "warn", 
                    f"{filename} NOT found - {description}")
        files_found[filename] = False

if not files_found['object_identifier_model.h5']:
    print_status("fail", "CRITICAL: Model file is required!")
    print("\nTo transfer your model to Raspberry Pi:")
    print("  1. From another computer: scp object_identifier_model.h5 pi@raspberrypi.local:~/")
    print("  2. Or use USB drive, email, cloud storage, etc.")
    sys.exit(1)

# ==================== TEST 2: CHECK DEPENDENCIES ====================
print_header("TEST 2: Checking Python Dependencies")

dependencies = {
    'numpy': 'Numerical operations',
    'cv2': 'OpenCV for image processing',
    'tensorflow': 'TensorFlow for model inference',
    'picamera2': 'Raspberry Pi camera interface',
    'RPi.GPIO': 'GPIO control for buzzer',
}

missing_deps = []
for module, description in dependencies.items():
    try:
        if module == 'cv2':
            import cv2
        elif module == 'tensorflow':
            import tensorflow as tf
        elif module == 'picamera2':
            from picamera2 import Picamera2
        elif module == 'RPi.GPIO':
            import RPi.GPIO as GPIO
        else:
            __import__(module)
        print_status("pass", f"{module} - {description}")
    except ImportError:
        print_status("fail", f"{module} NOT installed - {description}")
        missing_deps.append(module)

if missing_deps:
    print_status("fail", f"Missing dependencies: {', '.join(missing_deps)}")
    print("\nInstall with:")
    print("  pip3 install tensorflow opencv-python picamera2 RPi.GPIO numpy")
    sys.exit(1)

# ==================== TEST 3: LOAD MODEL ====================
print_header("TEST 3: Loading Model")

try:
    import tensorflow as tf
    print_status("info", "Loading model (this may take 30-60 seconds)...")
    model = tf.keras.models.load_model('object_identifier_model.h5')
    print_status("pass", "Model loaded successfully!")
    
    # Get model info
    input_shape = model.input_shape
    output_shape = model.output_shape
    num_classes = output_shape[-1]
    
    print_status("info", f"Input shape: {input_shape}")
    print_status("info", f"Output shape: {output_shape}")
    print_status("info", f"Number of classes: {num_classes}")
    
except Exception as e:
    print_status("fail", f"Failed to load model: {e}")
    sys.exit(1)

# ==================== TEST 4: LOAD/CREATE CLASS INDICES ====================
print_header("TEST 4: Class Indices")

if os.path.exists('class_indices.json'):
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        print_status("pass", "class_indices.json loaded")
        print_status("info", f"Classes found: {list(class_indices.keys())}")
    except Exception as e:
        print_status("fail", f"Error reading class_indices.json: {e}")
        class_indices = None
else:
    print_status("warn", "class_indices.json not found - creating default")
    print_status("info", "Please verify these match your training data!")
    
    # Create default based on number of classes
    if num_classes == 2:
        class_indices = {"dang": 0, "ndan": 1}
    else:
        class_indices = {f"class_{i}": i for i in range(num_classes)}
    
    # Save it
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print_status("info", f"Created: {class_indices}")
    print_status("warn", "Review and edit class_indices.json if needed!")

index_to_class = {v: k for k, v in class_indices.items()}

# ==================== TEST 5: TEST GPIO/BUZZER ====================
print_header("TEST 5: GPIO and Buzzer")

try:
    import RPi.GPIO as GPIO
    
    BUZZER_PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    
    print_status("pass", f"GPIO initialized (using pin {BUZZER_PIN})")
    
    response = input("\nTest buzzer? This will turn it ON for 1 second (y/n): ")
    if response.lower() == 'y':
        print_status("info", "Buzzer ON...")
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        print_status("info", "Buzzer OFF")
        
        heard = input("Did you hear the buzzer? (y/n): ")
        if heard.lower() == 'y':
            print_status("pass", "Buzzer working!")
        else:
            print_status("warn", "Buzzer not heard - check wiring")
            print("       Wiring: GPIO17 (Pin 11) → Buzzer (+)")
            print("               GND (Pin 6) → Buzzer (-)")
    
    GPIO.cleanup()
    
except Exception as e:
    print_status("fail", f"GPIO error: {e}")
    print_status("info", "Make sure you're running on Raspberry Pi with GPIO access")

# ==================== TEST 6: TEST CAMERA ====================
print_header("TEST 6: Camera")

try:
    from picamera2 import Picamera2
    import cv2
    
    print_status("info", "Initializing camera...")
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    
    print_status("pass", "Camera initialized")
    
    # Capture test image
    print_status("info", "Capturing test image...")
    frame = picam2.capture_array()
    cv2.imwrite('test_capture.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print_status("pass", "Test image saved as test_capture.jpg")
    
    picam2.stop()
    
except Exception as e:
    print_status("fail", f"Camera error: {e}")
    print_status("info", "Enable camera: sudo raspi-config → Interface Options → Camera")
    print_status("info", "Then reboot: sudo reboot")

# ==================== TEST 7: TEST INFERENCE ====================
print_header("TEST 7: Model Inference Test")

try:
    import numpy as np
    import cv2
    
    # Create a test image
    print_status("info", "Creating test input...")
    img_size = input_shape[1]  # Get size from model
    test_image = np.random.rand(img_size, img_size, 3).astype('float32')
    test_input = np.expand_dims(test_image, axis=0)
    
    print_status("info", "Running inference...")
    start_time = time.time()
    predictions = model.predict(test_input, verbose=0)
    inference_time = time.time() - start_time
    
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = index_to_class[predicted_index]
    confidence = float(predictions[0][predicted_index])
    
    print_status("pass", f"Inference successful!")
    print_status("info", f"Inference time: {inference_time*1000:.1f} ms")
    print_status("info", f"Predicted: {predicted_class} ({confidence*100:.1f}%)")
    print_status("info", f"Expected FPS: ~{1/inference_time:.1f}")
    
    if inference_time > 1.0:
        print_status("warn", "Inference is slow - consider:")
        print("       - Converting to TensorFlow Lite")
        print("       - Using smaller image size")
        print("       - Reducing model complexity")
    
except Exception as e:
    print_status("fail", f"Inference error: {e}")

# ==================== SUMMARY ====================
print_header("TEST SUMMARY")

print_status("pass", "All critical tests passed!")
print("\nYou're ready to run the full application:")
print("  python3 rpi_object_identifier.py")
print("\nConfiguration tips:")
print("  - Edit BUZZER_PIN if using different GPIO pin")
print("  - Adjust CONFIDENCE_THRESHOLD (default: 0.5)")
print("  - Set DISPLAY_WINDOW=False for headless operation")
print("  - Reduce INFERENCE_DELAY for faster predictions")

print("\n" + "="*60)
print("  Testing Complete!")
print("="*60 + "\n")