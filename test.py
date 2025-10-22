#!/usr/bin/env python3
"""
Raspberry Pi Camera Data Collection Script
Captures images for object detection training with buzzer flagging
"""

import os
import shutil
import time
from datetime import datetime
from picamera2 import Picamera2
import numpy as np
from PIL import Image

# Configuration
DATA_DIR = "./training_data"
buzzer_class = None  # Global variable to store flagged class

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)

def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

def take_photos_continuous(cls_name, flag, duration=5, interval=0.5):
    """
    Capture photos continuously for specified duration
    
    Args:
        cls_name: Class name for the images
        flag: 'dang' or 'ndan' flag
        duration: Duration in seconds to capture
        interval: Interval between captures in seconds
    
    Returns:
        List of captured filenames
    """
    print(f"\nStarting camera for {duration} seconds...")
    print("Get ready! Capturing will start in 2 seconds...")
    time.sleep(2)
    
    # Start camera
    picam2.start()
    time.sleep(1)  # Allow camera to warm up
    
    filenames = []
    start_time = time.time()
    capture_count = 0
    
    print("📸 Capturing images...")
    
    try:
        while time.time() - start_time < duration:
            # Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp}.jpg"
            
            # Capture and save
            picam2.capture_file(filename)
            filenames.append(filename)
            capture_count += 1
            
            print(f"  Captured image {capture_count}: {filename}")
            
            # Wait for next capture
            time.sleep(interval)
    
    finally:
        # Stop camera
        picam2.stop()
        print(f"✓ Capture complete! Total images: {capture_count}\n")
    
    return filenames

def capture_images(cls_name):
    """
    Main function to capture and organize images
    
    Args:
        cls_name: Name of the class to capture
    """
    if cls_name == '':
        print("❌ Please enter a valid class name.")
        return
    
    # Determine flag based on buzzer_class
    flag = 'dang' if buzzer_class == cls_name else 'ndan'
    
    # Create class directory
    class_dir = os.path.join(DATA_DIR, f"{cls_name}_{flag}")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"✓ Created folder: {class_dir}")
    
    # Capture photos
    filenames = take_photos_continuous(cls_name, flag, duration=5, interval=0.5)
    
    # Move files to class directory
    for fname in filenames:
        dest = os.path.join(class_dir, f"{cls_name}{flag}{os.path.basename(fname)}")
        shutil.move(fname, dest)
    
    print(f"✓ Saved {len(filenames)} images to {class_dir}")
    print(f"  Flag: {'⚠  DANGEROUS (will trigger buzzer)' if flag == 'dang' else '✓ SAFE (no buzzer)'}\n")

def flag_class(cls_name):
    """
    Flag a class to trigger buzzer during inference
    
    Args:
        cls_name: Name of class to flag
    """
    global buzzer_class
    
    if cls_name == '':
        print("❌ Enter a valid class name to flag.")
        return
    
    buzzer_class = cls_name
    print(f"⚠  Class '{cls_name}' flagged to trigger buzzer during live inference.\n")

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*60)
    print("  RASPBERRY PI CAMERA DATA COLLECTION")
    print("="*60)
    print("\nOptions:")
    print("  1. Capture images for a class (5 seconds)")
    print("  2. Flag a class for buzzer alert")
    print("  3. View current buzzer class")
    print("  4. List captured data")
    print("  5. Preview camera (2 seconds)")
    print("  6. Exit")
    print("-"*60)

def preview_camera():
    """Quick camera preview"""
    print("\n📷 Starting 2-second camera preview...")
    picam2.start()
    time.sleep(2)
    picam2.stop()
    print("✓ Preview complete\n")

def list_captured_data():
    """List all captured classes and image counts"""
    print("\n" + "="*60)
    print("  CAPTURED DATA SUMMARY")
    print("="*60)
    
    if not os.path.exists(DATA_DIR):
        print("No data captured yet.")
        return
    
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    
    if not folders:
        print("No data captured yet.")
        return
    
    total_images = 0
    for folder in sorted(folders):
        folder_path = os.path.join(DATA_DIR, folder)
        image_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        total_images += image_count
        
        flag_emoji = "⚠ " if "_dang" in folder else "✓ "
        print(f"{flag_emoji} {folder}: {image_count} images")
    
    print("-"*60)
    print(f"Total images: {total_images}")
    print("="*60 + "\n")

def main():
    """Main interactive loop"""
    global buzzer_class
    
    ensure_data_dir()
    
    print("\n🎥 Raspberry Pi Camera Data Collection Tool")
    print("Capture training images for object detection\n")
    
    while True:
        show_menu()
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            cls_name = input("\nEnter class name: ").strip()
            if cls_name:
                capture_images(cls_name)
            else:
                print("❌ Invalid class name.\n")
        
        elif choice == '2':
            cls_name = input("\nEnter class name to flag for buzzer: ").strip()
            flag_class(cls_name)
        
        elif choice == '3':
            if buzzer_class:
                print(f"\n⚠  Current buzzer class: '{buzzer_class}'\n")
            else:
                print("\n✓ No class currently flagged for buzzer.\n")
        
        elif choice == '4':
            list_captured_data()
        
        elif choice == '5':
            preview_camera()
        
        elif choice == '6':
            print("\n👋 Exiting... Goodbye!\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-6.\n")

if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
    finally:
        # Cleanup
        try:
            picam2.stop()
        except:
            pass