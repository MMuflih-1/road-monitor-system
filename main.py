from ultralytics import YOLO
import cv2
import glob
import os
import numpy as np
import re
import time
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv
import logging
import torch

torch.use_deterministic_algorithms(True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

results = {}
model = YOLO('best_train_6.pt')
license_plate_class = 0
vehicles = [1]  # trucks
image_folder = './images'

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Load all images (this defines the variable "images")
images = sorted(glob.glob(os.path.join(image_folder, '*.*')), key=natural_keys)
logger.info(f"Total images found: {len(images)}")

# Process each image independently
for i, img_path in enumerate(images):
    logger.info(f"Processing image {i+1}/{len(images)}: {img_path}")
    
    frame = cv2.imread(img_path)
    if frame is None:
        logger.error(f"Failed to load image: {img_path}")
        continue

    results[i] = {}

    # Reinitialize the tracker for each image (independent tracking)
    mot_tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.2)

    # Detect objects in the current frame
    detections = model.predict(
        frame, 
        conf=0.4,  
        iou=0.4,   
        agnostic_nms=True, 
        deterministic=True  # Ensure deterministic output
    )[0]
    logger.info(f"Detections in frame {i}: {len(detections.boxes)}")

    # Prepare detections for vehicle tracking
    vehicle_detections = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            vehicle_detections.append([x1, y1, x2, y2, score])
    
    # Update tracker for the current image
    track_ids = mot_tracker.update(np.asarray(vehicle_detections))
    logger.info(f"Tracked vehicles: {len(track_ids)}")

    # Process license plates in the current frame
    license_plates = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == license_plate_class:
            license_plates.append([x1, y1, x2, y2, score, class_id])

    logger.info(f"License plates found: {len(license_plates)}")

    # For each detected license plate, find the best matching truck using IoU
    for license_plate in license_plates:
        x1, y1, x2, y2, score, class_id = license_plate
        
        # Use the IoU-based matching function
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        
        # If no matching vehicle is found, skip this license plate
        if car_id == -1:
            continue
            
        # Crop and process the license plate image
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

        # Store the results for the current frame and vehicle
        if car_id not in results[i]:
            results[i][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plates': []
            }

        results[i][car_id]['license_plates'].append({
            'bbox': [x1, y1, x2, y2],
            'bbox_score': score,
            'text': license_plate_text if license_plate_text else '',
            'text_score': license_plate_text_score if license_plate_text_score else 0
        })

        time.sleep(1)  # Adjust sleep time if necessary

    logger.info(f"Completed processing frame {i}")

# Write the results to a CSV file
write_csv(results, './test.csv')
logger.info("Processing complete")
