import os
import cv2
import requests
import time
import threading
import queue
import uuid  # For unique task identifiers
import csv
import numpy as np
from scipy.interpolate import interp1d

# Azure API key and endpoint
subscription_key = "YOUR_API_KEY" # API KEY
endpoint = "YOUR_ENDPOINT" # ENDPOINT


# OCR API URL
text_recognition_url = endpoint + "vision/v3.2/read/analyze"

# Queue for OCR tasks
ocr_queue = queue.Queue()

# Dictionary to store OCR results
# Key: task_id (UUID), Value: (text, score)
ocr_results = {}

# Lock for thread-safe access to ocr_results
results_lock = threading.Lock()

def ocr_worker():
    """
    Worker thread function that processes OCR tasks from the queue
    at a rate of 20 calls per minute.
    """
    while True:
        task = ocr_queue.get()
        if task is None:
            # Sentinel received, terminate the worker
            break

        task_id, license_plate_crop = task

        # Perform OCR
        text, score = perform_ocr(license_plate_crop)

        # if OCR detects letters that arent in saudi arabia license plate, replace them with the correct coresponding letter e.g. "ج" with "ح" in the OCR result
        if text:
            text = text.replace('ج', 'ح')
            text = text.replace('ش', 'س')
            text = text.replace('ض', 'ص')
            text = text.replace('ظ', 'ط')
            text = text.replace('ت', 'ب')
            text = text.replace('ث', 'ب')
            text = text.replace('ي', 'ى')
            text = text.replace('غ', 'ع')
            text = text.replace('ف', 'ق')
            text = text.replace('ز', 'ر')
            text = text.replace('ذ', 'د')
            text = text.replace('إ', 'ا')
            text = text.replace('Y', 'V')

        # Store the result
        with results_lock:
            ocr_results[task_id] = (text, score)

        # Sleep to maintain rate limit (20 calls per minute => 5 seconds per call)
        time.sleep(5)

        ocr_queue.task_done()

def perform_ocr(license_plate_crop):
    """
    Sends the license plate image to Azure OCR and retrieves the text.
    Returns a tuple of (text, confidence_score).
    """
    try:
        h, w, _ = license_plate_crop.shape
        if h < 50 or w < 50:
            print(f"License plate image too small: {w}x{h}. Attempting to resize.")
            scale_factor = max(50 / h, 50 / w)
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            license_plate_crop = cv2.resize(license_plate_crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            h, w, _ = license_plate_crop.shape
            print(f"Resized license plate image to: {w}x{h}.")
            if h < 50 or w < 50:
                print(f"Resized license plate image still too small: {w}x{h}. Skipping OCR.")
                return 'N/A', 0.0

        # Encode image to JPEG
        success, image_data = cv2.imencode('.jpg', license_plate_crop)
        if not success:
            print("Image encoding failed.")
            return None, None

        image_bytes = image_data.tobytes()

        headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-Type': 'application/octet-stream'
        }

        response = requests.post(text_recognition_url, headers=headers, data=image_bytes)
        if response.status_code != 202:
            print(f"Azure OCR request failed: {response.text}")
            return None, None

        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            print("No Operation-Location found in headers.")
            return None, None

        # Poll for OCR results
        max_retries = 10
        retry_count = 0
        while True:
            result_response = requests.get(operation_location, headers={'Ocp-Apim-Subscription-Key': subscription_key})
            analysis = result_response.json()
            status = analysis.get('status')

            if status == 'succeeded':
                break
            elif status == 'failed':
                print("Azure OCR analysis failed.")
                return None, None

            time.sleep(1)  # Wait before polling again
            retry_count += 1
            if retry_count > max_retries:
                print("Timeout waiting for Azure OCR results.")
                return None, None

        read_results = analysis.get('analyzeResult', {}).get('readResults', [])
        if not read_results:
            return None, None

        all_text = []
        for page in read_results:
            for line in page.get('lines', []):
                text = line.get('text', '').strip()
                if text:
                    all_text.append(text)

        if not all_text:
            return None, None

        combined_text = ' '.join(all_text)

        return combined_text, 1.0  # Azure OCR doesn't provide confidence per text
    except Exception as e:
        print(f"Exception during OCR request: {e}")
        return None, None

# Start the OCR worker thread
worker_thread = threading.Thread(target=ocr_worker, daemon=True)
worker_thread.start()

def read_license_plate(license_plate_crop):
    """
    Enqueue the OCR task and wait for the result.
    Returns a tuple of (text, confidence_score).
    """
    # Generate a unique task ID
    task_id = uuid.uuid4()
    ocr_queue.put((task_id, license_plate_crop))

    # Wait for the OCR result
    while True:
        with results_lock:
            if task_id in ocr_results:
                text, score = ocr_results.pop(task_id)
                return text, score
        time.sleep(0.1)  # Prevent busy waiting

def shutdown_ocr_worker():
    """
    Signals the OCR worker to terminate and waits for the thread to finish.
    """
    ocr_queue.put((None, None))  # Sentinel value to stop the worker
    worker_thread.join()

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]
    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    """
    Write the results to a CSV file. Supports multiple license plates per car_id per frame.
    Each plate results in a separate row.
    """
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
            'license_number_score'
        ))

        for frame_nmr in sorted(results.keys()):
            for car_id in results[frame_nmr].keys():
                car_bbox = results[frame_nmr][car_id]['car']['bbox']
                if 'license_plates' in results[frame_nmr][car_id] and len(results[frame_nmr][car_id]['license_plates']) > 0:
                    # Multiple plates: write a row for each
                    for plate in results[frame_nmr][car_id]['license_plates']:
                        f.write('{},{},[{} {} {} {}],[{} {} {} {}],{},{},{}\n'.format(
                            frame_nmr,
                            car_id,
                            car_bbox[0], car_bbox[1], car_bbox[2], car_bbox[3],
                            plate['bbox'][0], plate['bbox'][1], plate['bbox'][2], plate['bbox'][3],
                            plate['bbox_score'],
                            plate['text'],
                            plate['text_score']
                        ))
                else:
                    # No plates, write a single row with zeros
                    f.write('{},{},[{} {} {} {}],[0 0 0 0],0,0,0\n'.format(
                        frame_nmr,
                        car_id,
                        car_bbox[0], car_bbox[1], car_bbox[2], car_bbox[3]
                    ))
