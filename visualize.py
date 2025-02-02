import ast
import cv2
import numpy as np
import pandas as pd
import glob
import os
from PIL import Image, ImageDraw, ImageFont

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img

# Load results (from test_interpolated.csv)
results = pd.read_csv('./test_interpolated.csv', encoding='utf-8-sig')

image_folder = './images'
images = sorted(glob.glob(os.path.join(image_folder, '*.*')))

output_dir = './annotated_images'
os.makedirs(output_dir, exist_ok=True)

# Choose a font that supports multiple languages, e.g., NotoNaskhArabic for Arabic and NotoSans for Latin
# You can use a single font that supports both if available
font_path = "NotoNaskhArabic-VariableFont_wght.ttf"

# Verify font file exists
if not os.path.exists(font_path):
    print(f"Font file {font_path} not found. Please ensure it's in the working directory.")
    exit(1)

# Load the font
try:
    font = ImageFont.truetype(font_path, 80)
except IOError:
    print(f"Font file {font_path} not found or is corrupted.")
    exit(1)

for frame_nmr in sorted(results['frame_nmr'].unique(), key=lambda x: int(x)):
    fn = int(frame_nmr)
    if fn < len(images):
        frame = cv2.imread(images[fn])
        if frame is None:
            continue

        df_ = results[results['frame_nmr'] == frame_nmr]

        for _, row in df_.iterrows():
            # Draw car bbox
            car_bbox_str = row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str)
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, line_length_x=200, line_length_y=200)

            # Draw license plate bbox
            lp_bbox_str = row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            x1, y1, x2, y2 = ast.literal_eval(lp_bbox_str)

            if (x1, y1, x2, y2) == (0,0,0,0):
                continue

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_crop.size > 0:
                h_plate = int(y2 - y1)
                w_plate = int(x2 - x1)
                new_height = 400
                scale = new_height / h_plate
                new_width = int(w_plate * scale)
                license_crop_resized = cv2.resize(license_crop, (new_width, new_height))

                H, W, _ = license_crop_resized.shape

                try:
                    # Calculate the position to place the license plate image
                    top_y = int(car_y1) - H - 100  # 100 pixels above the car bbox
                    left_x = int((car_x2 + car_x1 - W) / 2)  # Center horizontally relative to car bbox
                    if top_y < 0:
                        top_y = 0
                    if left_x < 0:
                        left_x = 0
                    if (top_y + H) > frame.shape[0]:
                        top_y = frame.shape[0] - H
                    if (left_x + W) > frame.shape[1]:
                        left_x = frame.shape[1] - W

                    # Ensure the license plate image fits within the frame
                    if (top_y + H) <= frame.shape[0] and (left_x + W) <= frame.shape[1]:
                        frame[top_y:top_y+H, left_x:left_x+W, :] = license_crop_resized

                    # Define white box for text at the bottom left corner
                    text_box_height = 120  # Increased height to accommodate dots
                    text_box_y1 = frame.shape[0] - text_box_height
                    text_box_x1 = 0
                    text_box_width = frame.shape[1]  # Adjust as needed based on text size
                    frame[text_box_y1:frame.shape[0], text_box_x1:text_box_x1+text_box_width, :] = (255, 255, 255)

                    license_text = row['license_number']
                    if pd.isnull(license_text):
                        license_text = ''

                    print("Raw License Text:", license_text)
                    # No reshaping needed since we're handling all languages
                    display_text = license_text

                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    # Calculate text size using textbbox
                    bbox = draw.textbbox((0, 0), display_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Calculate position to center the text within the white box
                    text_x = text_box_x1 + (text_box_width - text_width) // 2
                    text_y = text_box_y1 + (text_box_height - text_height) // 2

                    # Ensure text is within image bounds
                    if text_x < text_box_x1:
                        text_x = text_box_x1
                    if text_y < text_box_y1:
                        text_y = text_box_y1
                    if (text_x + text_width) > (text_box_x1 + text_box_width):
                        text_x = text_box_x1 + text_box_width - text_width
                    if (text_y + text_height) > (text_box_y1 + text_box_height):
                        text_y = text_box_y1 + text_box_height - text_height

                    # Draw the text
                    draw.text((text_x, text_y), display_text, font=font, fill=(0,0,0))
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print("Error placing license text:", e)

        cv2.imwrite(os.path.join(output_dir, f"frame_{fn}.png"), frame)

print("Annotation complete. Check annotated_images folder for results.")
