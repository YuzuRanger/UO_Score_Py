# grade_half_sheets_from_pdf.py

# A Python script to grade half sheets scanned to PDFs.
# Adapted from R script to identify and record marks on the UO Score Answer form by Jeremy Piger
# Assisted by M365 Copilot & Gemini

import cv2
import numpy as np
import os
import csv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from datetime import datetime
import math

# 1. Answer Key & Points 

ANSWER_KEY = { } 

# ANSWER_KEY = { 
#     # Question Index (0-299) : Choice Index (0=A, 1=B, 2=C, 3=D, 4=E) 
#     0: 1, 1: 4, 2: 0, 3: 3, 4: 1, # Example answers for questions 1-5 
# }

POINT_VALUES = { }

# POINT_VALUES = { 
#     # Question Index : Point value 
#     0: 2, # Questions 1 worth 2 points 
#     # All other questions will be worth DEFAULT_POINTS 
# } 

DEFAULT_POINTS = 1

CHOICE_INDEX = [
    "A", "B", "C", "D", "E"
]

def image_to_grid(img, student):
    img = Image.fromarray(img)
    img_width, img_height = img.size

    # Define grid dimensions
    rows = 50
    cols = 5

    # Calculate cell dimensions
    cell_width = img_width / cols
    cell_height = img_height / rows

    # Create a drawing context
    draw = ImageDraw.Draw(img)


    # Dark pixel detection parameters
    dark_pixel_cutoff = 90  # Grayscale value below which a pixel is considered dark
    dark_pixel_threshold_percent = 7  # Percentage threshold to consider a cell as dark


    # Generate and draw grid cells with brightness detection
    grid_cells = []
    detection_results = {}

    for r in range(rows):
        row_detections = []
        for c in range(cols):
            x_start = int(c * cell_width)
            y_start = int(r * cell_height)
            x_end = int(x_start + cell_width)
            y_end = int(y_start + cell_height)

            # Crop cell and convert to grayscale
            cell_img = img.crop((x_start, y_start, x_end, y_end)).convert("L")
            pixels = list(cell_img.getdata())
            dark_pixels = [p for p in pixels if p < dark_pixel_cutoff]
            dark_percent = (len(dark_pixels) / len(pixels)) * 100


            # Check against threshold
            if dark_percent > dark_pixel_threshold_percent:
                draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=2)
                row_detections.append(True)
            else:
                # draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=2)
                row_detections.append(False)


            grid_cells.append(((x_start, y_start), (x_end, y_end)))

        detection_results[f"row_{r}"] = row_detections

    # Save the output image
    script_dir = get_parent_dir(os.path.dirname(os.path.abspath(__file__)))
    output_path = script_dir + "\\output\\detected_answers-" + str(student) + '.png'
    img.save(output_path)
    # print(f"Grid layout with dark pixel detection saved as {output_path}.")
    return detection_results

def rotate_image(image, student):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 83, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    selected_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 2220: # for the big rectangle markers
            x, y, w, h = cv2.boundingRect(cnt)
            selected_contours.append((x, y, x + w, y + h))

    # Convert image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Draw rectangles around filtered contours
    for cnt in selected_contours:
        draw.rectangle(cnt, outline="red", width=3)
    # Save the visualization
    # image_pil.save("guide_marker_visualization" + student + ".png")
    # print("Guide markers visualized and saved as 'guide_marker_visualization.png'.")

    # Proceed only if two contours are found
    if len(selected_contours) == 3:
        # remove left side rectangle
        removed_element = selected_contours.pop(1)
        # print(selected_contours)
        # Compute centers of the two contours
        corners = []
        for cnt in selected_contours:
                cx = cnt[0]
                cy = cnt[1]
                corners.append((cx, cy))

        # print("corners:", corners)
        # Calculate angle between the two corners
        (x1, y1), (x2, y2) = corners
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Adjust angle if necessary
        if angle < -45:
            angle += 90

        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # cv2.imwrite("rotated_omr_cv.png", rotated)
        # print(f"Image rotated by {angle:.2f} degrees using two contours.")
        return rotated
    else:
        print("!! " + student + " Warning: Not enough contours found for rotation. Verify answers on " + student + " or re-scan.")
        return image

def crop_to_answers(image, student):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 83, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    selected_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 2220: # for the big rect markers
            x, y, w, h = cv2.boundingRect(cnt)
            selected_contours.append((x, y, x + w, y + h))

    # Convert image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # draw = ImageDraw.Draw(image_pil)

    # Proceed only if three contours are found
    if len(selected_contours) == 3:
        # remove top rectangle
        removed_element = selected_contours.pop(2)
        # print(selected_contours)
        # Compute centers of the two contours
        y_corners = []
        x_corners = []
        for cnt in selected_contours:
                cx = cnt[2]
                cy = cnt[1]
                y_corners.append(cy)
                x_corners.append(cx)

        # print("y corners:", y_corners)
        # print("x corners:", x_corners)
        x_start, y_start, x_end, y_end  = x_corners[1] + 80, y_corners[1] + 35, x_corners[0] - 460, y_corners[0] - 115
        image = image[y_start:y_end, x_start:x_end]
        # filename = "temp_image.png"
        # cv2.imwrite(filename, image)
        # print(f"Image cropped using two contours.")
        return image
    else:
        # Crop a region from (x_start, y_start) to (x_end, y_end)
        # Y values: the top row should start at about 480, and each 5th/10 qs is 500 px long. 
        # X values: Each bubble is 60px wide with 20px on each side as a margin, so 100px total width
        x_start, y_start, x_end, y_end = 230, 500, 730, 3000
        image = image[y_start:y_end, x_start:x_end]
        # filename = "temp_image_default.png"
        # cv2.imwrite(filename, image)
        print("!! " + student + " Warning: Not enough contours found for cropping. Verify answers on " + student + ".")
        return image

def process_scanned_page(image, student): 

    """ 
    Processes a single image (one side of a scanned bubble form) and returns its score. 
    """ 

    detection_results_grid = image_to_grid(image, student)

    page_score = 0
    answers = []

    # print("Detection results by row:")
    # for row, detections in enumerate(detection_results_grid.items()):
    # for i in range(len(ANSWER_KEY)):
    count_limit = len(ANSWER_KEY)
    for row, detections in enumerate(detection_results_grid.items()):
        if row >= count_limit:
            break
        #print(f"{row}: {detections}")
        if True in detections[1]:
            # print("true was found")
            first_true_index = detections[1].index(True)
            answers.append(CHOICE_INDEX[first_true_index])
            # print("first_true_index", first_true_index)
            correct_answer = ANSWER_KEY.get(row) 
            # print("correct_answer", correct_answer)
            if first_true_index == correct_answer:
                # print("A CORRECT ANSWER!")
                points = POINT_VALUES.get(row, DEFAULT_POINTS) 
                page_score += points 
        else:
            #print("blank answer")
            answers.append("BLANK")
    
    return [page_score, answers]

def get_parent_dir(directory):
    return os.path.dirname(directory)

def main():

    script_dir = get_parent_dir(os.path.dirname(os.path.abspath(__file__)))

    input_dir = script_dir + "\\input"

    answer_key_dir = script_dir + "\\Answer Key"

    output_csv = script_dir + "\\output\\Half-Sheet_Results_" + str(datetime.now().strftime('%Y-%m-%d')) + ".csv"

    # Delete any previous pngs saved of answers
    output_images_dir = script_dir + "\\Output"
    for filename in os.listdir(output_images_dir):
        if filename.endswith('.png'):
            file_path = os.path.join(output_images_dir, filename)
            os.remove(file_path)
            # print(f"Deleted: {filename}")

    for ak_filename in sorted(os.listdir(answer_key_dir)): 
        if ak_filename.lower().endswith('.csv'): 
            # print(ak_filename)
            answer_key_path = os.path.join(answer_key_dir, ak_filename) 
            # print(answer_key_path)
            with open(answer_key_path, 'r', newline='') as answer_key:
                csv_dict_reader = csv.DictReader(answer_key)
                for index, row_dict in enumerate(csv_dict_reader, start=0):
                    # print(index)
                    # print(row_dict)
                    ANSWER_KEY[index] = CHOICE_INDEX.index(row_dict['Answer'])
                    if row_dict['Points']:
                        POINT_VALUES[index] = int(row_dict['Points'])

        else:
            print(f"!! No answer key csv found in '{answer_key_dir}'")
            input("Press Enter to exit...")
            # sys.exit(1)


    # print(ANSWER_KEY)
    # print(POINT_VALUES)

    with open(output_csv, 'w', newline='') as f: 

        writer = csv.writer(f) 

        header_list =["Scanned Page", "Score"]
        for i in range(len(ANSWER_KEY)):
            header_list.append(f"Q_{i+1}")

        writer.writerow(header_list) # CSV Header 

        print("Starting batch processing of scanned half sheet bubble form PDFs...") 

        for filename in sorted(os.listdir(input_dir)): 

            if filename.lower().endswith('.pdf'): 

                pdf_path = os.path.join(input_dir, filename) 
                local_poppler_path = script_dir + "\\poppler-25.07.0\\Library\\bin"

                try: 
                    pages_as_images = convert_from_path(pdf_path, dpi=300, poppler_path=local_poppler_path) 

                    for i, page_image_pil in enumerate(pages_as_images): 
                        student = f"Page_{i+1}"
                        # Convert PIL image to OpenCV format (numpy array) 
                        page_image_cv = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR) 
                        # filename = "page_image_cv.png"
                        # cv2.imwrite(filename, page_image_cv)

                        page_image_cv = rotate_image(page_image_cv, student)
                        page_image_cv = crop_to_answers(page_image_cv, student)

                        exam_results = process_scanned_page(page_image_cv, student) 
                        
                        student_row = [student, exam_results[0]]
                        student_row.extend(exam_results[1])

                        # print(f"  - Page {i+1} of {filename}: {exam_results[0]} points") 
                        writer.writerow(student_row) 

                except Exception as e: 
                    print(f"!! ERROR processing {filename}: {e}") 
                    writer.writerow(["ERROR"]) 

    print(f"Batch processing complete. Scores saved to '{output_csv}'.") 
    input("Press Enter to exit...")
    # sys.exit(0)

if __name__ == "__main__": 

    main() 