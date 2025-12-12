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
from pypdf import PdfReader

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

def rotate_image(image, student, log_file):
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
            if (h < 100): # need to ignore barcodes on back
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
        custom_print("!! " + student + " Warning: Wrong number of contours (" + str(len(selected_contours)) + ") found for rotation. Verify answers on " + student + " or re-scan.", log_file)
        return image

def crop_to_answers(image, student, log_file, IS_BACK_SIDE=False):
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
            if (h < 100): # need to ignore barcodes on back
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
        if IS_BACK_SIDE:
            x_start, y_start, x_end, y_end  = x_corners[1] + 80, y_corners[1] + 35, x_corners[0] - 460, y_corners[0] - 355
        else:
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
        custom_print("!! " + student + " Warning: Wrong number of contours (" + str(len(selected_contours)) + ") found for cropping. Verify answers on " + student + ".", log_file)
        return image

def process_scanned_page(image, student, log_file): 

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
    TWO_SIDED = 0
    BACK_PAGE = 0
    if count_limit > 50:
        # this is a two sided situation
        TWO_SIDED = 1
        print(student)
        page_number = str(student)
        print(page_number)
        page_number = student[-1]
        page_number = int(page_number)
        if page_number % 2 == 0:
            BACK_PAGE = 1
    if TWO_SIDED == 0:
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
                custom_print("Blank answer detected for question:" + str(row) + "- Consider checking.", log_file)

    else:
        # two sided
        if BACK_PAGE == 0:
            count_limit = 50 # do the first 50 only
            for row, detections in enumerate(detection_results_grid.items()):
                if row >= count_limit:
                    break
                if True in detections[1]:
                    first_true_index = detections[1].index(True)
                    answers.append(CHOICE_INDEX[first_true_index])
                    correct_answer = ANSWER_KEY.get(row) 
                    if first_true_index == correct_answer:
                        points = POINT_VALUES.get(row, DEFAULT_POINTS) 
                        page_score += points 
                else:
                    answers.append("BLANK")
                    custom_print("Blank answer detected for question:" + str(row) + "- Consider checking.", log_file)
        else:
            for row, detections in enumerate(detection_results_grid.items()):
                if row + 50 >= count_limit:
                    break
                if True in detections[1]:
                    first_true_index = detections[1].index(True)
                    answers.append(CHOICE_INDEX[first_true_index])
                    correct_answer = ANSWER_KEY.get(row + 50) 
                    if first_true_index == correct_answer:
                        points = POINT_VALUES.get(row + 50, DEFAULT_POINTS) 
                        page_score += points 
                else:
                    answers.append("BLANK")
                    custom_print("Blank answer detected for question:" + str(row + 50) + "- Consider checking.", log_file)
    
    return [page_score, answers]

def get_parent_dir(directory):
    return os.path.dirname(directory)

def custom_print(message_to_print, log_file):
    print(message_to_print)
    with open(log_file, 'a') as of:
        of.write(message_to_print + '\n')

def main():

    script_dir = get_parent_dir(os.path.dirname(os.path.abspath(__file__)))

    input_dir = script_dir + "\\input"

    answer_key_dir = script_dir + "\\Answer Key"

    output_csv = script_dir + "\\output\\Half-Sheet_Results_" + str(datetime.now().strftime('%Y-%m-%d-%H-%M')) + ".csv"
    log_file = script_dir + "\\output\\Log_" + str(datetime.now().strftime('%Y-%m-%d-%H-%M')) + ".txt"

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

        # else:
        #     print(f"!! No answer key csv found in '{answer_key_dir}'")
        #     input("Press Enter to exit...")
        #     # sys.exit(1)

    DOUBLE_SIDED_FLAG = 0
    # print(ANSWER_KEY)
    # print(POINT_VALUES)
    user_input = input("Is your test double-sided/greater than 50 questions? Enter the letter y for yes, or nothing for no, and press enter: ")
    if user_input:
        if user_input[0].lower() == 'y':
            custom_print("User indicated double-sided/greater than 50 questions.", log_file)
            DOUBLE_SIDED_FLAG = 1
        else:
            custom_print("Unexpected selection. Defaulting to single-sided/50 questions or less.", log_file)
    else:
        custom_print("User indicated single-sided/50 questions or less.", log_file)
    

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

                # try:
                reader = PdfReader(pdf_path)
                num_pages = len(reader.pages)
                print(f"The PDF has {num_pages} pages.")

                pages_as_images = list()
                page_floor = num_pages // 50
                page_modulo = num_pages % 50
                start_page = 1
                if page_floor > 0:
                    for i in range(page_floor):
                        end_page = start_page + 49
                        print(f"Processing pages_as_images for {start_page} through {end_page} of {num_pages}...")
                        pages_as_images.extend(convert_from_path(pdf_path, dpi=300, poppler_path=local_poppler_path,
                                                                    first_page=start_page, last_page=end_page))
                        start_page += 50
                if page_modulo > 0:
                    end_page = start_page + page_modulo - 1
                    print(f"Processing pages_as_images for {start_page} through {end_page} of {num_pages}...")
                    pages_as_images.extend(convert_from_path(pdf_path, dpi=300, poppler_path=local_poppler_path,
                                        first_page=start_page, last_page=end_page))

                print("PDF read successfully. Preparing to grade.")
                if DOUBLE_SIDED_FLAG == 0:
                    for i, page_image_pil in enumerate(pages_as_images): 
                        student = f"Page_{i+1}"
                        custom_print(student, log_file)

                        page_image_cv = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR) 
                        page_image_cv = rotate_image(page_image_cv, student, log_file)
                        page_image_cv = crop_to_answers(page_image_cv, student, log_file)

                        exam_results = process_scanned_page(page_image_cv, student, log_file) 
                        
                        student_row = [student, exam_results[0]]
                        student_row.extend(exam_results[1])
                        writer.writerow(student_row)
                else:
                    temp_exam_results = []
                    for i, page_image_pil in enumerate(pages_as_images): 
                        if (i + 1) % 2 == 0:
                            # the second/back page
                            student = f"Page_{i+1}"
                            custom_print(student, log_file)
                            page_image_cv = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR) 
                            page_image_cv = rotate_image(page_image_cv, student, log_file)
                            page_image_cv = crop_to_answers(page_image_cv, student, log_file, True)
                            
                            back_page_results = process_scanned_page(page_image_cv, student, log_file)
                            # print(back_page_results)
                            # print(temp_exam_results)

                            score1 = temp_exam_results[0]
                            score2 = back_page_results[0]

                            # print("score 1:", score1)
                            # print("score 2:", score2)

                            score = int(score1) + int(score2)
                            answers = temp_exam_results[1]
                            answers.extend(back_page_results[1])

                            student_row = [student, score]
                            student_row.extend(answers)
                            writer.writerow(student_row)
                        else:
                            # a new student
                            student = f"Page_{i+1}"
                            temp_exam_results.clear()
                            custom_print(student, log_file)
                            page_image_cv = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR) 
                            page_image_cv = rotate_image(page_image_cv, student, log_file)
                            page_image_cv = crop_to_answers(page_image_cv, student, log_file)

                            temp_exam_results = process_scanned_page(page_image_cv, student, log_file)


                # except Exception as e: 
                #     print(f"!! ERROR processing {filename}: {e}") 
                #     writer.writerow(["ERROR"]) 

    print(f"Batch processing complete. Scores saved to '{output_csv}'. Scores saved to '{output_csv}'. Console alerts saved to '{log_file}'.") 
    input("Press Enter to exit...")
    # sys.exit(0)

if __name__ == "__main__": 

    main() 