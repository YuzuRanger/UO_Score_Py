# grade_red_half_sheets_from_pdf.py

# A Python script to grade the red half-page sheets scanned to PDFs.
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
# refactor this to be dynamic later

ANSWER_KEY_1 = { }
ANSWER_KEY_2 = { }
ANSWER_KEY_3 = { }
ANSWER_KEY_4 = { }
ANSWER_KEY_5 = { }

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
    cols = 23

    # Calculate cell dimensions
    cell_width = img_width / cols
    cell_height = img_height / rows

    # Create a drawing context
    draw = ImageDraw.Draw(img)


    # Dark pixel detection parameters
    dark_pixel_cutoff = 135  # Grayscale value below which a pixel is considered dark
    dark_pixel_threshold_percent = 5  # Percentage threshold to consider a cell as dark


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
                # draw.rectangle([x_start, y_start, x_end, y_end], outline="blue", width=2)
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
    # Reduce green
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the red color in HSV
    # Saturation and Value ranges can be adjusted for your specific image.
    # lower_red = np.array([0, 100, 100])
    # upper_red = np.array([10, 255, 255])

    # Define lower and upper bounds for red color in HSV
    # Lower red range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Upper red range (since red wraps around in HSV hue)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for each red range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    # red_only_image = cv2.bitwise_and(image, image, mask=red_mask)
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    image[red_mask > 0] = (255, 255, 255)

    # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # cv2.imshow('Original Image', image)
    # cv2.namedWindow('Red Mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('Red Mask', red_mask)
    # cv2.namedWindow('Red Only', cv2.WINDOW_NORMAL)
    # cv2.imshow('Red Only', red_only_image)

    # # cv2.imshow('Red Range to White', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    selected_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 600 < area: # for the small rectangle markers. there are 63 total but only 61 we care about
            x, y, w, h = cv2.boundingRect(cnt)
            if x < 90: # we need to ignore any dark over-filled bubbles and the extra bars
                selected_contours.append((x, y, x + w, y + h))

    # Convert image to PIL format
    # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # draw = ImageDraw.Draw(image_pil)

    # Draw rectangles around filtered contours
    # for cnt in selected_contours:
        # draw.rectangle(cnt, outline="blue", width=3)
    # # Save the visualization
    # image_pil.save("guide_marker_visualization" + student + ".png")
    # print(len(selected_contours), "guide markers visualized and saved as 'guide_marker_visualization.png'.")

    # Proceed if 61 or 62 contours are found
    if 61 <= len(selected_contours) <= 62:
        # print(selected_contours)
        if len(selected_contours) == 62:
            # print(f"Original list: {selected_contours}")
            # remove the one with the furthest x value (x, y, x + w, y + h)
            max_x_tuple = max(selected_contours, key=lambda c: c[2])
            max_x_value = max_x_tuple[0]
            # print(f"Tuple with max x-value ({max_x_value}): {max_x_tuple}")
            selected_contours = [c for c in selected_contours if c[2] < max_x_value]
            # print(f"Filtered list: {selected_contours}")


        # Compute centers of the two or more contours
        corners = []
        for cnt in selected_contours:
                cx = cnt[0]
                cy = cnt[1]
                corners.append((cx, cy))

        # print("corners:", corners)
        # take the first and last corners
        top_corner = corners[0]
        bottom_corner = corners[-1]
        # Calculate angle between the two corners
        (x1, y1), (x2, y2) = top_corner, bottom_corner
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
        # Save the visualization
        # image_pil.save("guide_marker_visualization" + student + ".png")
        # print(len(selected_contours), "guide markers visualized and saved as 'guide_marker_visualization.png'.")
        custom_print("!! " + student + " Warning: Wrong number of contours (" + str(len(selected_contours)) + ") found for rotation. Verify answers on " + student + " or re-scan.", log_file)
        return image

def crop_to_answers(image, student, log_file):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('Threshold', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    selected_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 600 < area: # for the small rectangle markers. there are 63 total but only 61 we care about
            x, y, w, h = cv2.boundingRect(cnt)
            if x < 90: # we need to ignore any dark over-filled bubbles
                selected_contours.append((x, y, x + w, y + h))

    # Convert image to PIL format
    # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # draw = ImageDraw.Draw(image_pil)
    # Draw rectangles around filtered contours
    # for cnt in selected_contours:
        # draw.rectangle(cnt, outline="orange", width=3)
    # # Save the visualization
    # image_pil.save("crop_visualization" + student + ".png")
    # print(len(selected_contours), "guide markers visualized and saved as 'guide_marker_visualization.png'.")

    # if we are missing a few we will take our chances
    if  1 < len(selected_contours) <= 61:
        if len(selected_contours) != 61:
            custom_print("!! " + student + " Warning: Not enough contours (" + str(len(selected_contours)) + ") found for accurate cropping. Verify answers on " + student + ".", log_file)
        # print(selected_contours)
        # Compute centers
        y_corners = []
        x_corners = []
        for cnt in selected_contours:
                cx = cnt[2]
                cy = cnt[1]
                y_corners.append(cy)
                x_corners.append(cx)
        y_corners.sort()
        x_corners.sort()
        # print("y corners:", y_corners)
        # print("x corners:", x_corners)
        x_start, y_start, x_end, y_end  = x_corners[0], y_corners[0] + 540, x_corners[-1] + 1155, y_corners[-1] + 33
        image = image[y_start:y_end, x_start:x_end]
        # filename = "temp_image" + student + ".png"
        # cv2.imwrite(filename, image)
        # print(f"Image cropped using two contours.")
        return image
    else:
        # Save the visualization
        # image_pil.save("crop_visualization" + student + ".png")
        # print(len(selected_contours), "guide markers visualized and saved as 'guide_marker_visualization.png'.")
        custom_print("!! " + student + " Warning: Wrong number of contours (" + str(len(selected_contours)) + ") found for cropping. Verify answers on " + student + ".", log_file)
        # Crop a region from (x_start, y_start) to (x_end, y_end)
        # Y values: the top row should start at about 480, and each 5th/10 qs is 500 px long.
        # X values: Each bubble is 60px wide with 20px on each side as a margin, so 100px total width
        x_start, y_start, x_end, y_end = 100, 620, 1255, 3120
        image = image[y_start:y_end, x_start:x_end]
        # filename = "temp_image_default.png"
        # cv2.imwrite(filename, image)
        return image

def process_scanned_page(image, student, log_file):

    """
    Processes a single image (one side of a scanned bubble form) and returns its score.
    """

    detection_results_grid = image_to_grid(image, student)
    numpy_array_values = np.array(list(detection_results_grid.values()))

    last_name = []
    first_name = []
    test_num = []
    page_score = 0
    answers = []

    try:
        # ID Number
        test_form_grid = numpy_array_values[10:20, 14:23]
        # print(test_form_grid)
        indices = []
        for column in test_form_grid.T:
            # print(f"Column: {column}")
            # indices.append(np.where(column)[0])
            index = np.where(column)[0]
            # print(index[0])
            indices.append(int(index[0]))
        # print(indices)
        id_num = ''.join(map(str, indices))
        # print(id_num)
    except:
        id_num = 99
        custom_print("ID number unreadable on " + student, log_file)

    # Test Form Letter/Number
    # looks like [False False  True False False False] when B is filled
    test_form_grid = numpy_array_values[24, 15:21]
    # print(test_form_grid)
    indices = np.where(test_form_grid)
    test_num = ''.join(map(str, indices[0]))
    # print("test num:", test_num)

    custom_print(student+ ',' + str(id_num) + ',' + str(test_num), log_file)
    if test_num:
        # defaults to first selected
        test_num = int(test_num[0])
        if int(test_num) == 0:
            form = 1 # for A
        elif int(test_num) == 2:
            form = 2 # for B
        elif int(test_num) == 4:
            form = 3 # for C
        elif int(test_num) == 6:
            form = 4 # for D
        # form = int(test_num)
        else:
            form = 99
            custom_print("Test Form number unreadable. Defaulting to Answer Key 1. Verify answers on " + student + " or re-scan.", log_file)
    else:
        form = 99
        custom_print("Test Form number unreadable. Defaulting to Answer Key 1. Verify answers on " + student + " or re-scan.", log_file)


    if form == 1:
        answer_key = ANSWER_KEY_1
    elif form == 2:
        answer_key = ANSWER_KEY_2
    elif form == 3:
        answer_key = ANSWER_KEY_3
    elif form == 4:
        answer_key = ANSWER_KEY_4
    else:
        answer_key = ANSWER_KEY_1

    count_limit = len(answer_key)
    question_iterator = 1
    column_iterator = 0
    for column_iterator in range(1):
        if question_iterator - 1 > count_limit:
            break

        if column_iterator == 0:
            test_form_grid = numpy_array_values[0:50, 1:6]
        elif column_iterator == 1:
            test_form_grid = numpy_array_values[35:50, 8:13]

        for detections in test_form_grid:
            if question_iterator > count_limit:
                break
            # print(detections)
            if True in detections:
                true_indices = np.where(detections)
                # print(true_indices)
                true_indices = ''.join(map(str, true_indices[0]))
                # print(true_indices)
                # defaults to first selected answer, add support for multi later
                first_selected_answer = int(true_indices[0])
                answers.append(CHOICE_INDEX[first_selected_answer])
                correct_answer = answer_key.get(question_iterator - 1)
                # print("true indices", true_indices)
                # print("correct answer", correct_answer)
                if int(first_selected_answer) == correct_answer:
                    points = POINT_VALUES.get(question_iterator - 1, DEFAULT_POINTS)
                    page_score += points
                    # print(question_iterator, "correct. points:", page_score)
                # else:
                    # print(question_iterator, "incorrect. answered: ", first_selected_answer, "correct was:", correct_answer)
            else:
                answers.append("BLANK")
                custom_print("Blank answer detected for question:" + str(question_iterator) + "- Consider checking.", log_file)
            question_iterator += 1
        column_iterator += 1


    # for row, detections in enumerate(detection_results_grid.items()):
    #     if row >= count_limit:
    #         break
    #     if True in detections[1]:
    #         first_true_index = detections[1].index(True)
    #         answers.append(CHOICE_INDEX[first_true_index])
    #         correct_answer = answer_key.get(row)
    #         if first_true_index == correct_answer:
    #             points = POINT_VALUES.get(row, DEFAULT_POINTS)
    #             page_score += points
    #     else:
    #         answers.append("BLANK")

    return [page_score, answers, id_num, form, last_name, first_name]

def get_parent_dir(directory):
    return os.path.dirname(directory)

def index_to_letter(number):
    number += 1
    if 1 <= number <= 26:
        return chr(ord('A') + number - 1)
    else:
        return ""

def custom_print(message_to_print, log_file):
    print(message_to_print)
    with open(log_file, 'a') as of:
        of.write(message_to_print + '\n')

def main():

    script_dir = get_parent_dir(os.path.dirname(os.path.abspath(__file__)))

    input_dir = script_dir + "\\input"

    answer_key_dir = script_dir + "\\Answer Key"

    output_csv = script_dir + "\\output\\Full-Sheet_Results_" + str(datetime.now().strftime('%Y-%m-%d-%H-%M')) + ".csv"

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
                    ANSWER_KEY_1[index] = CHOICE_INDEX.index(row_dict['Form 1'])
                    if row_dict['Form 2']:
                        ANSWER_KEY_2[index] = CHOICE_INDEX.index(row_dict['Form 2'])
                    if row_dict['Form 3']:
                        ANSWER_KEY_3[index] = CHOICE_INDEX.index(row_dict['Form 3'])
                    if row_dict['Form 4']:
                        ANSWER_KEY_4[index] = CHOICE_INDEX.index(row_dict['Form 4'])
                    if row_dict['Points']:
                        POINT_VALUES[index] = int(row_dict['Points'])

        else:
            print(f"!! No answer key csv found in '{answer_key_dir}'")
            input("Press Enter to exit...")
            # sys.exit(1)


    # print(ANSWER_KEY_1)
    # print(POINT_VALUES)

    with open(output_csv, 'w', newline='') as f:

        writer = csv.writer(f)

        header_list = ["Scanned Page", "ID", "Form", "Score"]
        for i in range(len(ANSWER_KEY_1)):
            header_list.append(f"Q_{i+1}")

        writer.writerow(header_list) # CSV Header

        print("Starting batch processing of scanned full sheet bubble form PDFs...")

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
                for i, page_image_pil in enumerate(pages_as_images):
                    student = f"Page_{i+1}"
                    custom_print(student, log_file)
                    # Convert PIL image to OpenCV format (numpy array)
                    page_image_cv = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
                    # filename = "page_image_cv.png"
                    # cv2.imwrite(filename, page_image_cv)

                    page_image_cv = rotate_image(page_image_cv, student, log_file)
                    page_image_cv = crop_to_answers(page_image_cv, student, log_file)

                    exam_results = process_scanned_page(page_image_cv, student, log_file)

                    # [page_score, answers, id_num, form]
                    # ["Scanned Page", "ID", "Form", "Score"]
                    student_row = [student, exam_results[2], exam_results[3], exam_results[0]]
                    student_row.extend(exam_results[1])

                    # print(f"  - Page {i+1} of {filename}: {exam_results[0]} points")
                    writer.writerow(student_row)

                # except Exception as e:
                #     print(f"!! ERROR processing {filename}: {e}")
                #     writer.writerow(["ERROR"])

    print(f"Batch processing complete. Scores saved to '{output_csv}'. Console alerts saved to '{log_file}'.")
    input("Press Enter to exit...")
    # sys.exit(0)

if __name__ == "__main__":

    main()