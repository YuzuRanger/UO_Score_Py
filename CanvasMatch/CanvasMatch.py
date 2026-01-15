import tkinter as tk
from tkinter import filedialog
import os
import csv
from datetime import datetime

class Student:
    """
    Data Structure for holding student ID and Score
    """
    def __init__(self, id, score, sis_user_id="", sis_login="", name=""):
        self.id = id
        self.score = score
        self.sis_user_id = sis_user_id
        self.sis_login = sis_login
        self.name = name

def get_parent_dir(directory):
    return os.path.dirname(directory)

def canvas_match_results(results_file_path, canvas_roster_file_path):
    output_csv = get_parent_dir(os.path.dirname(os.path.abspath(__file__))) + "\\Output_for_PS2CANVAS_" + str(datetime.now().strftime('%Y-%m-%d-%H-%M')) + ".csv"
    
    students = []
    with open(results_file_path, 'r', newline='') as results_file:
        csv_dict_reader = csv.DictReader(results_file)
        for row_dict in csv_dict_reader:
            students.append(Student(row_dict['ID'], row_dict['Score']))
            # print(row_dict)
    
    found_students = []
    with open(canvas_roster_file_path, 'r', newline='') as roster_file:
        csv_dict_reader = csv.DictReader(roster_file)
        for row_dict in csv_dict_reader:
            target_uoid = row_dict['ID']
            found_student = next((obj for obj in students if obj.id == target_uoid), None)
            # print(row_dict)
            # print(found_student)
            if found_student:
                # print(found_student.id)
                found_student.sis_user_id = row_dict['SIS User ID']
                found_student.sis_login = row_dict['SIS Login ID']
                found_student.name = row_dict['Student']
                found_students.append(found_student)


    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header_list = ["Student", "ID", "SIS User ID", "SIS Login ID", "Section", "TBD"]
        writer.writerow(header_list)
        header_list = ["", "", "", "", "", "Muted"]
        writer.writerow(header_list)
        header_list = ["Points Possible", "", "", "", "", "TBD"]
        writer.writerow(header_list)

        for student in found_students:
            # print(student.name, student.id, student.sis_user_id, student.sis_login, student.score)
            results_row = [student.name, student.id, student.sis_user_id, student.sis_login, "", student.score]
            writer.writerow(results_row)
            
    print(f"Matching complete. Results saved to '{output_csv}'.")

def select_file():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw() 
    
        
    print("Select a UO Score Results File")
    # Open the file dialog and get the selected file path
    results_file_path = filedialog.askopenfilename(
        initialdir = get_parent_dir(os.path.dirname(os.path.abspath(__file__))),
        title = "Select a UO Score Results File", # Sets the dialog title
        filetypes = [('CSV Files', '*.csv'), ('All Files', '*.*')] # Filters file types
    )
    
    if results_file_path:
        print(f"Selected UO Score Results File: {results_file_path}")
        print("Select a Canvas Roster File")
        canvas_roster_file_path = filedialog.askopenfilename(
            initialdir = get_parent_dir(os.path.dirname(os.path.abspath(__file__))),
            title = "Select a Canvas Roster File", # Sets the dialog title
            filetypes = [('CSV Files', '*.csv'), ('All Files', '*.*')] # Filters file types
        )

        if canvas_roster_file_path:
            print(f"Selected Canvas Roster File: {canvas_roster_file_path}")
            canvas_match_results(results_file_path, canvas_roster_file_path)
        else:
            print("File selection cancelled.")
        
    else:
        print("File selection cancelled.")

if __name__ == "__main__":
    select_file()
    input("Press Enter to exit...")
