# UO_Score_Py
Python script to identify and record answers from half sheet bubble forms and the UO Score Answer Form.

## What should I use? Context?
- [UO Score](https://github.com/YuzuRanger/UO_Score) is based on set of R scripts originally developed by Jeremy Piger to read UO Score full-page bubble forms. If you are proficient in R, you may opt to use this.
- [DesktopDeployR_UO-Score](https://github.com/YuzuRanger/DesktopDeployR/tree/ScoreApp) is UO Score but with a user interface and requires no coding knowledge or R proficiency. It can be used to read small batches of UO Score forms, but is resource intensive.
- UO Score Py: You are here! The Python distributions seek to address the following issues:
  - The Desktop Deployment is slow and resources intensive. It crashes when processing more than 100 pages. We needed something more lightweight that can handle large class sizes while still being usable by non-coders.
  - Some faculty use the half-sheet bubble forms, which use different grid markers that the full sheet. We wanted to offer this as well. 

## About the App
### How do I get started?
If you are using half-sheet bubble forms:
- Download the latest Half Sheet Distribution zip file. Extract.
If you are using full-sheet UO Score bubble forms:
- (COMING SOON) Download the latest Full Sheet Distribution zip file. Extract.

### What do I need?
1. Open up the extracted distribution folder.
2. In the *Answer Key* folder, you'll find an example answer key csv. Take note that for the "Points" column, blanks will be read as the default (1 point).
3. Place you scanned pdf in the *Input* folder.
4. Double click on the .exe file to run the program.

### What will the application produce?
Upon running the program, it will begin processing each page of the PDF to generate a list of answers for each student. Then, it will match answers to the answer key.
The script will update you as it progresses and warn of any bubble forms which may need to be manually reviewed. Upon sucessful completion, check the *Output* folder for:
- A csv of detected answers and scores based on the provided answer key.
- A series of png files of each sheet's answers. Detected filled answer bubbles will be labeled by a red box.
  - This allows you to check answers visually as well as verify that the correct region was identified in cases where automatic grid detection failed. 
