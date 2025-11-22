import numpy as np
import cv2
from table_detection import detect_table_from_image_data
import argparse
from classes.pdf_parser import JoradpFileParse
from pathlib import Path
import re

# cases 1 python .\test_tables.py -d -input_pdf_file ./data_test/F1962008.pdf
# cases 2 python .\test_tables.py -d -input_pdf_file ./data_test/F2024007.pdf
# cases 3 python .\test_tables.py -d -input_pdf_file ./data_test/F2025009.pdf

def args_parser():
    parser = argparse.ArgumentParser(prog="Table detection tester")

    parser.add_argument("-d", "--debug_mode", action="store_true")
    parser.add_argument("-input_pdf_file", type=str, default="")
    parser.add_argument("-input_pdf_dir", type=str, default="")

    return parser.parse_args()

def get_formatted_pdfs(folder_path):
    """
    Scans a folder for PDF files matching the pattern Fxxxxxxx.pdf
    (Where 'x' represents exactly 7 digits).
    """
    folder = Path(folder_path)
    
    # Regex explanation:
    # ^      : Start of string
    # F      : Literal character 'F'
    # \d{7}  : Exactly 7 digits (0-9)
    # \.pdf$ : Ends with literal .pdf
    pattern = re.compile(r"^F\d{7}\.pdf$")
    
    matching_files = []
    
    # robust check to ensure folder exists
    if not folder.exists():
        return []

    for file_path in folder.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            # Returns the full path string. Change to file_path.name if you just want filenames.
            matching_files.append(str(file_path))
        
    return matching_files

def parse_images(pdf_path):
    parserImages = JoradpFileParse(pdf_path)
    
    
    parserImages.get_images_with_pymupdf()
    parserImages.resize_image_to_fit_ocr()

    print(f"Parsed {len(parserImages.images)} images from the PDF file")

    inut_data_path = Path(pdf_path)

    if inut_data_path.name.startswith("F1978"):
        # 1978
        parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)
    elif inut_data_path.name.startswith("F1962"):
        # 1978
        parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)
    elif inut_data_path.name.startswith("F2025"):
        parserImages.crop_all_images(top=110, left=70, right=70, bottom=90)
        # parserImages.adjust_all_images_rotations_parallel()

    elif inut_data_path.name.startswith("F2024"):
        # 2024
        parserImages.crop_all_images(top=110, left=70, right=70, bottom=90)
        parserImages.adjust_all_images_rotations_parallel()
    else:
        raise ValueError("There was no crop with this pdf file, are you sure about that?")
    return parserImages.images[1:]
    

if __name__ == "__main__":
    args = args_parser()
    image_to_process = []

    if args.input_pdf_dir != "":
        pdf_files_list = get_formatted_pdfs(args.input_pdf_dir)
        for pdf_path in pdf_files_list:
            image_to_process.extend(parse_images(pdf_path))
    elif args.input_pdf_file != "":
        image_to_process = parse_images(args.input_pdf_file)

    window_name = ""
    if args.debug_mode:
        # Create a resizable window in advance
        window_name = "Table Detection Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- Analyzing Extracted Pages ---")

    for i, pil_page_image in enumerate(image_to_process):
        # 1. Convert PIL Image (assuming RGB) to OpenCV format (BGR)
        page_image_rgb = np.array(pil_page_image)

        # 2. detect tables from image
        table_boxes, img_grid = detect_table_from_image_data(page_image_rgb)

        # Create a copy of the image to draw on
        debug_image = page_image_rgb
        if table_boxes:
            # 3. Draw the boxes
            for x, y, w, h in table_boxes:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                print(str(((x, y), (x + w, y + h))))

            print(f"--- Page {i+1}: Table(s) detected! ---")

        else:
            print(f"--- Page {i+1}: No tables found. ---")

        if args.debug_mode:

            combined_display = cv2.hconcat(
                [cv2.cvtColor(img_grid, cv2.COLOR_GRAY2BGR), debug_image]
            )
            # 6. Show the combined image and wait
            cv2.imshow(window_name, combined_display)
            # Wait indefinitely for a key press
            key = cv2.waitKey(0)

            # 7. Add a quit condition
            if key == ord("q"):
                print("Quitting loop...")
                break
    # 8. Clean up windows after the loop
    cv2.destroyAllWindows()
