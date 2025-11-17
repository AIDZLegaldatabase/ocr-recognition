import numpy as np
import cv2
from table_detection import detect_table_from_image_data
import argparse
from classes.pdf_parser import JoradpFileParse
from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser(prog="Table detection tester")

    parser.add_argument("-d", "--debug_mode", action="store_true")
    parser.add_argument("-input_pdf_file", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    parserImages = JoradpFileParse(args.input_pdf_file)
    parserImages.get_images_with_pymupdf()
    parserImages.resize_image_to_fit_ocr()

    print(f"Parsed {len(parserImages.images)} images from the PDF file")

    inut_data_path = Path(args.input_pdf_file)

    if inut_data_path.name.startswith("F1978"):
        # 1978
        parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)

    elif inut_data_path.name.startswith("F2024"):
        # 2024
        parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
        parserImages.adjust_all_images_rotations_parallel()

    window_name = ""
    if args.debug_mode:
        # Create a resizable window in advance
        window_name = "Table Detection Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- Analyzing Extracted Pages ---")

    for i, pil_page_image in enumerate(parserImages.images):

        if i == 0:
            continue
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
