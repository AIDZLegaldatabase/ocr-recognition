
import os
import json
from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter


def get_joradp_files_list(year: int):
    directory_path = "./year_files_raw/" + str(year)
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def run_ocr_by_year(year: int):
    margin_closed_configurations = [
        {
            "range": ("F2005075", "F2005075"),  # Single file range
            "margins": {"top": 200, "left": 150, "right": 150, "bottom": 160},
        },
        {
            "range": ("F2006001", "F2006100"),  # First neighbor range
            "margins": {"top": 0, "left": 25, "right": 25, "bottom": 25},
        },
        {
            "range": ("F2006101", "F2006200"),  # Second neighbor range
            "margins": {"top": 120, "left": 90, "right": 90, "bottom": 130},
        },
        # Add more configurations here if needed
    ]

    target_pdf_files = get_joradp_files_list(year)
    os.makedirs("./result_json/"+ str(year), exist_ok=True)  # Create a folder for JSON files

    for file_path in target_pdf_files:
        #saving
        new_result_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
        # Default margins
        margins = {"top": 120, "left": 80, "right": 80, "bottom": 100}
        # search for right margin 
        for config in margin_closed_configurations:
            range_start, range_end = config["range"]
            if range_start <= new_result_name.split(".")[0] <= range_end:
                margins = config["margins"]
                break
        
        print(margins)

        # selecting the right margin by year and version
        #parserImages = JoradpFileParse(file_path)
        #ocr = OcrProcessor()
        #parserImages.get_images_with_pymupdf()
        #parserImages.resize_image_to_fit_ocr()
        #parserImages.crop_all_images(**margins)
        #parserImages.adjust_all_images_rotations_parallel()
        #data = parserImages.parse_images_to_text_structure(ocr)

        
        #with open('./result_json/'+ str(year) +'/'+ new_result_name, 'w') as convert_file:
            #convert_file.write(json.dumps(data))


for i in range(2005, 2007):
    run_ocr_by_year(i)
    print(i)