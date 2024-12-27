
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
    target_pdf_files = get_joradp_files_list(year)
    os.makedirs("./result_json/"+ str(year), exist_ok=True)  # Create a folder for JSON files

    for file_path in target_pdf_files:
        parserImages = JoradpFileParse(file_path)
        ocr = OcrProcessor()
        parserImages.get_images_with_pymupdf()
        parserImages.resize_image_to_fit_ocr()
        parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
        parserImages.adjust_all_images_rotations_parallel()
        data = parserImages.parse_images_to_text_structure(ocr)

        #saving
        new_result_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
        with open('./result_json/'+ str(year) +'/'+ new_result_name, 'w') as convert_file:
            convert_file.write(json.dumps(data))


for i in range(2020, 2025):
    run_ocr_by_year(i)
    print(i)