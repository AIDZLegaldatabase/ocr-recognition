
import os
import json
from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter
import glob


class CropConfigurationManager:
    """Manages crop configurations for different year intervals"""
    
    _CROP_CONFIGURATIONS = [
        {
            'start': 'F1962000',
            'end': 'F1980004',
            'crop_params': {
                'top': 70, 
                'left': 0, 
                'right': 0, 
                'bottom': 30
            }
        },
        {
            'start': 'F1980005',
            'end': 'F1995004',
            'crop_params': {
                'top': 80, 
                'left': 10, 
                'right': 10, 
                'bottom': 10
            }
        },
        {
            'start': 'F1995005',
            'end': 'F2005004',
            'crop_params': {
                'top': 90, 
                'left': 20, 
                'right': 20, 
                'bottom': 20
            }
        },
        # Add more intervals as needed
    ]
    
    @classmethod
    def get_crop_configuration(cls, filename):
        for config in cls._CROP_CONFIGURATIONS:
            if config['start'] <= filename <= config['end']:
                return config['crop_params']
        
        raise ValueError(f"No crop configuration found for filename: {filename}")


def get_joradp_files_list(year: int):
    directory_path = "./year_files_raw/" + str(year)
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_latest_processed_file(year):
    """Get the latest processed file name for a given year"""
    year_path = f"./result_json/{year}"
    if not os.path.exists(year_path):
        return None
    
    # Get all JSON files in the year directory
    json_files = glob.glob(os.path.join(year_path, "*.json"))
    if not json_files:
        return None
    
    # Extract base names and find the latest
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    return max(base_names) if base_names else None

def run_ocr_by_year(year: int):


    target_pdf_files = get_joradp_files_list(year)
    os.makedirs("./result_json/"+ str(year), exist_ok=True)  # Create a folder for JSON files

    latest_processed = get_latest_processed_file(year)

    for file_path in target_pdf_files:
        #saving
        new_result_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"

        # check the filename current is bigger than what has been computed
        current_file_base = os.path.splitext(os.path.basename(file_path))[0]
        if latest_processed and current_file_base <= latest_processed:
            continue


        # selecting the right margin by year and version
        parserImages = JoradpFileParse(file_path)
        ocr = OcrProcessor()
        parserImages.get_images_with_pymupdf()
        parserImages.resize_image_to_fit_ocr()

        crop_config = CropConfigurationManager.get_crop_configuration(current_file_base)
        parserImages.crop_all_images(
            top=crop_config['top'], 
            left=crop_config['left'], 
            right=crop_config['right'], 
            bottom=crop_config['bottom']
        )
        print(" Current file " + current_file_base)
        print(crop_config)
        parserImages.adjust_all_images_rotations_parallel()
        data = parserImages.parse_images_to_text_structure(ocr)

        
        with open('./result_json/'+ str(year) +'/'+ new_result_name, 'w') as convert_file:
            convert_file.write(json.dumps(data))


for i in range(1962, 1992):
    run_ocr_by_year(i)
    print(i)