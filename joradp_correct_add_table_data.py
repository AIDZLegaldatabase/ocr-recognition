import ast
import os
import json
import csv
from collections import defaultdict
from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter
import glob


class CropConfigurationManager:
    """Manages crop configurations for different year intervals"""
    
    _CROP_CONFIGURATIONS = [
        # checked
        {
            'start': 'F1962000',
            'end': 'F1997199',
            'crop_params': {
                'top': 85, 
                'left': 0, 
                'right': 0, 
                'bottom': 15
            }
        },
        #checked
        {
            'start': 'F1998000',
            'end': 'F2001199',
            'crop_params': {
                'top': 100, 
                'left': 40, 
                'right': 30, 
                'bottom': 40
            }
        },
        # checked
        {
            'start': 'F2002000',
            'end': 'F2005069',
            'crop_params': {
                'top': 120, 
                'left': 70, 
                'right': 70, 
                'bottom': 20
            }
        },
        # checked
        {
            'start': 'F2005070',
            'end': 'F2005078',
            'crop_params': {
                'top': 180, 
                'left': 130, 
                'right': 130, 
                'bottom': 150
            }
        },
        # checked
        {
            'start': 'F2005079',
            'end': 'F2018003',
            'crop_params': {
                'top': 140, 
                'left': 45, 
                'right': 100, 
                'bottom': 60
            }
        },
        # checked
        {
            'start': 'F2018004',
            'end': 'F2025199',
            'crop_params': {
                'top': 110, 
                'left': 70, 
                'right': 70, 
                'bottom': 90
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


def load_selective_processing_config(csv_path='report_tables.csv'):

    selective_config = defaultdict(list)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                document_name = row['document_name'].replace('.json', '')  # Remove .json extension
                page_number = int(row['page_number']) 
                full_page = row['full_page'].lower() == "true"
                table_boxes = ast.literal_eval(row['table_boxes'])
                
                # Only add if page_number >= 0 (since we subtract 1 and skip first page)
                if page_number >= 0:
                    if (full_page and (len(table_boxes)>0)):
                        raise TypeError("should not have unsynck like that")
                    selective_config[document_name].append({'page_number': page_number, 'full_page': full_page, 'table_boxes': table_boxes })
                
        # Sort page numbers for each document
        for doc_name in selective_config:
            selective_config[doc_name].sort(key=lambda x: x["page_number"])
            
        print(f"Loaded selective processing config for {len(selective_config)} documents")
        ###
        print(selective_config)
        return dict(selective_config)
        
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Processing all documents normally.")
        return {}
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}


def get_joradp_files_list(year: int):
    directory_path = "./year_files_raw/" + str(year)
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_latest_processed_file(year):
    """Get the latest processed file name for a given year"""
    year_path = f"./result_json_tables/{year}"
    if not os.path.exists(year_path):
        return None
    
    # Get all JSON files in the year directory
    json_files = glob.glob(os.path.join(year_path, "*.json"))
    if not json_files:
        return None
    
    # Extract base names and find the latest
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    return max(base_names) if base_names else None

def run_table_recognition_by_year_selective(year: int, selective_config: dict, min_filename=None):
    """
    Run OCR processing for a specific year, but only process documents and pages specified in selective_config.
    
    Args:
        year (int): Year to process
        selective_config (dict): Dictionary mapping document names to lists of dict containing
            - page_number (int)
            - full_page (bool)
            - table_boxes (list)
        min_filename (str): Minimum filename to start processing from
    """
    ocr = OcrProcessor()
    ocr.load_table_models()
            
    target_pdf_files = get_joradp_files_list(year)
    os.makedirs("./result_json_tables/"+ str(year), exist_ok=True)  # Create a folder for JSON files

    latest_processed = get_latest_processed_file(year)

    for file_path in sorted(target_pdf_files):
        # Get current file base name
        current_file_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Skip if file is before minimum filename
        if min_filename and current_file_base < min_filename:
            continue
            
        # Skip if already processed
        if latest_processed and current_file_base <= latest_processed:
            continue
        
        # Check if this document is in our selective config
        if current_file_base not in selective_config:
            print(f"Skipping {current_file_base} - not in selective processing list")
            continue
        
        # Get the page indices to process for this document
        page_indices = [ i['page_number'] for i in selective_config[current_file_base] ]
        page_table_boxes = [ i['table_boxes'] for i in selective_config[current_file_base] ]
        
        print(f"Processing {current_file_base} - Pages: {[p+1 for p in page_indices]} (1-based)")
        
        # saving
        new_result_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"

        # selecting the right margin by year and version
        parserImages = JoradpFileParse(file_path)

        parserImages.get_images_with_pymupdf()
        parserImages.resize_image_to_fit_ocr()

        crop_config = CropConfigurationManager.get_crop_configuration(current_file_base)
        parserImages.crop_all_images(
            top=crop_config['top'], 
            left=crop_config['left'], 
            right=crop_config['right'], 
            bottom=crop_config['bottom']
        )
        print(f"Crop config: {crop_config}")
        parserImages.adjust_all_images_rotations_parallel()
        
        # Use selective processing with the specified page indices
        data = parserImages.compute_images_table_data_selective(ocr, page_indices, page_table_boxes, False)

        with open('./result_json_tables/'+ str(year) +'/'+ new_result_name, 'w') as convert_file:
            convert_file.write(json.dumps(data))
        
        print(f"Completed table processing {current_file_base}")
    
    ocr.clear_all_models()


# Load the selective processing configuration from CSV
selective_config = load_selective_processing_config('report_tables.csv')

if not selective_config:
    print("Counld not extact csv")
    exit()
    
# Process each year that has documents in the selective config
years_to_process = set()
for doc_name in selective_config.keys():
    # Extract year from document name (assuming format like F1962001)
    if doc_name.startswith('F') and len(doc_name) >= 8:
        print(doc_name)
        try:
            year = int(doc_name[1:5])  # Extract year from F1962001
            years_to_process.add(year)
        except ValueError:
            print(f"Could not extract year from document name: {doc_name}")

# Process each year with selective config
for year in sorted(years_to_process):
    print(f"Processing year {year} with selective configuration...")
    run_table_recognition_by_year_selective(year, selective_config)
    print(f"Completed selective table processing for {year}")
      