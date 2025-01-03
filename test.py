from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter


parserImages = JoradpFileParse("./data_test/F2024080.pdf")
ocr = OcrProcessor()
parserImages.get_images_with_pymupdf()
parserImages.resize_image_to_fit_ocr()
parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
parserImages.adjust_all_images_rotations()
parserImages.parse_images_to_text_structure(ocr)