from typing import List
from classes.image_builder import ImageBuilder
from classes.ocr_processor import OcrProcessor
from PIL import Image
import fitz 
from concurrent.futures import ThreadPoolExecutor, as_completed

class JoradpFileParse:
    def __init__(self, pdf_path: str):
        """
        Initialize with the path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.images = []
    
    def get_images_with_pymupdf(self, dpi: int = 300) -> list:
        """
        Convert the PDF into a list of images using PyMuPDF.

        Args:
            dpi (int): DPI for image conversion.

        Returns:
            list: List of PIL.Image objects.
        """
        try:
            doc = fitz.open(self.pdf_path)
            images = []
            zoom = dpi / 72  # PyMuPDF works with 72 DPI base
            mat = fitz.Matrix(zoom, zoom)

            for page_number in range(len(doc)):
                page = doc[page_number]
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)

            self.images = images
            return self.images
        except Exception as e:
            print(f"An error occurred while converting PDF to images using PyMuPDF: {e}")
            return []

    def resize_image_to_fit_ocr(self) -> list:
        """
        Resize images to have height 1400px, and scale width proportionally
        Returns:
            list: List of modified PIL.Image objects.
        """
        resized_images = []
        for img in self.images:
            width, height = img.size
            aspect_ratio = width / height
            new_height = 1500
            new_width = int(aspect_ratio * new_height)
            resized_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_image)

        self.images =  resized_images

    def crop_all_images(self, top=0, left=0, right=0, bottom=0):
        """
        Crop the specified number of pixels from the imported images.

        Args:
            image (PIL.Image.Image): The image to crop.
            top (int): Number of pixels to crop from the top.
            left (int): Number of pixels to crop from the left.
            right (int): Number of pixels to crop from the right.
            bottom (int): Number of pixels to crop from the bottom.

        Returns:
            PIL.Image.Image: The cropped image.
        """
        resized_images = []
        for img in self.images:
            image_cropped = self.crop(img, top=top, left=left, right=right, bottom=bottom)
            resized_images.append(image_cropped)

        self.images =  resized_images

    def adjust_all_images_rotations(self):
        """
        Crop the specified number of pixels from the imported images.

        Args:
            image (PIL.Image.Image): The image to crop.
            top (int): Number of pixels to crop from the top.
            left (int): Number of pixels to crop from the left.
            right (int): Number of pixels to crop from the right.
            bottom (int): Number of pixels to crop from the bottom.

        Returns:
            PIL.Image.Image: The cropped image.
        """
        resized_images = []
        for img in self.images:
            rotation = OcrProcessor.get_orientation_adjustments(img)
            resized_images.append(self.rotate_image_degrees_auto(img, rotation))

        self.images =  resized_images
    
    @staticmethod
    def detect_one_image_rotation(image):
        rotation = OcrProcessor.get_orientation_adjustments(image)
        return JoradpFileParse.rotate_image_degrees_auto(image, rotation)
    
    def adjust_all_images_rotations_parallel(self, max_workers=None):
        """
        Adjust the orientation of all images in parallel using ProcessPoolExecutor.
        This maintains the same functionality as the sequential version but processes
        images concurrently for better performance.
        
        Args:
            ocr: OcrProcessor instance
            max_workers: Number of worker processes to use (defaults to number of CPU cores)
        """
        # Create a partial function with the ocr parameter fixed
        
        # Use ProcessPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process all images in parallel while maintaining order
            self.images = list(executor.map(JoradpFileParse.detect_one_image_rotation, self.images))
    
    def parse_images_to_text_structure(self, ocr: OcrProcessor):
        usedImages = self.images[1:]
        layout_group_result = []
        text_group_result = []
        
        ocr.load_layout_models()

        for img in usedImages:
            layouts = ocr.run_layout_order_detection(img)
            layout_group_result.append(layouts)

        ocr.clear_all_models()
        ocr.load_text_models()

        for img in usedImages:
            detected_textes = ocr.run_ocr_separate_text_recognition_fr(img)
            text_group_result.append(detected_textes)
            
        ocr.clear_all_models()
        result_ocr = []
        page = 1

        for index in range(0, len(text_group_result)):
            layouts = layout_group_result[index]
            detected_textes = text_group_result[index]
            # add image if debugging is needed
            imageTest = ImageBuilder(image=None, layout_data=layouts, text_data=detected_textes)
            # 10 is best margin through tests
            result = imageTest.match_making_texts_to_layouts(margin=10)
            result_ocr.append({ 'index': page, 'page': result['result'], 'rest': result['rest']})
            page += 1

        return result_ocr
    

    def parse_images_to_text_structure_optimized(self, ocr: OcrProcessor):
        usedImages = self.images[1:]

        ocr.load_layout_models()
        layout_group_result = ocr.run_layout_order_detection_by_images_list(usedImages)
        ocr.clear_all_models()
        ocr.load_text_models()
        text_group_result = ocr.run_ocr_separate_text_recognition_fr(usedImages)  
        ocr.clear_all_models()
        
        result_ocr = []
        page = 1

        for index in range(0, len(text_group_result)):
            layouts = layout_group_result[index]
            detected_textes = text_group_result[index]
            # add image if debugging is needed
            imageTest = ImageBuilder(image=None, layout_data=layouts, text_data=detected_textes)
            # 10 is best margin through tests
            result = imageTest.match_making_texts_to_layouts(margin=10)
            result_ocr.append({ 'index': page, 'page': result['result'], 'rest': result['rest']})
            page += 1
        
        # it 
        return result_ocr
    
    def parse_images_to_text_structure_selective(self, ocr: OcrProcessor, page_indices: list):
        """
        Parse only specific pages to text structure using OCR.
        Pages not in the indices list will have empty results.

        Args:
            ocr (OcrProcessor): The OCR processor instance.
            page_indices (list): List of page indices to process (0-based, excluding first page).
                                For example, [0, 2, 4] will process pages 2, 4, and 6 of the PDF
                                (since we skip the first page).

        Returns:
            list: List of OCR results with empty results for ignored pages.
        """
        total_pages = len(self.images)
        
        if (len(page_indices) == 0):
            return []
        
        # Validate page indices
        valid_indices = [i for i in page_indices if 0 <= i < total_pages]
        if len(valid_indices) != len(page_indices):
            invalid_indices = [i for i in page_indices if i < 0 or i >= total_pages]
            print(f"Warning: Invalid page indices ignored: {invalid_indices}")
        
        # Create lists to store results for all pages
        layout_group_result = [] 
        text_group_result = [] 
        
        # Process only selected pages
        if valid_indices:
            # Get images for selected pages only
            selected_images = [self.images[i] for i in valid_indices]
            
            layout_group_result = []
            text_group_result = []
            
            ocr.load_layout_models()

            
            layout_group_result = ocr.run_layout_order_detection_by_images_list(selected_images)

            ocr.clear_all_models()
            ocr.load_text_models()

            text_group_result = ocr.run_ocr_separate_text_recognition_fr_by_images_list(selected_images)
                
            ocr.clear_all_models()
            
        
        # Build final results
        result_ocr = []
        
        for index in range(len(valid_indices)):
            page_index = valid_indices[index]
            
            layouts = layout_group_result[index]
            detected_textes = text_group_result[index]
            
            imageTest = ImageBuilder(image=None, layout_data=layouts, text_data=detected_textes)
            result = imageTest.match_making_texts_to_layouts(margin=10)
            result_ocr.append({
                'index': page_index, 
                'page': result['result'], 
                'rest': result['rest']
            })
            
        
        return result_ocr
    
    ## layout coords
    def compute_images_table_data_selective(self, ocr: OcrProcessor, page_indices: list, table_boxes: List[List[List[float]]], manage_ocr):
        """
        Parse only specific pages to text structure using OCR.
        Pages not in the indices list will have empty results.

        Args:
            ocr (OcrProcessor): The OCR processor instance.
            page_indices (list): List of page indices to process (0-based, excluding first page).
                                For example, [0, 2, 4] will process pages 2, 4, and 6 of the PDF
                                (since we skip the first page).

        Returns:
            list: List of OCR results with empty results for ignored pages.
        """
        total_pages = len(self.images)
        
        if (len(page_indices) == 0):
            return []
        
        if (len(page_indices) != len(table_boxes)):
            raise ValueError("layout not identified with targets")
        
        # Validate page indices
        valid_indices = [i for i in page_indices if 0 <= i < total_pages]
        if len(valid_indices) != len(page_indices):
            invalid_indices = [i for i in page_indices if i < 0 or i >= total_pages]
            print(f"Warning: Invalid page indices ignored: {invalid_indices}")
        
        
        # Process only selected pages
        if valid_indices:
            # Get images for selected pages only
            selected_images = [self.images[i] for i in valid_indices]
            
            if manage_ocr:
                ocr.load_table_models()
            
            table_data_all_images = []
            print("gloabl bboxes : " + str(table_boxes))
            print(type(table_boxes))
            
            for image, index, bboxes in zip(selected_images, valid_indices, table_boxes):
                page_data = []
                # table_coords = ocr.localize_tables_in_image(image)
                # returns a list of tables coord in the page
                # if it is empty then consider the whole page as a table
                print("index : " + str(index))
                print("bboxes : " + str(bboxes))
                print(type(bboxes))
                if len(bboxes) == 0:
                    tmp_data = ocr.extract_selected_table_cells(image, True, [])
                    page_data.append({'table': [0, 0, image.width, image.height], 'table_data': tmp_data})
                else:
                    for table_coord in bboxes:
                        tmp_data = ocr.extract_selected_table_cells(image, False, table_coord['bbox'])
                        page_data.append({'table': table_coord['bbox'], 'table_data': tmp_data})
                table_data_all_images.append({'index': index, 'page_data': page_data})

            if manage_ocr:
                ocr.clear_all_models()
                
        
        return table_data_all_images

    def parse_images_to_text_structure_selective_heavy(self, ocr: OcrProcessor, page_indices: list):
        """
        Parse only specific pages to text structure using OCR.
        Pages not in the indices list will have empty results.

        Args:
            ocr (OcrProcessor): The OCR processor instance.
            page_indices (list): List of page indices to process (0-based, excluding first page).
                                For example, [0, 2, 4] will process pages 2, 4, and 6 of the PDF
                                (since we skip the first page).

        Returns:
            list: List of OCR results with empty results for ignored pages.
        """
        total_pages = len(self.images)
        
        if (len(page_indices) == 0):
            return []
        
        # Validate page indices
        valid_indices = [i for i in page_indices if 0 <= i < total_pages]
        if len(valid_indices) != len(page_indices):
            invalid_indices = [i for i in page_indices if i < 0 or i >= total_pages]
            print(f"Warning: Invalid page indices ignored: {invalid_indices}")
        
        # Create lists to store results for all pages
        layout_group_result = [] 
        text_group_result = [] 
        
        # Process only selected pages
        if valid_indices:
            # Get images for selected pages only
            selected_images = [self.images[i] for i in valid_indices]
            
            layout_group_result = []
            text_group_result = []
            

            
            layout_group_result = ocr.run_layout_order_detection_by_images_list(selected_images)


            text_group_result = ocr.run_ocr_separate_text_recognition_fr_by_images_list(selected_images)

            
        
        # Build final results
        result_ocr = []
        
        for index in range(len(valid_indices)):
            page_index = valid_indices[index]
            
            layouts = layout_group_result[index]
            detected_textes = text_group_result[index]
            
            imageTest = ImageBuilder(image=None, layout_data=layouts, text_data=detected_textes)
            result = imageTest.match_making_texts_to_layouts(margin=10)
            result_ocr.append({
                'index': page_index, 
                'page': result['result'], 
                'rest': result['rest']
            })
            
        
        return result_ocr
    
    @staticmethod
    def crop(image, top=0, left=0, right=0, bottom=0)-> Image:
        """
        Crop the specified number of pixels from the edges of an image.

        Args:
            image (PIL.Image.Image): The image to crop.
            top (int): Number of pixels to crop from the top.
            left (int): Number of pixels to crop from the left.
            right (int): Number of pixels to crop from the right.
            bottom (int): Number of pixels to crop from the bottom.

        Returns:
            PIL.Image.Image: The cropped image.
        """
        width, height = image.size
        # Validate the crop dimensions
        if top + bottom >= height or left + right >= width:
            raise ValueError("The crop dimensions exceed the image size.")
        # Define the cropping box (left, upper, right, lower)
        crop_box = (left, top, width - right, height - bottom)
        # Crop the image
        cropped_image = image.crop(crop_box)
        return cropped_image
    

    
    @staticmethod
    def rotate_image_degrees_auto(image: Image, degrees: int) -> Image:
        """
        Rotate an image by 90 degrees while adjusting the height and width.

        Args:
            image (PIL.Image.Image): The image to rotate.
            clockwise (bool): Rotate clockwise if True, otherwise counterclockwise.

        Returns:
            PIL.Image.Image: The rotated image.
        """
        if (degrees == 0):
            return image
        # Rotate the image
        if (degrees == 270):
            return  image.transpose(Image.Transpose.ROTATE_270)
        elif (degrees == 90):
            return image.transpose(Image.Transpose.ROTATE_90)
        else:
            print(" Wrong rotation asked for " + str(degrees))
        
        return image