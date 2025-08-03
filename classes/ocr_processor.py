from PIL import Image
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor

from surya.recognition import RecognitionPredictor
from typing import List



from pytesseract import image_to_osd
import torch

class OcrProcessor:
    def __init__(self):
        """
        Initialize with Surya models and processors for detection and layout.
        """
        # detection & layout init
        self.detection_predictor = DetectionPredictor()
        # layout manager
        self.layout_manager = None
        # recognition init
        self.recognition_manager = None
    
    def load_layout_models(self):
        self.layout_manager = LayoutPredictor()
        

    def load_text_models(self):
        self.recognition_manager = RecognitionPredictor()
        

    def clear_all_models(self):
        # detection & layout init
        if(self.layout_manager is not None):
            del self.layout_manager
            self.layout_manager = None

        if(self.recognition_manager is not None):
            del self.recognition_manager
            self.recognition_manager = None

        torch.cuda.empty_cache()

    def run_layout_order_detection(self, image: Image)-> list:
        """
        returns the list of LayoutBox(
                polygon=[
                    [206.3074891269207, 20.10954011231661],
                    [275.1441529393196, 20.10954011231661],
                    [273.2496216893196, 37.61239942163229],
                    [204.4129578769207, 37.61239942163229],
                ],
                confidence=0.9843224883079529,
                label="SectionHeader",
                position=0,
                },
                bbox=[
                    206.3074891269207,
                    20.10954011231661,
                    275.1441529393196,
                    37.61239942163229,
                ],
            ),
        """

        layout_predictions = self.layout_manager([image])
        return layout_predictions[0].bboxes
    
    def run_ocr_separate_text_recognition_fr(self, image: RecognitionPredictor)-> list:
        """
        returns the list of TextLine(
                polygon=[
                    [505.0, 206.0],
                    [961.0, 206.0],
                    [961.0, 223.0],
                    [505.0, 223.0],
                ],
                confidence=None,
                text=" CARA CONSERVANCE OFF CALL CARRENT PACK TE CORES T X COURS AN A GRANT OF X THARRY OF DAY",
                bbox=[505.0, 206.0, 961.0, 223.0],
                )
        """

        predictions = self.recognition_manager([image], det_predictor=self.detection_predictor)
        
        return predictions[0].text_lines
    
    def run_layout_order_detection_by_images_list(self, images: List[Image.Image])-> list:
        """
        returns the list of LayoutBox(
                polygon=[
                    [206.3074891269207, 20.10954011231661],
                    [275.1441529393196, 20.10954011231661],
                    [273.2496216893196, 37.61239942163229],
                    [204.4129578769207, 37.61239942163229],
                ],
                confidence=0.9843224883079529,
                label="SectionHeader",
                position=0,
                },
                bbox=[
                    206.3074891269207,
                    20.10954011231661,
                    275.1441529393196,
                    37.61239942163229,
                ],
            ),
        """
        layout_predictions = layout_predictions = self.layout_manager(images)
        return [l.bboxes for l in layout_predictions]
        
    def run_ocr_separate_text_recognition_fr_by_images_list(self, images: List[Image.Image])-> list:
        """
        returns the list of TextLine(
                polygon=[
                    [505.0, 206.0],
                    [961.0, 206.0],
                    [961.0, 223.0],
                    [505.0, 223.0],
                ],
                confidence=None,
                text=" CARA CONSERVANCE OFF CALL CARRENT PACK TE CORES T X COURS AN A GRANT OF X THARRY OF DAY",
                bbox=[505.0, 206.0, 961.0, 223.0],
                )
        """

        predictions = self.recognition_manager(images, det_predictor=self.detection_predictor)
        return [l.text_lines for l in predictions]
    
    @staticmethod
    def get_orientation_adjustments(image, debug=False)->int:
        """
        Detect the orientation of a page using Tesseract's OSD
        Args:
            image Image
        Returns:
            int: Degree of needed orientation
        """
        try:

            # Use Tesseract's OSD feature to detect orientation
            osd_data = image_to_osd(image)
            
            # Parse the output for relevant details
            osd_lines = osd_data.split("\n")
            orientation = {
                "angle": int(osd_lines[1].split(":")[-1].strip()),  # Rotation angle
                "script": osd_lines[4].split(":")[-1].strip(),      # Detected script
                "confidence": float(osd_lines[5].split(":")[-1].strip()),  # Confidence
            }
            if (orientation['angle'] != 0) and debug:
                #print("this image needed orientation by " + str(orientation['angle']))
                pass
            return int(orientation['angle'])
        except Exception as e:
            print(f"Error detecting orientation: {e}")
            return 0