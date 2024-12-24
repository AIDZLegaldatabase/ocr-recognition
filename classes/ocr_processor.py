from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection

from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor


from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

from surya.ocr import run_recognition
from surya.ocr import run_ocr

from pytesseract import image_to_osd

class OcrProcessor:
    def __init__(self):
        """
        Initialize with Surya models and processors for detection and layout.
        """
        # detection & layout init
        self.layout_model = load_layout_model()
        self.layout_processor = load_layout_processor()
        self.detection_model = load_det_model()
        self.detection_processor = load_det_processor()

        # recognition init
        self.recognition_model = load_rec_model()
        self.recognition_processor = load_rec_processor()

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
        line_predictions = batch_text_detection([image], self.detection_model, self.detection_processor)
        layout_predictions = batch_layout_detection([image], self.layout_model, self.layout_processor, line_predictions)
        return layout_predictions[0].bboxes

    def run_text_recognition_fr(self, image: Image, layout_data: list)-> list:
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
        #parse to int
        layout_polygons = [[[int(coord[0]), int(coord[1])] for coord in layout_box.polygon] for layout_box in layout_data]

        layout_recognition_results = run_recognition(
            images=[image],               # Input image
            langs=[['fr']],                   # Languages
            rec_model=self.recognition_model,           # Recognition model
            rec_processor=self.recognition_processor,   # Recognition processor
            polygons=[layout_polygons])
        return layout_recognition_results[0].text_lines
        
    def run_ocr_separate_text_recognition_fr(self, image: Image)-> list:
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
        predictions = run_ocr([image], [['fr']], self.detection_model, self.detection_processor, self.recognition_model, self.recognition_processor)
        return predictions[0].text_lines
    
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
