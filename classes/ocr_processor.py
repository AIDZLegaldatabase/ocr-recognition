from PIL import Image
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor

from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
from torchvision import transforms

from surya.recognition import RecognitionPredictor
from typing import List, Union



from pytesseract import image_to_osd
import torch

from classes.image_builder import ImageBuilder

class OcrProcessor:
    def __init__(self):
        """
        Initialize with Surya models and processors for detection and layout.
        """
        # detection & layout init
        self.detection_recognition_manager = None
        # layout manager
        self.layout_manager = None
        # recognition init
        self.recognition_manager = None
        # table model
        self.table_detection_manager = None
        self.table_recognition_manager = None
        
        # device used
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_layout_models(self):
        self.layout_manager = LayoutPredictor()
        

    def load_text_models(self):
        self.recognition_manager = RecognitionPredictor()
        self.detection_recognition_manager = DetectionPredictor()
        
    def load_table_models(self):
        self.table_detection_manager = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
        self.table_detection_manager.to(self.device)
        self.table_recognition_manager = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
        self.table_recognition_manager.to(self.device)
        
        

    def clear_all_models(self):
        # detection & layout init
        if(self.layout_manager is not None):
            del self.layout_manager
            self.layout_manager = None

        if(self.recognition_manager is not None):
            del self.recognition_manager
            self.recognition_manager = None
            
        if(self.detection_recognition_manager is not None):
            del self.detection_recognition_manager
            self.detection_recognition_manager = None
        
        if (self.table_recognition_manager is not None):
            del self.table_recognition_manager
            self.table_recognition_manager = None
        
        if (self.table_detection_manager is not None):
            del self.table_detection_manager
            self.table_detection_manager = None

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

        predictions = self.recognition_manager([image], det_predictor=self.detection_recognition_manager)
        
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
        layout_predictions = self.layout_manager(images)
        return [l.bboxes for l in layout_predictions]

    def localize_tables_in_image(self, image: Image.Image)-> list:
        """
        @note: not 100% accurate when the table in 100% page size
        @note: auto adds a padding of 25px
        returns [{'label': 'table',
            'score': 0.9782191514968872,
            'bbox': [95.17469787597656, 1120.81396484375, 963.0751342773438, 1368.6875]}]
        """
        # image to rgb 
        image_rgb = image.convert("RGB")
        # prepare image to tensor
        detection_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = detection_transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)   
        image_pixel_values = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.table_detection_manager(image_pixel_values)

        # post processing
        id2label = self.table_detection_manager.config.id2label
        id2label[len(self.table_detection_manager.config.id2label)] = "no object"
        
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in OcrProcessor.rescale_bboxes(pred_bboxes, image_rgb.size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if class_label != "no object":
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})
                
        ### 
        padding = 25

        return [ {'label': i['label'], 'score': i['score'], 
                    'bbox': [ max(0, i['bbox'][0] - padding), max(0, i['bbox'][1] - padding),
                              min(image.width, i['bbox'][2] + padding), min(image.height, i['bbox'][3] + padding) ] }
                    for i in objects]
    
    def extract_selected_table_cells(self, image: Image.Image, full_image: bool, coords: List[Union[int, float]])-> list:
        """
        @note: not 100% accurate when the table in 100% page size
        @note labels can be: 'table column', 'table spanning cell', 'table column header' ...
        returns [{'label': 'table column',
                'score': 0.9999761581420898,
                'bbox': [210.84799194335938,
                1.7172718048095703,
                321.04205322265625,
                521.047607421875]}]
                

        """
        target_coord = [0,0, image.width, image.height]
        if not full_image:
            target_coord = coords
            if (len(target_coord)) < 4:
                raise("Too few coordinates")
        
        target_image = ImageBuilder.select_inner_image(image, target_coord)
            
        # image to rgb 
        image_rgb = target_image.convert("RGB")
        # prepare image to tensor
        detection_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = detection_transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)   
        image_pixel_values = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.table_recognition_manager(image_pixel_values)

        # post processing
        id2label = self.table_recognition_manager.config.id2label
        id2label[len(self.table_recognition_manager.config.id2label)] = "no object"
        
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in OcrProcessor.rescale_bboxes(pred_bboxes, image_rgb.size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if class_label != "no object":
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return [ {'label': i['label'], 'score': i['score'], 
                    'bbox': [i['bbox'][0] + target_coord[0], i['bbox'][1] + target_coord[1], i['bbox'][2] + target_coord[0], i['bbox'][3] + target_coord[1] ] }
                    for i in objects]
        
        
        
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

        predictions = self.recognition_manager(images, det_predictor=self.detection_recognition_manager)
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
        
    @staticmethod
    def box_center_width_to_coord(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = OcrProcessor.box_center_width_to_coord(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b