from PIL import Image, ImageDraw, ImageFont

class ImageBuilder:
    def __init__(self, image=None, layout_data=None, text_data=None):
        """
        Initialize the ImageBuilder with image, layout data, and text data.

        Args:
            - image (PIL.Image.Image, optional): The raw image.
            - layout_data (list, optional): List of LayoutBox(
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
            - text_data (list, optional): List of text recognition results.
        """
        self.raw_image = image
        self.layout_data = layout_data
        self.text_data = text_data


    def draw_order_layout_on_image(self):
        """
        Draw layout bounding boxes on the image with labels and positions.

        Raises:
            ValueError: If raw_image or layout_data is not set.
        """
        if self.raw_image is None or self.layout_data is None:
            raise ValueError("Data of layout or raw image is not properly set.")

        return self._draw_layout_boxes(self.raw_image, self.layout_data)
    
    def draw_text_on_image(self):
        """
        Draw layout bounding boxes on the image with labels and positions.

        Raises:
            ValueError: If raw_image or layout_data is not set.
        """
        if self.raw_image is None or self.text_data is None:
            raise ValueError("Data of layout or raw image is not properly set.")

        return self._draw_layout_boxes(self.raw_image, self.text_data)

    @staticmethod
    def _draw_layout_boxes(image, layout_results):
        """
        Draw layout bounding boxes on the image with labels and positions.

        Args:
            image (PIL.Image.Image): The image to draw on.
            layout_results (list): List of layout results, where each result has
                                   attributes polygon, label, position, and confidence.

        Returns:
            PIL.Image.Image: The annotated image.
        """
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Font for text (optional, adjust path to a font file if needed)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Iterate through each layout box
        for box in layout_results:
            # Extract polygon and label
            polygon = [(int(x), int(y)) for x, y in box.polygon]  # Convert to integers
            label = box.label
            position = box.position
            confidence = box.confidence

            # Draw polygon
            draw.polygon(polygon, outline="red", width=2)

            # Annotate with label and position
            label_text = f"{label} (Pos: {position}, Conf: {confidence:.2f})"
            text_position = (polygon[0][0], polygon[0][1] - 10)  # Above the box
            draw.text(text_position, label_text, fill="blue", font=font)

        return annotated_image

    def display_image(self):
        """
        Display the raw image for quick debugging.
        """
        if self.raw_image is None:
            raise ValueError("Raw image is not set.")
        self.raw_image.show()

    def match_making_texts_to_layouts(self, margin=5):
        """
        Match text boxes to layout boxes based on adjusted bounding boxes, optimized for one-to-one mapping.

        Args:
            layout_data (list): List of LayoutBox objects containing layout information.
            texts_data (list): List of TextLine objects containing text information.
            margin (int, optional): Margin to adjust bounding boxes for matching. Defaults to 5.

        Returns:
            list: A list of dictionaries with matched layout and text information.
        """
        matched_results = []

        # Iterate over layout data
        for layout in self.layout_data:
            # Adjust layout bbox with margin
            layout_bbox = [
                layout.bbox[0] - margin,  # x_min - margin
                layout.bbox[1] - margin,  # y_min - margin
                layout.bbox[2] + margin,  # x_max + margin
                layout.bbox[3] + margin,  # y_max + margin
            ]

            # Find the first matching text box
            for text in self.texts_data:
                text_bbox = text.bbox
                if (
                    layout_bbox[0] <= text_bbox[0] and layout_bbox[1] <= text_bbox[1] and
                    layout_bbox[2] >= text_bbox[2] and layout_bbox[3] >= text_bbox[3]
                ):
                    matched_results.append({
                        "bbox_text": text_bbox,
                        "bbox_layout": layout_bbox,
                        "text": text.text,
                        "label": layout.label,
                        "position": layout.position,
                    })
                    # texts_data.remove(text)  # Remove matched text to avoid redundant checks
                    break  # Stop searching for texts for this layout

        return matched_results
