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
        
        annotated_image = self.raw_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Font for text (optional, adjust path to a font file if needed)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Iterate through each layout box
        for box in self.layout_data:
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

    
    def draw_text_on_image(self):
        """
        Draw layout bounding boxes on the image with labels and positions.

        Raises:
            ValueError: If raw_image or layout_data is not set.
        """
        if self.raw_image is None or self.text_data is None:
            raise ValueError("Data of layout or raw image is not properly set.")

        """
        Draw layout bounding boxes on the image with labels and positions.

        Raises:
            ValueError: If raw_image or layout_data is not set.
        """
        if self.raw_image is None or self.text_data is None:
            raise ValueError("Data of layout or raw image is not properly set.")
        
        annotated_image = self.raw_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Font for text (optional, adjust path to a font file if needed)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Iterate through each layout box
        for box in self.text_data:
            # Extract polygon and label
            polygon = [(int(x), int(y)) for x, y in box.polygon]  # Convert to integers
            label = box.text
            if len(label) > 15:
                label = f"{label[:5]}...{label[-5:]}"

            # Draw polygon
            draw.polygon(polygon, outline="red", width=2)

            # Annotate with label and position
            label_text = f"{label}"
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
    
    def visualize_margin_and_layout(self, margin=5):
        """
        Visualize the layout boxes with added margins and the text boxes within them.

        Args:
            margin (int): Margin to adjust bounding boxes for visualization. Defaults to 5.

        Returns:
            PIL.Image.Image: Annotated image.
        """
        if self.raw_image is None or self.layout_data is None or self.text_data is None:
            raise ValueError("Image, layout data, or text data is not set.")

        annotated_image = self.raw_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Font for text annotations
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Iterate over layout boxes
        for layout in self.layout_data:
            # Adjust layout bbox with margin
            layout_bbox = [
                layout.bbox[0] - margin,  # x_min - margin
                layout.bbox[1] - margin,  # y_min - margin
                layout.bbox[2] + margin,  # x_max + margin
                layout.bbox[3] + margin,  # y_max + margin
            ]

            # Draw the adjusted layout box
            draw.rectangle(layout_bbox, outline="green", width=2)

            # Collect all text boxes within the layout bbox
            texts_in_layout = []
            for text in self.text_data:
                text_bbox = text.bbox
                if (
                    layout_bbox[0] <= text_bbox[0] and layout_bbox[1] <= text_bbox[1] and
                    layout_bbox[2] >= text_bbox[2] and layout_bbox[3] >= text_bbox[3]
                ):
                    texts_in_layout.append({
                        "text": text.text,
                        "bbox": text_bbox,
                        "start_point": text_bbox[:2],  # First point (x_min, y_min)
                    })

            # Sort text boxes by y-coordinate, then x-coordinate
            sorted_texts = sorted(
                texts_in_layout,
                key=lambda item: (item["start_point"][1], item["start_point"][0])
            )

            # Merge texts in order
            merged_text = " ".join([text["text"] for text in sorted_texts])

            # Draw each text box and annotate
            for text in sorted_texts:
                text_bbox = text["bbox"]
                draw.rectangle(text_bbox, outline="blue", width=1)

            # Annotate the layout box with the merged text
            label_text = f"{layout.label}: {merged_text}"
            draw.text((layout_bbox[0], layout_bbox[1] - 15), label_text, fill="red", font=font)

        return annotated_image

    def match_making_texts_to_layouts(self, margin=5)-> dict[str, list]:
        """
        Match and merge text boxes into layout boxes based on adjusted bounding boxes and ordering.

        Args:
            margin (int, optional): Margin to adjust bounding boxes for matching. Defaults to 5.

        Returns:
            dict: A dictionary containing:
                - 'matched_results': List of dictionaries with merged text and layout information
                - 'unmatched_texts': List of text boxes that don't belong to any layout
        """
        matched_results = []
        matched_text_indices = set()  # Track which texts have been matched

        # Iterate over layout data
        for layout in self.layout_data:
            # Adjust layout bbox with margin
            layout_bbox = [
                layout.bbox[0] - margin,  # x_min - margin
                layout.bbox[1] - margin,  # y_min - margin
                layout.bbox[2] + margin,  # x_max + margin
                layout.bbox[3] + margin,  # y_max + margin
            ]

            # Collect all text boxes within the layout bbox
            texts_in_layout = []
            for i, text in enumerate(self.text_data):
                text_bbox = text.bbox
                if (
                    layout_bbox[0] <= text_bbox[0] and layout_bbox[1] <= text_bbox[1] and
                    layout_bbox[2] >= text_bbox[2] and layout_bbox[3] >= text_bbox[3]
                ):
                    texts_in_layout.append({
                        "text": text.text,
                        "bbox": text_bbox,
                        "start_point": text_bbox[:2],  # First point (x_min, y_min)
                    })
                    matched_text_indices.add(i)  # Mark this text as matched

            # Sort text boxes by y-coordinate, then x-coordinate
            sorted_texts = sorted(
                texts_in_layout,
                key=lambda item: (item["start_point"][1], item["start_point"][0])
            )

            # Append the result
            if sorted_texts:  # Only add if there are texts within the layout box
                matched_results.append({
                    "bbox_text": [text["bbox"] for text in sorted_texts],
                    "bbox_layout": layout_bbox,
                    "text": [text["text"] for text in sorted_texts],
                    "label": layout.label,
                    "position": layout.position,
                })

        # Collect unmatched texts
        unmatched_texts = []
        for i, text in enumerate(self.text_data):
            if i not in matched_text_indices:
                unmatched_texts.append({
                    "text": text.text,
                    "bbox": text.bbox,
                    "start_point": text.bbox[:2],
                })

        # Sort unmatched texts by y-coordinate, then x-coordinate for consistency
        unmatched_texts = sorted(
            unmatched_texts,
            key=lambda item: (item["start_point"][1], item["start_point"][0])
        )

        return {
            "result": matched_results,
            "rest": unmatched_texts
        }