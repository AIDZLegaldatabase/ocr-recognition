from pdf2image import convert_from_path
from PIL import Image

class JoradpFileParse:
    def __init__(self, pdf_path: str):
        """
        Initialize with the path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.images = []
        self.ocr_layout = []
        self.ocr_lines = []
        self.ocr_texts = []

    def get_images(self, dpi: int = 300) -> list:
        """
        Convert the PDF into a list of images (one per page).

        Args:
            dpi (int): DPI for image conversion.

        Returns:
            list: List of PIL.Image objects.
        """
        try:
            # Convert the PDF to images
            self.images = convert_from_path(self.pdf_path, dpi)
        except Exception as e:
            print(f"An error occurred while converting PDF to images: {e}")
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
            new_height = 1600
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