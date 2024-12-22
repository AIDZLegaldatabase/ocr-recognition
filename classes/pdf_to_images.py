from pdf2image import convert_from_path
from PIL import Image

class PDFParser:
    def __init__(self, pdf_path):
        """
        Initialize the PDFParser with the path to the PDF file.
        """
        self.pdf_path = pdf_path

    def get_pages_as_images(self):
        """
        Convert the PDF to a list of images, one for each page.

        Returns:
            list: List of PIL.Image objects representing the pages of the PDF.
        """
        try:
            # Convert the PDF to images
            images = convert_from_path(self.pdf_path, 96)
            return images
        except Exception as e:
            print(f"An error occurred while converting PDF to images: {e}")
            return []

    def display_third_page(self):
        """
        Display the 3rd page of the PDF as an image.
        """
        try:
            # Get all pages as images
            images = self.get_pages_as_images()
            
            # Ensure the PDF has at least 3 pages
            if len(images) < 3:
                print("The PDF has less than 3 pages.")
                return
            
            # Select the 3rd page (index 2 in the list)
            third_page_image = images[2]
            
            # Display the 3rd page as an image
            third_page_image.show()
            print("Displaying the 3rd page of the PDF.")
        except Exception as e:
            print(f"An error occurred: {e}")