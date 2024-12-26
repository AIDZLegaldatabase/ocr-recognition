import os
import requests
import shutil
from pathlib import Path

class JoradpImporter:

    @staticmethod
    def download_pdfs_for_year(year, save_directory="./downloads/"):
        """
        Downloads all PDF files for a given year from the joradp URL pattern.
        
        Args:
            year (int): The year to download files for.
            save_directory (str): Directory to save the downloaded files.
        """
        base_url = "https://www.joradp.dz/FTP/jo-francais/"
        save_directory += str(year)
        os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

        number = 1
        while True:
            # Format the file number as a three-digit string
            file_number = f"{number:03}"
            url = f"{base_url}/F{year}{file_number}.pdf"
            save_path = os.path.join(save_directory, f"F{year}{file_number}.pdf")

            try:
                #print(f"Attempting to download: {url}")
                response = requests.get(url, stream=True)

                # Check for 404 status to stop the loop
                if response.status_code == 404:
                    # print(f"File not found: {url}. Stopping.")
                    break

                # Save the PDF file if found
                with open(save_path, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=1024):
                        pdf_file.write(chunk)

                # print(f"Downloaded: {save_path}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                break

            number += 1

    @staticmethod
    def get_pdf_files(directory):
        """
        Retrieves a list of all PDF files in the given directory.

        Args:
            directory (str): The path to the directory containing the PDF files.

        Returns:
            list: A list of file paths to the PDF files.
        """
        if not os.path.exists(directory):
            raise ValueError(f"The directory '{directory}' does not exist.")
        
        pdf_files = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".pdf")
        ]
        
        return pdf_files
    

    @staticmethod
    def create_yearly_zip_files(base_directory, output_directory):
        """
        Creates a ZIP file for each year folder inside the base directory and saves it in the output directory.
        EX:
        base_directory = "./downloads"  # Directory containing year folders
        output_directory = "./year_zip-files"  # Directory to save ZIP files
        create_yearly_zip_files(base_directory, output_directory)
        """
        base_path = Path(base_directory)
        output_path = Path(output_directory)

        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Iterate over year directories in the base directory
        for year_dir in base_path.iterdir():
            if year_dir.is_dir():  # Check if it's a directory
                year = year_dir.name
                zip_file_path = output_path / f"{year}.zip"

                # Create a ZIP file for the year directory
                shutil.make_archive(zip_file_path.with_suffix(""), "zip", year_dir)
                print(f"Created ZIP file: {zip_file_path}")