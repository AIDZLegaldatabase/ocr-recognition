
import os
import zipfile
from pathlib import Path

def unzip_all_grouped_by_year(zip_folder, output_folder):
    """
    Unzips all ZIP files in the given folder into subfolders grouped by year.

    Args:
        zip_folder (str): The path to the folder containing ZIP files.
        output_folder (str): The path to the folder where the contents will be extracted.
    """
    zip_folder_path = Path(zip_folder)
    output_folder_path = Path(output_folder)

    # Ensure the output folder exists
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all ZIP files in the folder
    for zip_file in zip_folder_path.glob("*.zip"):
        year = zip_file.stem  # Extract the year from the ZIP file name (e.g., "1962" from "1962.zip")
        year_output_folder = output_folder_path / year

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract into a subfolder for the specific year
            year_output_folder.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(year_output_folder)
        print(f"Extracted: {zip_file} to {year_output_folder}")

unzip_all_grouped_by_year("./year_zip-files","./year_files_raw")