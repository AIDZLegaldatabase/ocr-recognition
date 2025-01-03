
from classes.joradp_importer import JoradpImporter


if __name__ == "__main__":
    for i in range(1962, 2024):
        JoradpImporter.download_pdfs_for_year(i, save_directory="./year_files_raw/")
