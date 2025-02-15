
from classes.joradp_importer import JoradpImporter


if __name__ == "__main__":
    JoradpImporter.download_pdfs_for_year(1974, save_directory="./year_files_raw/")
    for i in range(1990, 2026):
        JoradpImporter.download_pdfs_for_year(i, save_directory="./year_files_raw/")
