
from classes.joradp_importer import JoradpImporter


if __name__ == "__main__":
    for i in range(1975, 2026):
        JoradpImporter.download_pdfs_for_year(i, save_directory="./year_files_raw/")
