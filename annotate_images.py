from annotation_utils.table_annotator import run_annotator
import tkinter as tk
import argparse
from test_tables import get_formatted_pdfs, parse_images
import os
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser(prog="Table annotator app")
    parser.add_argument("-input_pdf_file", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    image_to_process = []
    
    image_to_process = parse_images(args.input_pdf_file)
    dir_path = os.path.dirname(args.input_pdf_file)
    file_name = os.path.splitext(os.path.basename(args.input_pdf_file))[0]
    output_path = os.path.join(dir_path,f"{file_name}-annotations.json")
    np_images = [np.array(image) for image in image_to_process]

    # Make them gray/white so we can see drawing
    run_annotator(np_images, output_path)
    