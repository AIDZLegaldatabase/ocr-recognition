import numpy as np
import cv2
from table_detection.table_detection import (
    detect_table_from_image_data,
    detect_table_cells,
)
import argparse
from classes.pdf_parser import JoradpFileParse
from pathlib import Path
import re
import json
import glog as log
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile
import webbrowser
import os

# cases 1 python .\test_tables.py -d -input_pdf_file ./data_test/F1962008.pdf
# cases 2 python .\test_tables.py -d -input_pdf_file ./data_test/F2024007.pdf
# cases 3 python .\test_tables.py -d -input_pdf_file ./data_test/F2025009.pdf


def args_parser():
    parser = argparse.ArgumentParser(prog="Table detection tester")

    parser.add_argument("-r", "--generate_report", action="store_true")
    parser.add_argument("-d", "--debug_mode", action="store_true")
    parser.add_argument("-input_pdf_file", type=str, default="")
    parser.add_argument("-input_pdf_dir", type=str, default="")

    return parser.parse_args()


def get_formatted_pdfs(folder_path):
    """
    Scans a folder for PDF files matching the pattern Fxxxxxxx.pdf
    (Where 'x' represents exactly 7 digits).
    """
    folder = Path(folder_path)

    # Regex explanation:
    # ^      : Start of string
    # F      : Literal character 'F'
    # \d{7}  : Exactly 7 digits (0-9)
    # \.pdf$ : Ends with literal .pdf
    pattern = re.compile(r"^F\d{7}\.pdf$")

    matching_files = []

    # robust check to ensure folder exists
    if not folder.exists():
        return []

    for file_path in folder.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            # Returns the full path string. Change to file_path.name if you just want filenames.
            matching_files.append(str(file_path))

    return matching_files


def parse_images(pdf_path):
    parserImages = JoradpFileParse(pdf_path)

    parserImages.get_images_with_pymupdf()
    parserImages.resize_image_to_fit_ocr()

    print(f"Parsed {len(parserImages.images)} images from the PDF file")

    inut_data_path = Path(pdf_path)

    if inut_data_path.name.startswith("F1978"):
        # 1978
        parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)
    elif inut_data_path.name.startswith("F1962"):
        # 1978
        parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)
    elif inut_data_path.name.startswith("F2025"):
        parserImages.crop_all_images(top=110, left=70, right=70, bottom=90)
        # parserImages.adjust_all_images_rotations_parallel()

    elif inut_data_path.name.startswith("F2024"):
        # 2024
        parserImages.crop_all_images(top=110, left=70, right=70, bottom=90)
        parserImages.adjust_all_images_rotations_parallel()
    else:
        raise ValueError(
            "There was no crop with this pdf file, are you sure about that?"
        )
    return parserImages.images[1:]


# -----------------
# Helper: IoU Calculation
# -----------------
def calculate_iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# -----------------
# Helper: HTML Report Generator (Modified)
# -----------------
class HtmlReporter:
    def __init__(self):
        self.failures = []

    def add_failure(self, page_idx, test_file, image_bgr, reason):
        # Convert BGR to RGB then to Base64
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

        self.failures.append(
            {"page": page_idx, "test_file": test_file, "img": img_str, "reason": reason}
        )

    def _generate_html_content(self):
        # FIX: We use an f-string (f""")
        # We must use DOUBLE braces {{ }} for the CSS so Python ignores them.
        # We use SINGLE braces { } for the Python variables we want to inject.
        html = f"""
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{ font-family: sans-serif; padding: 20px; background: #f0f0f0; }}
                .case {{ background: white; border: 1px solid #ccc; margin: 20px auto; padding: 20px; max-width: 900px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                img {{ max-width: 100%; height: auto; border: 2px solid #333; margin-top: 10px; }}
                .reason {{ color: #d9534f; font-weight: bold; font-size: 1.1em; }}
                h1 {{ text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Detection Failure Report</h1>
            <p style="text-align:center">Total Failures: <strong>{len(self.failures)}</strong></p>
        """

        if not self.failures:
            html += "<h3 style='text-align:center; color:green'>No failures detected! Great job.</h3>"

        for fail in self.failures:
            html += f"""
            <div class='case'>
                <h3>Page {fail['page']} of the file {fail['test_file']}</h3>
                <p class='reason'>Error: {fail['reason']}</p>
                <img src="data:image/jpeg;base64,{fail['img']}" />
            </div>
            """
        html += "</body></html>"
        return html

    def show(self):
        """Creates a temp file and opens it in the browser immediately."""
        fd, path = tempfile.mkstemp(suffix=".html")
        try:
            with os.fdopen(fd, "w") as tmp:
                tmp.write(self._generate_html_content())

            log.info(f"Opening temporary report: {path}")
            webbrowser.open(f"file://{path}")

        except Exception as e:
            log.error(f"Failed to open report: {e}")


if __name__ == "__main__":
    args = args_parser()
    pages_collections = {}
    if args.input_pdf_dir != "":
        pdf_files_list = get_formatted_pdfs(args.input_pdf_dir)
        for pdf_path in pdf_files_list:
            file_name = file_name = os.path.splitext(os.path.basename(pdf_path))[0]
            dir_path = args.input_pdf_dir
            pages_collections[file_name] = parse_images(pdf_path)

    elif args.input_pdf_file != "":
        file_name = file_name = os.path.splitext(os.path.basename(args.input_pdf_file))[
            0
        ]
        dir_path = os.path.dirname(args.input_pdf_file)
        pages_collections[file_name] = parse_images(args.input_pdf_file)
    else:
        raise ValueError(
            "input_pdf_file and input_pdf_dir inpiut flags cannot be both empty"
        )

    window_name = ""
    if args.debug_mode:
        # Create a resizable window in advance
        window_name = "Table Detection Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.generate_report:
        reporter = HtmlReporter()
    log.info("Starting evaluation")
    table_tp = 0
    table_fp = 0
    table_fn = 0

    cell_tp = 0
    cell_fp = 0
    cell_fn = 0

    for test_file in pages_collections.keys():
        # Load images
        image_to_process = pages_collections[test_file]
        log.info(f"Processing {test_file}...")
        # Load Ground Truth
        gt_path = os.path.join(dir_path, f"{test_file}-annotations.json")
        with open(gt_path, "r") as f:
            ground_truth = json.load(f)

        for i, pil_page_image in enumerate(image_to_process):
            # 1. Convert PIL Image (assuming RGB) to OpenCV format (BGR)
            # and create a copy of the image to draw on
            page_image_rgb = np.array(pil_page_image)
            debug_image = page_image_rgb.copy()

            # 2. detect tables from image
            predicted_tables, img_grid = detect_table_from_image_data(page_image_rgb)

            # Retrieve GT for this page
            page_idx_str = str(i)
            page_gt = ground_truth.get(page_idx_str, {"tables": []})
            gt_tables = page_gt["tables"]

            matched_table_indices = set()
            failed_table_detection = False
            failed_cell_detection = False
            for p_tbl_idx, p_table in enumerate(predicted_tables):
                p_table_x, p_table_y, p_table_width, p_table_height = p_table

                # Find best GT match
                best_table_iou = 0
                best_table_gt_idx = -1

                for g_idx, g_tbl in enumerate(gt_tables):
                    iou = calculate_iou(p_table, g_tbl["bbox"])
                    if iou > best_table_iou:
                        best_table_iou = iou
                        best_table_gt_idx = g_idx

                # Check Table Match
                if best_table_iou < 0.5:  # Threshold
                    if args.generate_report:
                        log.error(
                            f"Page {i} from file {test_file}: False Positive Table detected at {p_table}"
                        )
                    table_fp += 1
                    cv2.rectangle(
                        debug_image,
                        (p_table_x, p_table_y),
                        (p_table_x + p_table_width, p_table_y + p_table_height),
                        (0, 0, 255),
                        2,
                    )  # RED for Fail
                    failed_table_detection = True
                    continue  # Skip cell check for invalid table
                else:
                    if args.generate_report:
                        log.info(
                            f"Page {i} from file {test_file}: Table Matched (IoU: {best_table_iou:.2f})"
                        )
                    table_tp += 1
                    matched_table_indices.add(best_table_gt_idx)
                    # print green (good) prediction
                    cv2.rectangle(
                        debug_image,
                        (p_table_x, p_table_y),
                        (p_table_x + p_table_width, p_table_y + p_table_height),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        debug_image,
                        "TP",
                        (p_table_x, p_table_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    # Check Cells for this matched table
                    pred_cells = detect_table_cells(page_image_rgb, p_table)
                    gt_cells = gt_tables[best_table_gt_idx].get("cells", [])

                    # Check all indices for false negatives
                    matched_cell_indices = set()

                    # IoU for cells as well
                    for p_cell_idx, p_cell in enumerate(pred_cells):
                        p_cell_x, p_cell_y, p_cell_width, p_cell_height = p_cell

                        # Find best GT match
                        best_cell_iou = 0
                        best_cell_gt_idx = -1

                        for g_cell_idx, g_cell in enumerate(gt_cells):
                            iou = calculate_iou(p_cell, g_cell)
                            if iou > best_cell_iou:
                                best_cell_iou = iou
                                best_cell_gt_idx = g_cell_idx

                        # Check Cell Match
                        if best_cell_iou < 0.5:  # Threshold
                            if args.generate_report:
                                log.error(
                                    f"Page {i} from file {test_file}: False Positive Cell detected at {p_cell}"
                                )
                            cell_fp += 1
                            cv2.rectangle(
                                debug_image,
                                (p_cell_x, p_cell_y),
                                (p_cell_x + p_cell_width, p_cell_y + p_cell_height),
                                (0, 0, 255),
                                2,
                            )  # Red for Fail cell
                            failed_cell_detection = True
                        else:
                            if args.generate_report:
                                log.info(
                                    f"Page {i} from file {test_file}: Cell Matched (IoU: {best_cell_iou:.2f})"
                                )
                            cell_tp += 1
                            matched_cell_indices.add(best_cell_gt_idx)
                            cv2.rectangle(
                                debug_image,
                                (p_cell_x, p_cell_y),
                                (p_cell_x + p_cell_width, p_cell_y + p_cell_height),
                                (0, 255, 0),
                                2,
                            )

                    # Check for Missed Cells (False Negatives)
                    for g_cell_idx, g_cell in enumerate(gt_cells):
                        if g_cell_idx not in matched_cell_indices:
                            g_cell_x, g_cell_y, g_cell_w, g_cell_h = g_cell
                            if args.generate_report:
                                log.error(
                                    f"Page {i} from file {test_file}: Missed Cell (FN) at {g_cell}, index {g_cell_idx}"
                                )
                            cell_fn += 1
                            cv2.rectangle(
                                debug_image,
                                (g_cell_x, g_cell_y),
                                (g_cell_x + g_cell_w, g_cell_y + g_cell_h),
                                (255, 0, 255),
                                3,
                            )  # Purple for Missed
                            failed_cell_detection = True

            # Check for Missed Tables (False Negatives)
            for g_idx, g_tbl in enumerate(gt_tables):
                if g_idx not in matched_table_indices:
                    gx, gy, gw, gh = g_tbl["bbox"]
                    if args.generate_report:
                        log.error(
                            f"Page {i} from file {test_file}: Missed Table (FN) at {g_tbl['bbox']}"
                        )
                    table_fn += 1
                    cv2.rectangle(
                        debug_image, (gx, gy), (gx + gw, gy + gh), (255, 0, 255), 3
                    )  # Purple for Missed
                    failed_table_detection = True

            if args.generate_report and (
                failed_table_detection or failed_cell_detection
            ):
                reporter.add_failure(
                    i + 2,
                    test_file,
                    debug_image,
                    f"Failed table detection: {failed_table_detection}. Failed Cell detection: {failed_cell_detection}",
                )
            if args.debug_mode:

                combined_display = cv2.hconcat(
                    [cv2.cvtColor(img_grid, cv2.COLOR_GRAY2BGR), debug_image]
                )
                # 6. Show the combined image and wait
                cv2.imshow(window_name, combined_display)
                # Wait indefinitely for a key press
                key = cv2.waitKey(0)

                # 7. Add a quit condition
                if key == ord("q"):
                    print("Quitting loop...")
                    break
    # 8. Clean up windows after the loop
    cv2.destroyAllWindows()
    # Finalize
    if args.generate_report:
        reporter.show()
    log.info(
        "Evaluation complete....\n"
        "Table Detection:\n "
        f"Tables Detected: {table_tp} \t Tables Missed: {table_fn} \t Precison: {table_tp/(table_tp + table_fp)} \t Recall: {table_tp/(table_tp + table_fn)}\n"
        "Cell Detection:\n "
        f"Cells Detected: {cell_tp} \t Cells Missed: {cell_fn} \t Precison: {cell_tp/(cell_tp + cell_fp)} \t Recall: {cell_tp/(cell_tp + cell_fn)}"
    )
