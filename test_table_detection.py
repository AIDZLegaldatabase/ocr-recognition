import cv2
import numpy as np
import sys
import os
from typing import List, Dict
import time
from tqdm import tqdm
from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter


def find_clusters_1d(
    data: List[float], gap_threshold: float, min_cluster_size: int = 1
) -> Dict[int, List[float]]:
    """
    Finds clusters in a 1D list of numbers based on a simple
    gap threshold.

    Args:
        data: The list of numbers.
        gap_threshold: The maximum gap to allow *inside* a cluster.
        min_cluster_size: The minimum number of items to be
                          considered a "real" cluster.

    Returns:
        A dictionary where keys are cluster IDs (0, 1, 2...)
        and values are the lists of numbers in that cluster.
    """
    if not data:
        return {}

    # Step 1: Sort the data
    data.sort()

    clusters = {}
    current_cluster_id = 0
    current_cluster = [data[0]]

    # Step 2 & 3: Iterate and find gaps
    for i in range(len(data) - 1):
        # Calculate the gap
        gap = data[i + 1] - data[i]

        if gap > gap_threshold:
            # "Break" - end the current cluster and start a new one
            clusters[current_cluster_id] = current_cluster
            current_cluster_id += 1
            current_cluster = [data[i + 1]]
        else:
            # "No break" - add to the current cluster
            current_cluster.append(data[i + 1])

    # Add the last cluster
    clusters[current_cluster_id] = current_cluster

    # Step 4 (Optional): Filter by size
    final_clusters = {}
    final_id = 0
    for cluster_data in clusters.values():
        if len(cluster_data) >= min_cluster_size:
            final_clusters[final_id] = cluster_data
            final_id += 1

    return final_clusters


def find_inferred_boxes(
    contours_v: list, image_height: int, height_threshold_ratio: float = 0.95
):
    """
    Finds table boxes by inferring them from parallel, full-height
    vertical lines. This handles cases where top/bottom lines are cropped.

    Args:
        contours_v: The list of vertical line contours.
        image_height: The height of the image.
        height_threshold_ratio: What percentage of the total height
                                a line must be to be considered "full-height".

    Returns:
        A list of inferred bounding box tuples (x, y, w, h).
    """

    full_height_line_x_coords = []

    # 1. Find all "full-height" vertical lines
    for cnt in contours_v:
        (lx, ly, lw, lh) = cv2.boundingRect(cnt)

        # Check if the line's height is >= 95% of the image height
        if lh >= (image_height * height_threshold_ratio):
            full_height_line_x_coords.append(lx)

    # 2. Sort the x-coordinates
    full_height_line_x_coords.sort()

    # 3. Pair up adjacent lines to form boxes
    inferred_boxes = []
    if len(full_height_line_x_coords) >= 2:
        for i in range(len(full_height_line_x_coords) - 1):
            x1 = full_height_line_x_coords[i]
            x2 = full_height_line_x_coords[i + 1]

            # Create the full-height bounding box
            x = x1
            y = 0
            w = x2 - x1
            h = image_height

            inferred_boxes.append((x, y, w, h))

    return inferred_boxes


def find_table_bounding_boxes(table_grid):
    """
    Finds the bounding boxes of tables by "smearing" (closing)
    the detected horizontal and vertical lines.

    Args:
        morphed_horizontal: The binary image containing only long horizontal lines.
        morphed_vertical: The binary image containing only long vertical lines.

    Returns:
        A list of bounding box tuples (x, y, w, h).
    """

    TABLE_MIN_HEIGHT = 100
    TABLE_MIN_WIDTH = 185

    # 1. Combine the horizontal and vertical line images
    # table_grid = cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    # 2. Morphological Close ("Smear" the lines together)
    # This is the most important parameter to tune.
    # (15, 15) means it will connect lines that are up to 15px apart.
    kernel_size = (10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    closed_grid = cv2.morphologyEx(table_grid, cv2.MORPH_CLOSE, kernel)

    # 3. Find contours of the new "blobs"
    # These contours now represent whole tables, not individual lines.
    contours, _ = cv2.findContours(
        closed_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bounding_boxes = []
    for cnt in contours:
        # 4. Get the bounding box for each blob
        bbox = cv2.boundingRect(cnt)
        if bbox[2] > TABLE_MIN_WIDTH and bbox[3] > TABLE_MIN_HEIGHT:
            bounding_boxes.append(bbox)

    return bounding_boxes


def core_line_detection(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel_x = np.absolute(sobel_x)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel_y = np.absolute(sobel_y)

    # 2. Threshold
    _, thresh_x = cv2.threshold(abs_sobel_x, 200, 255, cv2.THRESH_BINARY)
    _, thresh_y = cv2.threshold(abs_sobel_y, 200, 255, cv2.THRESH_BINARY)

    # 3. Morphological Operations
    horizontal_kernel_len = int(gray.shape[1] / 20)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))

    vertical_kernel_len = int(gray.shape[0] / 20)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    morphed_horizontal = cv2.morphologyEx(
        thresh_y.astype(np.uint8), cv2.MORPH_OPEN, hor_kernel
    )
    morphed_vertical = cv2.morphologyEx(
        thresh_x.astype(np.uint8), cv2.MORPH_OPEN, ver_kernel
    )

    # Find contours (i.e., distinct lines) in the horizontal image
    contours_h, _ = cv2.findContours(
        morphed_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find contours (i.e., distinct lines) in the vertical image
    contours_v, _ = cv2.findContours(
        morphed_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    combined_grid = cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    return combined_grid, contours_v, contours_h


def filter_central_v_line(contours_v, img_width):
    CENTRE_LINE_TOLERANCE = 200
    vertical_lines_clusters = find_clusters_1d(
        [cv2.boundingRect(cnt)[0] for cnt in contours_v],
        gap_threshold=10,
        min_cluster_size=1,
    )

    # Find the cluster of lines in the middle of the page:
    centre_lines_coordinates_x = [
        vertical_lines_clusters[x]
        for x in vertical_lines_clusters.keys()
        if all(
            (img_width / 2) - CENTRE_LINE_TOLERANCE
            < elt
            < (img_width / 2) + CENTRE_LINE_TOLERANCE
            for elt in vertical_lines_clusters[x]
        )
    ]

    # If there's a central cluster then ommit any line item which has the x of
    # the bounding box in the list of cenytre lines coordinates
    contours_v_without_central = (
        contours_v
        if len(centre_lines_coordinates_x) == 0
        else [
            cnt
            for cnt in contours_v
            if cv2.boundingRect(cnt)[0] not in centre_lines_coordinates_x[0]
        ]
    )

    return contours_v_without_central


def detect_table_from_image_data(img: np.ndarray):
    """
    Detects if an image's data (numpy array) contains a table.

    Args:
        img: A NumPy array (OpenCV image)

    Returns:
        True if a table is likely present, False otherwise.
    """
    if img is None:
        print("Error: Invalid image data.")
        return False
    # PARAMS
    MIN_HORIZONTAL_LINES = 3
    MIN_VERTICAL_LINES = 2
    LINE_MINIMAL_WIDTH = 140
    NUM_TOTAL_LINES = 4
    LINE_MINIMAL_HEIGHT = 130
    has_table = False
    centre_line_x = 0

    # Crop image
    width, height, _ = img.shape

    # Perform the detection in the main function
    combined_grid, contours_v, contours_h = core_line_detection(img)

    # Validate number of countours

    # Filter lines by size:
    contours_h = [
        cnt for cnt in contours_h if cv2.boundingRect(cnt)[2] > LINE_MINIMAL_WIDTH
    ]

    contours_v = [
        cnt for cnt in contours_v if cv2.boundingRect(cnt)[3] > LINE_MINIMAL_HEIGHT
    ]

    # Remove central line from vertical lines
    contours_v = filter_central_v_line(contours_v, width)

    # Create new image with combined grids
    mask = np.zeros(combined_grid.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours_h, -1, (255), cv2.FILLED)
    cv2.drawContours(mask, contours_v, -1, (255), cv2.FILLED)
    img_grid = cv2.bitwise_and(combined_grid, combined_grid, mask=mask)

    # Find all the bounding boxes around group of lines
    # Find the bounding boxes of the tables
    table_boxes = find_table_bounding_boxes(img_grid)
    filtered_boxes = []

    # Filter the bounding boxes by the number of lines inside each one
    for bbox in table_boxes:
        (bx, by, bw, bh) = bbox
        cv2.rectangle(img_grid, (bx, by), (bx + bw, by + bh), 255, 3)
        h_count = 0
        v_count = 0

        # Check horizontal lines
        for cnt in contours_h:
            # Get the bounding box of the line
            (lx, ly, lw, lh) = cv2.boundingRect(cnt)

            # Find the center of the line's bounding box
            center_x = lx + lw // 2
            center_y = ly + lh // 2

            # Check if the line's center is inside the table's box
            if (bx < center_x < bx + bw) and (by < center_y < by + bh):
                h_count += 1

        # Check vertical lines
        for cnt in contours_v:
            (lx, ly, lw, lh) = cv2.boundingRect(cnt)
            center_x = lx + lw // 2
            center_y = ly + lh // 2
            # The last check is only done in vertical lines because they're continuous
            # usually so we can filter small vertical lines
            # TODO add similar check for verticals, you want to see if a line is floating and not connected anywhere
            # when you're looking fir bboxes
            if (
                (bx < center_x < bx + bw)
                and (by < center_y < by + bh)
                and (lh / bh > 0.5)
            ):
                v_count += 1
        if (h_count >= MIN_HORIZONTAL_LINES and v_count >= MIN_VERTICAL_LINES) or (
            h_count + v_count > NUM_TOTAL_LINES
        ):
            filtered_boxes.append(bbox)

    return filtered_boxes, img_grid


parserImages = JoradpFileParse("./data_test/F1978025.pdf")
print("with ocr")

parserImages.get_images_with_pymupdf()
parserImages.resize_image_to_fit_ocr()
# 1978
parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)

# 2024
#parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
#parserImages.adjust_all_images_rotations_parallel()


# --- Part C: Analyze the extracted pages ---
debug_mode = True
# Create a resizable window in advance
window_name = "Table Detection Debug"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


print("\n--- Analyzing Extracted Pages ---")

for i, pil_page_image in enumerate(parserImages.images):

    if i == 0:
        continue
    # 1. Convert PIL Image (assuming RGB) to OpenCV format (BGR)
    page_image_rgb = np.array(pil_page_image)
    # page_image_gray = cv2.cvtColor(page_image_rgb, cv2.COLOR_RGB2GRAY)

    # Use our refactored function
    table_boxes, img_grid = detect_table_from_image_data(page_image_rgb)
    # cv2.imwrite(f"page_{i+1}_grid.png", img_grid)
    print(f"Found {len(table_boxes)} table(s) on this page.")

    # Create a copy of the image to draw on
    debug_image = page_image_rgb
    if table_boxes:
        # 3. Draw the boxes
        for x, y, w, h in table_boxes:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # Save the result
        # cv2.imwrite(f"page_{i+1}_tables_detected.png", debug_image)

        print(f"--- Page {i+1}: Table(s) detected! ---")

    else:
        print(f"--- Page {i+1}: No tables found. ---")

    if debug_mode:

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
