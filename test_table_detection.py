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

def find_table_bounding_boxes(
    table_grid
):
    """
    Finds the bounding boxes of tables by "smearing" (closing)
    the detected horizontal and vertical lines.

    Args:
        morphed_horizontal: The binary image containing only long horizontal lines.
        morphed_vertical: The binary image containing only long vertical lines.

    Returns:
        A list of bounding box tuples (x, y, w, h).
    """
    
    # 1. Combine the horizontal and vertical line images
    #table_grid = cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    # 2. Morphological Close ("Smear" the lines together)
    # This is the most important parameter to tune.
    # (15, 15) means it will connect lines that are up to 15px apart.
    kernel_size = (15, 15) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    closed_grid = cv2.morphologyEx(table_grid, cv2.MORPH_CLOSE, kernel)

    # 3. Find contours of the new "blobs"
    # These contours now represent whole tables, not individual lines.
    contours, _ = cv2.findContours(
        closed_grid, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    

    bounding_boxes = []
    for cnt in contours:
        # 4. Get the bounding box for each blob
        bbox = cv2.boundingRect(cnt)
        if bbox[2] > 185 and bbox[3] > 100:
            bounding_boxes.append(bbox)
        
    return bounding_boxes


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
    LINE_MINIMAL_WIDTH = 185
    NUM_TOTAL_LINES = 4
    LINE_MINIMAL_HEIGHT = 100
    has_table = False
    centre_line_x = 0

    # Crop image
    width, height, _ = img.shape
    new_height = 1400
    y_start = max(0, height - new_height)

    # Perform the crop (slice)
    # [y_start: , :] means:
    # y-axis: from y_start to the end
    # x-axis: (:) all pixels
    resized_image = img[y_start:, :, :]

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # 1. Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.absolute(sobel_x)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
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
    
    combined_grid =  cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    # Validate number of countours

    # Remove small horizontal nad vertical lines, the width limit is purely empirical
    final_contours_h = [
        cnt for cnt in contours_h if cv2.boundingRect(cnt)[2] > LINE_MINIMAL_WIDTH
    ]
    
    final_contours_v = [
        cnt for cnt in contours_v if cv2.boundingRect(cnt)[3] > LINE_MINIMAL_HEIGHT
    ]
    
    # Create new image with combined grids
    
    mask = np.zeros(combined_grid.shape[:2], dtype=np.uint8)
    
    cv2.drawContours(mask, final_contours_h, -1, (255), cv2.FILLED)
    
    cv2.drawContours(mask, final_contours_v, -1, (255), cv2.FILLED)
    
    result = cv2.bitwise_and(combined_grid, combined_grid, mask=mask)
    
    #for cnt in contours_v:
    #    height= cv2.boundingRect(cnt)[3]
    #    print(f"The height of this line is {height}")    

    num_horizontal_lines = len(final_contours_h)
    num_vertical_lines = len(contours_v)


    vertical_lines_clusters = find_clusters_1d(
        [cv2.boundingRect(cnt)[0] for cnt in final_contours_v],
        gap_threshold=10,
        min_cluster_size=1,
    )
    

    num_vertical_lines = len(vertical_lines_clusters)
    if (
        num_horizontal_lines > MIN_HORIZONTAL_LINES
        and num_vertical_lines >= MIN_VERTICAL_LINES
    ):
        # Save debug images (optional)
        #cv2.imwrite(f"{i}_debug_horizontal.png", morphed_horizontal)
        #cv2.imwrite(f"{i}_debug_vertical.png", morphed_vertical)

        has_table = True
    # elif (
    #     (num_horizontal_lines + num_vertical_lines > NUM_TOTAL_LINES)
    #     and num_horizontal_lines > 0
    #     and (
    #         check_vertical_diversity([cv2.boundingRect(cnt)[2] for cnt in contours_v])
    #         > 1
    #     )
    # ):
    #     # Save debug images (optional)
    #     # cv2.imwrite(f"{i}_debug_horizontal.png", morphed_horizontal)
    #     cv2.imwrite(f"{i}_debug_vertical.png", morphed_vertical)

    #     has_table = True
    # IF we only have a single vertical line, get the first element of the first cluster 
    if num_vertical_lines == 1:
            centre_line_x = vertical_lines_clusters[0][0]
    
    return has_table, result


parserImages = JoradpFileParse("./data_test/F2024080.pdf")
print("with ocr")

parserImages.get_images_with_pymupdf()
parserImages.resize_image_to_fit_ocr()
# parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
# 1978
parserImages.crop_all_images(top=85, left=0, right=0, bottom=15)

# 2024
parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)

parserImages.adjust_all_images_rotations_parallel()

parserImages.adjust_all_images_rotations_parallel()

# --- Part C: Analyze the extracted pages ---

print("\n--- Analyzing Extracted Pages ---")

for i, page_image in enumerate(parserImages.images):
    
    if i==0:
        continue

    # Use our refactored function
    has_table, grid = detect_table_from_image_data(np.array(page_image))

    if has_table:
        print(f"--- Page {i+1}: Table(s) detected! ---")
        
        # 2. Find the bounding boxes of the tables
        table_boxes = find_table_bounding_boxes(grid)
        
        print(f"Found {len(table_boxes)} table(s) on this page.")

        # Create a copy of the image to draw on
        debug_image = np.array(page_image)

        if table_boxes:
            # 3. Draw the boxes
            for (x, y, w, h) in table_boxes:   
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 3) # Draw red box
            
                # Save the result
            cv2.imwrite(f"page_{i+1}_tables_detected.png", debug_image)

    else:
        print(f"--- Page {i+1}: No tables found. ---")

