import cv2
import numpy as np
from typing import List, Dict


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


def core_line_detection(img, kernel_size, invert_line_ratio):
    SOBEL_PIXEL_INTENSITY_THRESHOLD = 200

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Apply Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    abs_sobel_x = np.absolute(sobel_x)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    abs_sobel_y = np.absolute(sobel_y)

    # 2. Threshold
    _, thresh_x = cv2.threshold(
        abs_sobel_x, SOBEL_PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    _, thresh_y = cv2.threshold(
        abs_sobel_y, SOBEL_PIXEL_INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    # 3. Morphological Operations
    horizontal_kernel_len = int(gray.shape[1] / invert_line_ratio)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))

    vertical_kernel_len = int(gray.shape[0] / invert_line_ratio)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    morphed_horizontal = cv2.morphologyEx(
        thresh_y.astype(np.uint8), cv2.MORPH_OPEN, hor_kernel
    )
    morphed_vertical = cv2.morphologyEx(
        thresh_x.astype(np.uint8), cv2.MORPH_OPEN, ver_kernel
    )

    kernel_size = (10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    morphed_vertical_closed = cv2.morphologyEx(
        morphed_vertical, cv2.MORPH_CLOSE, kernel
    )

    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    morphed_horizontal_closed = cv2.morphologyEx(
        morphed_horizontal, cv2.MORPH_CLOSE, kernel
    )

    # Find contours (i.e., distinct lines) in the horizontal image
    contours_h, _ = cv2.findContours(
        morphed_horizontal_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Find contours (i.e., distinct lines) in the vertical image
    contours_v, _ = cv2.findContours(
        morphed_vertical_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    combined_grid = cv2.bitwise_or(morphed_horizontal_closed, morphed_vertical_closed)

    return combined_grid, contours_v, contours_h


def filter_central_v_line(contours_v, img_width):
    CENTRE_LINE_TOLERANCE = 100
    CLSUTERS_GAP_THRESHOLD = 10
    vertical_lines_clusters = find_clusters_1d(
        [cv2.boundingRect(cnt)[0] for cnt in contours_v],
        gap_threshold=CLSUTERS_GAP_THRESHOLD,
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
    idx_centre_lines = 0
    if len(centre_lines_coordinates_x) > 1:
        mean_diff = []
        for centre_line_cluster in centre_lines_coordinates_x:
            mean_diff.append(
                (
                    sum([abs(x - img_width / 2) for x in centre_line_cluster])
                    / len(centre_line_cluster)
                )
            )
        idx_centre_lines = mean_diff.index(min(mean_diff))

    # If there's a central cluster then ommit any line item which has the x of
    # the bounding box in the list of cenytre lines coordinates
    contours_v_without_central = (
        contours_v
        if len(centre_lines_coordinates_x) == 0
        else [
            cnt
            for cnt in contours_v
            if cv2.boundingRect(cnt)[0]
            not in centre_lines_coordinates_x[idx_centre_lines]
        ]
    )

    return contours_v_without_central


def detect_table_from_image_data(img: np.ndarray):
    """
    Detects if an image's data (numpy array) contains a table.

    Args:
        img: A NumPy array (OpenCV image)
    """
    if img is None:
        print("Error: Invalid image data.")
        return False
    # PARAMS
    MIN_HORIZONTAL_LINES = 1
    MIN_VERTICAL_LINES = 1
    NUM_TOTAL_LINES = 4
    LINE_MINIMAL_HEIGHT_RATIO = 0.087
    LINE_MINIMAL_WIDTH_RATIO = 0.137
    IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO = 0.0048
    IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO = 0.003

    image_height, image_width, _ = img.shape

    # Perform the detection in the main function
    combined_grid, contours_v, contours_h = core_line_detection(img, 5, 20)

    # Validate number of countours

    # Filter lines by size:
    contours_h = [
        cnt
        for cnt in contours_h
        if (
            cv2.boundingRect(cnt)[2] > (LINE_MINIMAL_WIDTH_RATIO * image_width)
            and (IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO * image_height)
            < cv2.boundingRect(cnt)[1]
            < (image_height * (1 - IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO))
        )
    ]

    contours_v = [
        cnt
        for cnt in contours_v
        if (
            cv2.boundingRect(cnt)[3] > LINE_MINIMAL_HEIGHT_RATIO
            and (IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO * image_width)
            < cv2.boundingRect(cnt)[0]
            < (image_width * (1 - IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO))
        )
    ]

    # Remove central line from vertical lines
    contours_v = filter_central_v_line(contours_v, image_width)

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
            if (
                (bx < center_x < bx + bw)
                and (by < center_y < by + bh)
                and (by + 15 < ly < by + bh - 15)
            ):
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
                and (bx + 15 < lx < bx + bw - 15)
            ):
                v_count += 1
        if (h_count >= MIN_HORIZONTAL_LINES and v_count >= MIN_VERTICAL_LINES) or (
            h_count + v_count > NUM_TOTAL_LINES
        ):
            filtered_boxes.append(bbox)

    return filtered_boxes, img_grid


def find_clusters_1d_arrays(
    lines_data, gap_threshold: float, min_cluster_size: int = 1, axis: int = 0
):
    """
    Finds clusters in a 1D list of numbers based on a simple
    gap threshold.

    Args:
        data: A list oflines list.
        gap_threshold: The maximum gap to allow *inside* a cluster.
        min_cluster_size: The minimum number of items to be
                          considered a "real" cluster.
        axis,: clustering axis, 0 for X axis and 1 for Y axis

    Returns:
        A dictionary where keys are cluster IDs (0, 1, 2...)
        and values are the lists of numbers in that cluster.
    """

    data = [line[axis] for line in lines_data]
    if not data:
        return {}

    # Step 1: Sort the data
    data.sort()

    clusters = {}
    current_cluster_id = 0
    current_cluster = [lines_data[0]]

    # Step 2 & 3: Iterate and find gaps
    for i in range(len(data) - 1):
        # Calculate the gap
        gap = data[i + 1] - data[i]

        if gap > gap_threshold:
            # "Break" - end the current cluster and start a new one
            clusters[current_cluster_id] = current_cluster
            current_cluster_id += 1
            current_cluster = [lines_data[i + 1]]
        else:
            # "No break" - add to the current cluster
            current_cluster.append(lines_data[i + 1])

    # Add the last cluster
    clusters[current_cluster_id] = current_cluster
    final_clusters = clusters

    return final_clusters


def display_lines(lines, image):
    # Create new image with combined grids
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for line in lines:
        cv2.rectangle(
            mask,
            (round(line[0]), round(line[1])),
            (round(line[2]), round(line[3])),
            255,
            -1,
        )

    return mask


def detect_table_cells(image, table_bbox):
    table_x_start = table_bbox[0]
    table_y_start = table_bbox[1]
    table_x_end = table_bbox[0] + table_bbox[2]
    table_y_end = table_bbox[1] + table_bbox[3]
    left_line = (0, 0, 5, table_bbox[3])
    top_line = (0, 0, table_bbox[2], 5)
    right_line = (table_bbox[2] - 10, 0, 5, table_bbox[3])
    bottom_line = (0, table_bbox[3] - 10, table_bbox[2], 5)

    np_img = np.array(image)
    # Do the processing only inside the table
    np_img_cropped = np_img[
        table_y_start:table_y_end,
        table_x_start:table_x_end,
        :,
    ]

    combined_grid, contours_v, contours_h = core_line_detection(np_img_cropped, 3, 10)

    vertical_lines = [cv2.boundingRect(cnt) for cnt in contours_v]

    horizontal_lines = [cv2.boundingRect(cnt) for cnt in contours_h]

    vertical_lines.sort(key=lambda c: c[0])
    horizontal_lines.sort(key=lambda c: c[1])

    # Add all surrounding lines
    vertical_lines.insert(0, left_line)
    horizontal_lines.insert(0, top_line)
    vertical_lines.insert(-1, right_line)
    horizontal_lines.insert(-1, bottom_line)

    ## Step 2

    def remove_line_duplicates(line_clusters, axis):
        secondary_axis = (axis + 1) % 2
        for cluster_key in line_clusters.keys():

            prev_line = line_clusters[cluster_key][0]
            filtered_lines = [line_clusters[cluster_key][0]]
            for line in line_clusters[cluster_key]:
                # Only remove lines that have the same starting point
                if abs(line[secondary_axis] - prev_line[secondary_axis]) > 10:
                    filtered_lines.append(line)

                prev_line = line
            line_clusters[cluster_key] = filtered_lines

    # Remove duplicate lines:
    vertical_clusters = find_clusters_1d_arrays(vertical_lines, 0.05 * table_bbox[3])
    horizontal_clusters = find_clusters_1d_arrays(
        horizontal_lines, 0.05 * table_bbox[2], 1, 1
    )

    remove_line_duplicates(vertical_clusters, 0)
    remove_line_duplicates(horizontal_clusters, 1)

    new_lines = [
        (line[0][0], line[0][1], line[0][0] + line[0][2], line[0][1] + line[0][3])
        for _, line in vertical_clusters.items()
    ]

    for _, lines in horizontal_clusters.items():
        for line in lines:
            new_lines.append((line[0], line[1], line[0] + line[2], line[1] + line[3]))

    new_lines_img = display_lines(new_lines, np_img_cropped)

    # 1. Combine the horizontal and vertical line images
    # table_grid = cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    # 2. Morphological Close ("Smear" the lines together)
    # This is the most important parameter to tune.
    # (15, 15) means it will connect lines that are up to 15px apart.
    kernel_size = (25, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    closed_grid = cv2.morphologyEx(new_lines_img, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        closed_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    table_bounding_boxes = []
    for cnt in contours:
        # 4. Get the bounding box for each blob
        bbox = cv2.boundingRect(cnt)
        if bbox[2] < table_bbox[2] * 0.9 or bbox[3] < table_bbox[3] * 0.95:
            table_bounding_boxes.append(
                [bbox[0] + table_bbox[0], bbox[1] + table_bbox[1], bbox[2], bbox[3]]
            )
    return table_bounding_boxes
