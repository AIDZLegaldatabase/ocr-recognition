from dataclasses import dataclass
import cv2
import numpy as np
from typing import List, Dict
from itertools import chain
import pprint
import logging
import uuid

# Next step: better debugging in the report
# Use the module name so we know where logs come from
logger = logging.getLogger(__name__)


@dataclass
class TableLine:
    bbox: List[float]
    _contour: List[float]

    def __init__(self, contour_or_bbox, init_from_bbox=False):
        if init_from_bbox:
            self._contour = []
            self.bbox = contour_or_bbox
        else:
            self._contour = contour_or_bbox
            self.bbox = cv2.boundingRect(contour_or_bbox)

    @property
    def x(self):
        return self.bbox[0]

    @property
    def y(self):
        return self.bbox[1]

    @property
    def length(self):
        return self.bbox[2] if self.bbox[2] > self.bbox[3] else self.bbox[3]

    def is_horizontal(self):
        return self.bbox[2] > self.bbox[3]

    def is_vertical(self):
        return self.bbox[3] > self.bbox[2]

    def has_contour(self):
        return len(self.contour) > 0

    @property
    def contour(self):
        return self._contour

    def __repr__(self):
        return pprint.pformat(self.bbox)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                (self.x == other.x)
                and (self.y == other.y)
                and (self.length == other.length)
                and (self.is_vertical() == other.is_vertical())
                and (self.is_horizontal() == other.is_horizontal())
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    @property
    def center(self):
        if self.is_vertical():
            return self.bbox[1] + self.bbox[3] // 2
        elif self.is_horizontal():
            return self.bbox[0] + self.bbox[2] // 2
        else:
            raise ValueError(
                f"Bounding box for line is neither vertical or horizontal: {self.bbox}"
            )


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
    logger.debug(
        f"Now we printed back the lines on the image, smeared lines together with the kernel {kernel_size}."
        f"Expected bounding boxes number:" + pprint.pformat(f"{len(contours)}")
    )
    for cnt in contours:
        # 4. Get the bounding box for each blob
        bbox = cv2.boundingRect(cnt)
        if bbox[2] > TABLE_MIN_WIDTH and bbox[3] > TABLE_MIN_HEIGHT:
            bounding_boxes.append(bbox)
        else:
            logger.debug(f"Size filtering removed bbox: {bbox}")
    logger.debug(
        f"Filtered with the sizes ({TABLE_MIN_WIDTH}, {TABLE_MIN_HEIGHT})"
        f"Bounding boxes candidates:\n" + pprint.pformat(f"{bounding_boxes}")
    )
    return bounding_boxes


def core_line_detection(img, kernel_size, min_line_ratio, close_gaps=False, close_gaps_kernel_size=10):
    """_summary_

    Args:
        img (_type_): _description_
        kernel_size (_type_): _description_
        min_line_ratio (_type_): Anything shorter than min_line_ratio% of the image width is noise
        close_gaps (bool): only use when detecting cells

    Returns:
        _type_: _description_
    """
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
    horizontal_kernel_len = int(gray.shape[1] * min_line_ratio)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))

    vertical_kernel_len = int(gray.shape[0] * min_line_ratio)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    morphed_horizontal = cv2.morphologyEx(
        thresh_y.astype(np.uint8), cv2.MORPH_OPEN, hor_kernel
    )
    morphed_vertical = cv2.morphologyEx(
        thresh_x.astype(np.uint8), cv2.MORPH_OPEN, ver_kernel
    )
    if close_gaps:
        kernel_size = (close_gaps_kernel_size, close_gaps_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        # "Closing" = Dilate (thicken) then Erode (thin)
        # This fills gaps and connects nearby lines.
        morphed_vertical = cv2.morphologyEx(morphed_vertical, cv2.MORPH_CLOSE, kernel)

        # "Closing" = Dilate (thicken) then Erode (thin)
        # This fills gaps and connects nearby lines.
        morphed_horizontal = cv2.morphologyEx(
            morphed_horizontal, cv2.MORPH_CLOSE, kernel
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

    contours_v_lines = [
        TableLine(cnt)
        for cnt in contours_v
        if cv2.boundingRect(cnt)[2] != cv2.boundingRect(cnt)[3]
    ]
    contours_h_lines = [
        TableLine(cnt)
        for cnt in contours_h
        if cv2.boundingRect(cnt)[2] != cv2.boundingRect(cnt)[3]
    ]
    return combined_grid, contours_v_lines, contours_h_lines


def filter_central_v_line(contours_v, img_width):
    CENTRE_LINE_TOLERANCE = 100
    CLSUTERS_GAP_THRESHOLD = 10
    vertical_lines_clusters = find_lines_clusters(
        contours_v,
        gap_threshold=CLSUTERS_GAP_THRESHOLD,
        min_cluster_size=1,
    )

    # Find the cluster of lines in the middle of the page:
    center_matching_clusters = [
        vertical_lines_clusters[key]
        for key in vertical_lines_clusters.keys()
        if all(
            (img_width / 2) - CENTRE_LINE_TOLERANCE
            < elt.x
            < (img_width / 2) + CENTRE_LINE_TOLERANCE
            for elt in vertical_lines_clusters[key]
        )
    ]
    # 2. Flatten them into one big list
    centre_lines = list(chain.from_iterable(center_matching_clusters))
    if len(center_matching_clusters) > 1:
        logger.debug(
            f"Detected the following lines as central and deleting them:\n"
            + pprint.pformat(f"{centre_lines}")
        )

    # If there's a central cluster then ommit any line item which has the x of
    # the bounding box in the list of centre lines coordinates
    contours_v_without_central = (
        contours_v
        if len(centre_lines) == 0
        else [line for line in contours_v if line not in centre_lines]
    )

    return contours_v_without_central


def detect_table_from_image_data(img: np.ndarray):
    """
    Detect tables within an image, returning both a list of
    table bounding boxes and a debug image

    Args:
        img: A NumPy array (OpenCV image)
    """
    if img is None:
        logger.error("Error: Invalid image data.")
        return False
    # PARAMS
    MIN_HORIZONTAL_LINES = 1
    MIN_VERTICAL_LINES = 1
    NUM_TOTAL_LINES = 4
    LINE_MINIMAL_HEIGHT_RATIO = 0.087
    LINE_MINIMAL_WIDTH_RATIO = 0.137
    IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO = 0.0048
    IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO = 0.003
    LINE_DETECTION_KERNEL_SIZE = 5
    LINE_LENGTH_RATIO_MIN = 0.05

    image_height, image_width, _ = img.shape
    logger.debug(
        f"Processing image with dimensions: width-{image_width}, height-{image_height}"
    )

    # Perform the detection in the main function
    combined_grid, vertical_lines, horizontal_lines = core_line_detection(
        img, LINE_DETECTION_KERNEL_SIZE, LINE_LENGTH_RATIO_MIN
    )

    # Validate number of countours

    # Filter lines by size:
    horizontal_lines = [
        line
        for line in horizontal_lines
        if (
            line.length > (LINE_MINIMAL_WIDTH_RATIO * image_width)
            and (IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO * image_height)
            < line.y
            < (image_height * (1 - IMAGE_Y_BORDERS_CROP_TOLERANCE_RATIO))
        )
    ]

    vertical_lines = [
        line
        for line in vertical_lines
        if (
            line.length > LINE_MINIMAL_HEIGHT_RATIO
            and (IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO * image_width)
            < line.x
            < (image_width * (1 - IMAGE_X_BORDERS_CROP_TOLERANCE_RATIO))
        )
    ]
    logger.debug(
        f"Performed line detection and size filtering and here are the outputs:\n"
        f"Vertical Lines:"
        + pprint.pformat(f"{vertical_lines}")
        + f"\nHorizontal Lines:"
        + pprint.pformat(f"{horizontal_lines}")
    )

    # Remove central line from vertical lines
    vertical_lines = filter_central_v_line(vertical_lines, image_width)

    # Create new image with combined grids
    mask = np.zeros(combined_grid.shape[:2], dtype=np.uint8)
    for line in horizontal_lines:
        cv2.drawContours(mask, [line.contour], -1, (255), cv2.FILLED)
    for line in vertical_lines:
        cv2.drawContours(mask, [line.contour], -1, (255), cv2.FILLED)
    img_grid = cv2.bitwise_and(combined_grid, combined_grid, mask=mask)
    # Find all the bounding boxes around group of lines
    table_boxes = find_table_bounding_boxes(img_grid)
    output_boxes = {}

    # Filter the bounding boxes by the number of lines inside each one
    for bbox in table_boxes:
        (bx, by, bw, bh) = bbox
        cv2.rectangle(img_grid, (bx, by), (bx + bw, by + bh), 255, 3)
        h_count = 0
        v_count = 0

        # Check horizontal lines
        for line in horizontal_lines:
            # Check if the line's center is inside the table's box
            if (
                (bx < line.center < bx + bw)
                and (by < line.y < by + bh)
                and (by + 15 < line.y < by + bh - 15)
            ):
                h_count += 1

        # Check vertical lines
        for line in vertical_lines:
            # The last check is only done in vertical lines because they're continuous
            # usually so we can filter small vertical lines
            # TODO add similar check for verticals, you want to see if a line is floating and not connected anywhere
            # when you're looking for bboxes
            if (
                (bx < line.x < bx + bw)
                and (by < line.center < by + bh)
                and (line.length / bh > 0.5)
                and (bx + 15 < line.x < bx + bw - 15)
            ):
                v_count += 1
        if (h_count >= MIN_HORIZONTAL_LINES and v_count >= MIN_VERTICAL_LINES) or (
            h_count + v_count > NUM_TOTAL_LINES
        ):
            box_uuid = uuid.uuid1()
            output_boxes[box_uuid] = bbox
            logger.debug(
                f"Found a table bounding box with {h_count} horizontal"
                f"lines and {v_count} vertical lines with coordinates: {bbox} and id: {box_uuid}"
            )
        cv2.putText(
            img_grid,
            "Table detection debug",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,5,
            (255),
            1,
        )

    return output_boxes, img_grid


def find_lines_clusters(
    data: List[TableLine], gap_threshold: float, min_cluster_size: int = 1
):
    """
    Finds clusters in a list of lines based on a simple
    gap threshold.

    Args:
        data: A list of sorted lines list.
        gap_threshold: The maximum gap to allow *inside* a cluster.
        min_cluster_size: The minimum number of items to be
                          considered a "real" cluster.

    Returns:
        A dictionary where keys are cluster IDs (0, 1, 2...)
        and values are the lists of numbers in that cluster.
    """
    if not data:
        return {}

    # Sort the data, infer all the lines orientation by using the first one
    # TODO: Verify this assumption
    if data[0].is_vertical():
        data.sort(key=lambda line: line.x)
    elif data[0].is_horizontal():
        data.sort(key=lambda line: line.y)

    clusters = {}
    current_cluster_id = 0
    current_cluster = [data[0]]

    # Step 2 & 3: Iterate and find gaps
    for i in range(len(data) - 1):
        # Calculate the gap
        if data[i].is_vertical():
            gap = data[i + 1].x - data[i].x
        elif data[i].is_horizontal():
            gap = data[i + 1].y - data[i].y
        else:
            raise ValueError(f"Line not horizontal or vertical: {data[i]}")

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
    final_clusters = clusters

    return final_clusters


def display_lines(lines, image):
    # Create new image with combined grids
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for line in lines:
        length_x = line.length if line.is_horizontal() else 5
        length_y = line.length if line.is_vertical() else 5
        cv2.rectangle(
            mask,
            (round(line.x), round(line.y)),
            (round(line.x + length_x), round(line.y + length_y)),
            255,
            -1,
        )

    return mask


def remove_line_duplicates(line_clusters, tolerance=5):
    for cluster_key in line_clusters.keys():
        if len(line_clusters[cluster_key]) == 1:
            continue
        line_cluster = line_clusters[cluster_key]
        line_cluster.sort(key=lambda l: l.y if l.is_vertical() else l.x)
        prev_line = line_cluster[0]
        filtered_lines = [line_cluster[0]]
        for i, line in enumerate(line_cluster):
            if i == 0:
                continue
            if line.is_vertical():
                # Only remove lines that have the same starting point
                if abs(line.y - prev_line.y) > tolerance:
                    filtered_lines.append(line)
            elif line.is_horizontal():
                # Only remove lines that have the same starting point
                if abs(line.x - prev_line.x) > tolerance:
                    filtered_lines.append(line)
            prev_line = line
        line_clusters[cluster_key] = filtered_lines


def detect_table_cells(image, table_bbox):
    table_x_start = table_bbox[0]
    table_y_start = table_bbox[1]
    table_x_end = table_bbox[0] + table_bbox[2]
    table_y_end = table_bbox[1] + table_bbox[3]
    left_line = TableLine([0, 0, 5, table_bbox[3]], init_from_bbox=True)
    top_line = TableLine([0, 0, table_bbox[2], 5], init_from_bbox=True)
    right_line = TableLine(
        [table_bbox[2] - 10, 0, 5, table_bbox[3]], init_from_bbox=True
    )
    bottom_line = TableLine(
        [0, table_bbox[3] - 10, table_bbox[2], 5], init_from_bbox=True
    )

    np_img = np.array(image)
    # Do the processing only inside the table
    np_img_cropped = np_img[
        table_y_start:table_y_end,
        table_x_start:table_x_end,
        :,
    ]

    _, vertical_lines, horizontal_lines = core_line_detection(np_img_cropped, 3, 0.1, close_gaps=True)

    vertical_lines.sort(key=lambda line: line.x)
    horizontal_lines.sort(key=lambda line: line.y)

    # Add all surrounding lines
    vertical_lines.insert(0, left_line)
    horizontal_lines.insert(0, top_line)
    vertical_lines.insert(-1, right_line)
    horizontal_lines.insert(-1, bottom_line)

    ## Step 2

    def get_minimal_line_diff_distance(v_lines, h_lines):
        assert len(h_lines) > 1 and len(v_lines) > 1
        prev_line = v_lines[0]
        # random number to start with
        current_min = 1000
        for i, line in v_lines.items():
            if i == 0:
                continue
            current_min = min(abs(prev_line[0].x - line[0].x), current_min)
            prev_line = line
        prev_line = h_lines[0]
        for i, line in h_lines.items():
            if i == 0:
                continue
            current_min = min(abs(prev_line[0].y - line[0].y), current_min)
            prev_line = line
        return current_min

    # Remove duplicate lines:
    vertical_clusters = find_lines_clusters(vertical_lines, 0.01 * table_bbox[3])
    horizontal_clusters = find_lines_clusters(horizontal_lines, 0.01 * table_bbox[2])
    logger.debug(
        f"At the start, the following line clusters are detected: \n"
        "Vertical Clusters: "
        + pprint.pformat(f"{vertical_clusters}")
        + "\nHorizontal Clusters: "
        + pprint.pformat(f"{horizontal_clusters}")
    )

    remove_line_duplicates(vertical_clusters)
    remove_line_duplicates(horizontal_clusters)

    logger.debug(
        f"After removing duplicates: \n"
        "Vertical Clusters: "
        + pprint.pformat(f"{vertical_clusters}")
        + "\nHorizontal Clusters: "
        + pprint.pformat(f"{horizontal_clusters}")
    )

    # 1. Chain the values of both dicts together
    # 2. Flatten them from the resulting iterable
    all_lines = list(
        chain.from_iterable(
            chain(vertical_clusters.values(), horizontal_clusters.values())
        )
    )

    new_lines_img = display_lines(all_lines, np_img_cropped)

    min_line_distance = get_minimal_line_diff_distance(
        vertical_clusters, horizontal_clusters
    )

    # 1. Combine the horizontal and vertical line images
    # table_grid = cv2.bitwise_or(morphed_horizontal, morphed_vertical)

    # 2. Morphological Close ("Smear" the lines together)
    # This is the most important parameter to tune.
    # (25, 25) means it will connect lines that are up to 25px apart.
    # Expirementation showed that using min distance between lines minus
    # a gap works the best
    if min_line_distance <= 10:
            min_line_distance = 15
    kernel_size = (abs(min_line_distance - 10), abs(min_line_distance - 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # "Closing" = Dilate (thicken) then Erode (thin)
    # This fills gaps and connects nearby lines.
    closed_grid = cv2.morphologyEx(new_lines_img, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        closed_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    logger.debug(
        f"With kernel {min_line_distance}, {len(contours)} cell candidates were found"
    )
    table_bounding_boxes = []
    for cnt in contours:
        # 4. Get the bounding box for each blob
        bbox = cv2.boundingRect(cnt)
        if bbox[3] / table_bbox[3] < 0.02 or bbox[2] / table_bbox[2] < 0.02:
            continue
        if bbox[2] < table_bbox[2] * 0.95 or bbox[3] < table_bbox[3] * 0.95:
            table_bounding_boxes.append(
                [bbox[0] + table_bbox[0], bbox[1] + table_bbox[1], bbox[2], bbox[3]]
            )
    logger.debug(
        f"After size filtering, {len(table_bounding_boxes)} cells were admitted"
    )
    return table_bounding_boxes, closed_grid
