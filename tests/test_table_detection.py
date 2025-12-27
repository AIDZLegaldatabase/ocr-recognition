import pytest
import numpy as np
import cv2
from table_detection.table_detection import (
    TableLine,
    find_clusters_1d,
    find_lines_clusters,
    find_table_bounding_boxes,
    core_line_detection,
    filter_central_v_line,
    remove_line_duplicates,
)

# --- Tests for TableLine Class ---


def test_table_line_properties():
    # Test horizontal line
    bbox = [10, 20, 100, 5]  # x, y, w, h
    line = TableLine(bbox, init_from_bbox=True)
    assert line.is_horizontal() is True
    assert line.is_vertical() is False
    assert line.length == 100
    assert line.center == 10 + (100 // 2)


def test_table_line_vertical():
    bbox = [50, 50, 10, 200]
    line = TableLine(bbox, init_from_bbox=True)
    assert line.is_vertical() is True
    assert line.center == 50 + (200 // 2)


# --- Tests for Clustering Functions ---


def test_find_clusters_1d():
    data = [1, 2, 10, 11, 12, 30]
    # With gap 5: [1,2] and [10,11,12] and [30]
    clusters = find_clusters_1d(data, gap_threshold=5)
    assert len(clusters) == 3
    assert clusters[0] == [1, 2]
    assert clusters[2] == [30]


def test_find_lines_clusters():
    # Create dummy TableLines
    line1 = TableLine([0, 10, 100, 5], init_from_bbox=True)  # y=10
    line2 = TableLine([0, 12, 100, 5], init_from_bbox=True)  # y=12
    line3 = TableLine([0, 50, 100, 5], init_from_bbox=True)  # y=50

    data = [line1, line2, line3]
    clusters = find_lines_clusters(data, gap_threshold=10)

    assert len(clusters) == 2
    assert len(clusters[0]) == 2  # line1 and line2 are close
    assert len(clusters[1]) == 1  # line3 is far


# --- Tests for Image Processing Functions ---


def test_find_table_bounding_boxes():
    # Create a blank black image
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw a white rectangle (a fake table grid)
    # x, y, w, h -> (200, 200, 200, 150)
    cv2.rectangle(img, (50, 50), (300, 300), 255, -1)

    bboxes = find_table_bounding_boxes(img)

    assert len(bboxes) == 1
    x, y, w, h = bboxes[0]
    assert w > 185  # Check against your TABLE_MIN_WIDTH
    assert h > 100  # Check against your TABLE_MIN_HEIGHT


# --- Edge Cases ---


def test_find_clusters_empty():
    assert find_clusters_1d([], 10) == {}
    assert find_lines_clusters([], 10) == {}


# --- Test Core Line Detection (Synthetic Image) ---


def test_core_line_detection_synthetic():
    # 1. Setup: Create a 500x500 black image
    img_size = 500
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Define params used in the function's morphological step
    # min_line_ratio = 0.1 means lines must be > 50px long
    min_line_ratio = 0.1

    # 2. Draw Synthetic Lines
    # Draw a long Horizontal line (should be detected)
    # Start(50, 100), End(450, 100). Length 400. Thickness 2.
    cv2.line(img, (50, 200), (450, 200), (255, 255, 255), 2)

    # Draw a long Vertical line (should be detected)
    # Start(250, 50), End(250, 450). Length 400. Thickness 2.
    cv2.line(img, (250, 50), (250, 450), (255, 255, 255), 2)

    # Draw a tiny "noise" line (should be filtered out by morph operations)
    # Length 20 (which is < 50px threshold)
    cv2.line(img, (10, 10), (30, 10), (255, 255, 255), 2)

    # Both of lines cross each other

    # 3. Execute function
    # Note: without close gaps we have 4 horizontal and 4 vertical lines
    grid, v_lines, h_lines = core_line_detection(
        img, kernel_size=3, min_line_ratio=min_line_ratio, close_gaps=True
    )

    # 4. Assertions
    # Should find exactly 1 horizontal and 1 vertical line
    assert len(h_lines) == 1
    assert len(v_lines) == 1

    # Verify properties of detected horizontal line
    h_line = h_lines[0]
    assert h_line.is_horizontal()
    # Width should be approx 400, height very small
    assert h_line.bbox[2] > 350
    assert h_line.bbox[3] < 10

    # Verify properties of detected vertical line
    v_line = v_lines[0]
    assert v_line.is_vertical()
    # Height should be approx 400, width very small
    assert v_line.bbox[3] > 350
    assert v_line.bbox[2] < 10

    # Verify the combined grid is not empty
    assert cv2.countNonZero(grid) > 0


# --- Tests for Central Vertical Line Filtering ---


def test_filter_central_v_line_removes_center():
    """Test scenario where central lines exist and should be removed."""
    img_width = 1000
    # Center is 500. The function has a tolerance of 100 (+/- 100 from center).
    # Range to remove: 400 to 600.

    # Create mock vertical lines using init_from_bbox=[x, y, w, h]
    # Far left line (keep)
    line_left = TableLine([100, 0, 5, 500], init_from_bbox=True)
    # Far right line (keep)
    line_right = TableLine([900, 0, 5, 500], init_from_bbox=True)
    # Central line 1 (remove) - x=490
    line_c1 = TableLine([490, 0, 5, 500], init_from_bbox=True)
    # Central line 2 (remove) - x=510 (part of same cluster as 490)
    line_c2 = TableLine([510, 0, 5, 500], init_from_bbox=True)

    # Input list MUST be sorted by X for find_lines_clusters to work
    input_lines = [line_left, line_c1, line_c2, line_right]

    filtered_lines = filter_central_v_line(input_lines, img_width)

    # Should have removed the two central lines, leaving 2
    assert len(filtered_lines) == 2
    # Ensure the remaining lines are the correct ones
    assert filtered_lines[0].x == 100
    assert filtered_lines[1].x == 900


def test_filter_central_v_line_no_center_detected():
    """Test scenario where no lines are near the center."""
    img_width = 1000
    # Range to remove: 400 to 600.

    # Line at x=200 (keep)
    line1 = TableLine([200, 0, 5, 500], init_from_bbox=True)
    # Line at x=300 (keep - outside tolerance)
    line2 = TableLine([300, 0, 5, 500], init_from_bbox=True)

    input_lines = [line1, line2]

    filtered_lines = filter_central_v_line(input_lines, img_width)

    # Should keep both lines
    assert len(filtered_lines) == 2
    assert filtered_lines[0] == line1
    assert filtered_lines[1] == line2


def test_remove_line_duplicates():
    # 1. Setup Mock Lines
    # Cluster 0: Two vertical lines starting at almost the same Y (Duplicates)
    v_dup1 = TableLine([100, 50, 5, 200], init_from_bbox=True)  # x=100, y=50
    v_dup2 = TableLine([102, 51, 5, 200], init_from_bbox=True)  # x=102, y=51

    # Cluster 1: Two vertical lines starting at different Y (Distinct segments in same column)
    v_dist1 = TableLine([300, 50, 5, 100], init_from_bbox=True)  # y=50
    v_dist2 = TableLine([300, 250, 5, 100], init_from_bbox=True)  # y=250

    # Cluster 2: Horizontal duplicates
    h_dup1 = TableLine([50, 100, 200, 5], init_from_bbox=True)  # x=50
    h_dup2 = TableLine([51, 100, 200, 5], init_from_bbox=True)  # x=51

    clusters = {0: [v_dup1, v_dup2], 1: [v_dist1, v_dist2], 2: [h_dup1, h_dup2]}

    # 2. Execute
    remove_line_duplicates(clusters, tolerance=5)

    # 3. Assertions
    # Cluster 0 should only have 1 line left
    assert len(clusters[0]) == 1
    assert clusters[0][0].y == 50

    # Cluster 1 should still have 2 lines (they are 200px apart)
    assert len(clusters[1]) == 2

    # Cluster 2 should only have 1 line left
    assert len(clusters[2]) == 1
    assert clusters[2][0].x == 50
