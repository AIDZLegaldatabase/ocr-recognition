import os
import json


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_layout_boxes(layout_boxes, figsize=(15, 20), colors=None, line_width=1.0, alpha=0.7):
    """
    Visualize a list of layout boxes on a canvas.
    
    Parameters:
    - layout_boxes: A list of bounding boxes in format [x_start, y_start, x_end, y_end]
                    or a string representation of such a list
    - figsize: Size of the figure (width, height)
    - colors: Optional color map or list of colors for boxes
    - line_width: Width of the box borders
    - alpha: Transparency of the boxes
    
    Returns:
    - matplotlib figure object
    """

    # Ensure layout_boxes is a list of lists
    if not isinstance(layout_boxes, list):
        raise ValueError("layout_boxes must be a list of bounding boxes")
    
    # Calculate document dimensions
    x_coords = [coord for bbox in layout_boxes for coord in [bbox[0], bbox[2]]]
    y_coords = [coord for bbox in layout_boxes for coord in [bbox[1], bbox[3]]]
    x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
    y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colormap if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(layout_boxes)))
    elif isinstance(colors, str):
        # If a single color is provided
        colors = [colors] * len(layout_boxes)
    
    # Draw each bounding box
    for i, (bbox, color) in enumerate(zip(layout_boxes, colors)):
        x_start, y_start, x_end, y_end = bbox
        width = x_end - x_start
        height = y_end - y_start
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_start, y_start), width, height,
            linewidth=line_width, edgecolor=color, facecolor='none', alpha=alpha
        )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add index number at the top-left corner of each box
        ax.text(
            x_start + 5, y_start + 15, 
            f"{i}", 
            color='black', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert y-axis to match document coordinates
    
    # Set title and labels
    ax.set_title("Document Layout Visualization", fontsize=16)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Add grid for reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a legend showing the reading order
    legend_elements = []
    # Create legend in chunks to avoid overcrowding
    chunk_size = 10
    for chunk_start in range(0, min(50, len(layout_boxes)), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(layout_boxes))
        for i in range(chunk_start, chunk_end):
            if i < len(layout_boxes):
                legend_elements.append(
                    patches.Patch(facecolor='none', edgecolor=colors[i], label=f"Box {i}")
                )
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(1.15, 1), fontsize=8)
    
    plt.tight_layout()
    return fig

def visualize_bbox_text(layout, figsize=(12, 8), show_text=True, show_indices=True):
    """
    Visualize the bounding boxes of text elements on a canvas.
    
    Parameters:
    - layout: Dictionary containing 'bbox_text' and 'text' keys
    - figsize: Size of the figure (width, height)
    - show_text: Whether to display the actual text content
    - show_indices: Whether to show indices for each bbox
    
    Returns:
    - matplotlib figure object
    """
    # Extract bbox_text and text from layout
    bbox_text = layout['bbox_text']
    text = layout['text']
    
    # Get the overall document dimensions from bbox_layout if available
    if 'bbox_layout' in layout:
        x_min, y_min, x_max, y_max = layout['bbox_layout']
    else:
        # Otherwise, calculate from all bboxes with some padding
        x_coords = [coord for bbox in bbox_text for coord in [bbox[0], bbox[2]]]
        y_coords = [coord for bbox in bbox_text for coord in [bbox[1], bbox[3]]]
        x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
        y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert y-axis to match document coordinates
    
    # Set title
    ax.set_title("OCR Bounding Boxes Visualization")
    
    # Colors for different bboxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(bbox_text)))
    
    # Draw each bounding box
    for i, (bbox, txt, color) in enumerate(zip(bbox_text, text, colors)):
        x_start, y_start, x_end, y_end = bbox
        width = x_end - x_start
        height = y_end - y_start
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_start, y_start), width, height,
            linewidth=1, edgecolor=color, facecolor='none', alpha=0.8
        )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add index number
        if show_indices:
            ax.text(
                x_start, y_start - 5, 
                f"{i}", 
                color='black', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
            )
        
        # Add text content
        if show_text:
            text_preview = txt[:20] + "..." if len(txt) > 20 else txt
            ax.text(
                x_start, y_start - 15, 
                text_preview, 
                color='black', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
            )
    
    # Add a legend for original ordering
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=colors[i], label=f"Box {i}") 
        for i in range(len(bbox_text))
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Add grid for reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add axis labels
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    plt.tight_layout()
    return fig

# Example usage:
# fig = visualize_bbox_text(layout)
# plt.show()
# To save: fig.savefig('bbox_visualization.png', dpi=300)

def reorder_ocr_text(layout):
    # Extract bbox_text and text from layout
    bbox_text = layout['bbox_text']
    text = layout['text']
    
    # Create paired data for sorting
    paired_data = list(zip(bbox_text, text))
    
    # Calculate average text height to determine tolerance
    heights = [bbox[3] - bbox[1] for bbox in bbox_text]  # y_end - y_start for each bbox
    avg_height = sum(heights) / len(heights) if heights else 1  # Default to 20 if no data
    
    # Set tolerance to 0.3 of average text height
    y_tolerance = 0.3 * avg_height
    
    def sort_key(item):
        bbox = item[0]  # bbox is [x_start, y_start, x_end, y_end]
        y_start = bbox[1]  # Get y_start
        x_start = bbox[0]  # Get x_start
        
        # Round y_start to nearest multiple of tolerance to group lines
        y_group = round(y_start / y_tolerance) * y_tolerance
        
        # Return tuple for sorting - first by rounded y, then by x
        return (y_group, x_start)
    
    # Sort the paired data
    sorted_data = sorted(paired_data, key=sort_key)
    
    # Extract the sorted components
    sorted_bbox_text = [item[0] for item in sorted_data]
    sorted_text = [item[1] for item in sorted_data]
    
    # Create new layout with sorted components
    new_layout = layout.copy()
    new_layout['bbox_text'] = sorted_bbox_text
    new_layout['text'] = sorted_text
    
    return new_layout

# Path to the directory containing yearly result_json folders
base_dir = "result_json"

# Iterate through each year folder
for year in sorted(os.listdir(base_dir), reverse=False):
    year_path = os.path.join(base_dir, year)
    
    if os.path.isdir(year_path):  # Ensure it's a directory
        print(f"Processing year: {year}")

        # Iterate through each JSON file in the year folder
        for filename in sorted(os.listdir(year_path)):
            if filename.endswith(".json"):
                file_path = os.path.join(year_path, filename)
                print(f"Processing file: {filename}")

                # Read the JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                #print("NB pages :" + str(len(data)))
                for page in data:
                    #print("NB boxes :" + str(len(page['page'])))
                    # page number 12
                    layouts_org = [layout['bbox_layout'] for layout in page['page']]
                    layout_data = [layout for layout in page['page']]
                    layout_peek = [layout for layout in layout_data 
                                   if ((layout['label'] == 'Text') and
                                        ((layout['bbox_layout'][2] - layout['bbox_layout'][0]) > 900) and 
                                        ((layout['bbox_layout'][3] - layout['bbox_layout'][1]) > 800))]
                    if (len(layout_peek) > 0):
                        print(filename)
                        print(page['index'])
                        visualize_layout_boxes(layouts_org)
                        visualize_bbox_text(page['page'][0])
                        plt.show()
                        new_layout = reorder_ocr_text(page['page'][0])['text']
                        print(new_layout)

                    """
                    if (page['index'] == 9):
                        
                        layouts_org = [layout['bbox_layout'] for layout in page['page']]
                        visualize_layout_boxes(layouts_org)
                        plt.show()
                        for layout in page['page']:
                            new_layout = reorder_ocr_text(layout)
                            #print(new_layout)
                            #fig = visualize_bbox_text(layout)
                            #plt.show()
                        exit()"
                    """

                        
                # Compute the total text size
                #total_text_length = sum(len(" ".join(page['page']["text"])) for page in data)

                # Print the result
                #print(f"{filename}: {total_text_length} characters")