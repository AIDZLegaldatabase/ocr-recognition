import sys
import json
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QMessageBox, 
                             QScrollArea, QFrame, QShortcut)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeySequence
from PyQt5.QtCore import Qt, QPoint, QRect

class ImageCanvas(QLabel):
    """
    Custom widget to handle image display and drawing.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_np = None  # The raw numpy array
        self.qt_pixmap = None # The displayed QPixmap
        self.annotator = None # Reference to main window for callbacks
        
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        
        # UI settings
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setMouseTracking(True) # Track mouse even when not clicking

    def set_image(self, image_np):
        self.image_np = image_np
        height, width, channel = image_np.shape
        bytesPerLine = 3 * width
        
        # Convert Numpy to QImage
        q_img = QImage(image_np.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.qt_pixmap = QPixmap.fromImage(q_img)
        
        self.setPixmap(self.qt_pixmap)
        self.adjustSize()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_np is not None:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.image_np is not None:
            self.end_point = event.pos()
            self.update() # Trigger paintEvent

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.end_point = event.pos()
            self.is_drawing = False
            
            # Normalize coordinates (x, y, w, h)
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            # Only save if box has size
            if bbox[2] > 5 and bbox[3] > 5:
                self.annotator.add_annotation(bbox)
            
            self.start_point = None
            self.end_point = None
            self.update()

    def paintEvent(self, event):
        # 1. Draw the base image
        super().paintEvent(event)
        
        if self.image_np is None: 
            return

        painter = QPainter(self)
        
        # 2. Draw Existing Annotations
        # We access data directly from the parent logic
        current_data = self.annotator.get_current_annotations()
        
        # Draw Tables (Blue)
        pen_table = QPen(QColor(0, 0, 255), 3)
        for tbl in current_data.get("tables", []):
            x, y, w, h = tbl["bbox"]
            painter.setPen(pen_table)
            painter.drawRect(x, y, w, h)
            
            # Draw Cells (Red) inside tables
            pen_cell = QPen(QColor(255, 0, 0), 2)
            for cell in tbl.get("cells", []):
                cx, cy, cw, ch = cell
                painter.setPen(pen_cell)
                painter.drawRect(cx, cy, cw, ch)

        # 3. Draw the rectangle currently being dragged
        if self.is_drawing and self.start_point and self.end_point:
            if self.annotator.mode == "TABLE":
                painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.DashLine))
            else:
                painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
                
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)


class AnnotatorWindow(QMainWindow):
    def __init__(self, image_list, output_json="annotations.json"):
        super().__init__()
        self.setWindowTitle("PyQt Table Annotator")
        self.resize(1200, 900)
        
        self.image_list = image_list
        self.output_json = output_json
        self.current_idx = 0
        self.annotations = {}
        self.mode = "TABLE" 
        
        self.load_annotations()
        self.init_ui()
        self.load_image()

    def init_ui(self):
        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        btn_prev = QPushButton("<< Prev")
        btn_prev.clicked.connect(self.prev_image)
        btn_next = QPushButton("Next >>")
        btn_next.clicked.connect(self.next_image)
        
        self.lbl_info = QLabel(f"Page 1/{len(self.image_list)}")
        
        # Mode Buttons
        self.btn_table = QPushButton("Draw Table (Blue)")
        self.btn_table.setCheckable(True)
        self.btn_table.setChecked(True)
        self.btn_table.clicked.connect(lambda: self.set_mode("TABLE"))
        self.btn_table.setStyleSheet("background-color : lightblue")
        
        self.btn_cell = QPushButton("Draw Cell (Red)")
        self.btn_cell.setCheckable(True)
        self.btn_cell.clicked.connect(lambda: self.set_mode("CELL"))
        
        btn_clear = QPushButton("Clear Page")
        btn_clear.clicked.connect(self.clear_page)
        
        btn_save = QPushButton("Save JSON")
        btn_save.clicked.connect(self.save_annotations)
        btn_save.setStyleSheet("font-weight: bold;")

        toolbar_layout.addWidget(btn_prev)
        toolbar_layout.addWidget(self.lbl_info)
        toolbar_layout.addWidget(btn_next)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(QLabel("Mode:"))
        toolbar_layout.addWidget(self.btn_table)
        toolbar_layout.addWidget(self.btn_cell)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(btn_clear)
        toolbar_layout.addWidget(btn_save)
        
        layout.addLayout(toolbar_layout)
        
        # Scroll Area for Image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.canvas = ImageCanvas()
        self.canvas.annotator = self # Link back
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area)
        
        # Shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_annotations)
        QShortcut(QKeySequence("Right"), self).activated.connect(self.next_image)
        QShortcut(QKeySequence("Left"), self).activated.connect(self.prev_image)

    def set_mode(self, mode):
        self.mode = mode
        if mode == "TABLE":
            self.btn_table.setChecked(True)
            self.btn_cell.setChecked(False)
            self.btn_table.setStyleSheet("background-color : lightblue")
            self.btn_cell.setStyleSheet("")
        else:
            self.btn_table.setChecked(False)
            self.btn_cell.setChecked(True)
            self.btn_table.setStyleSheet("")
            self.btn_cell.setStyleSheet("background-color : #ffcccc")

    def load_image(self):
        img_data = self.image_list[self.current_idx]
        # Ensure contiguous array for Qt
        if not img_data.flags['C_CONTIGUOUS']:
            img_data = np.ascontiguousarray(img_data)
        
        self.canvas.set_image(img_data)
        self.lbl_info.setText(f"Page {self.current_idx + 1}/{len(self.image_list)}")

    def get_current_annotations(self):
        return self.annotations.get(str(self.current_idx), {})

    def add_annotation(self, bbox):
        idx_str = str(self.current_idx)
        if idx_str not in self.annotations:
            self.annotations[idx_str] = {"tables": []}
            
        if self.mode == "TABLE":
            self.annotations[idx_str]["tables"].append({"bbox": bbox, "cells": []})
        
        elif self.mode == "CELL":
            # Assign cell to the table containing it
            cx, cy, cw, ch = bbox
            center_x, center_y = cx + cw/2, cy + ch/2
            
            found = False
            for table in self.annotations[idx_str]["tables"]:
                tx, ty, tw, th = table["bbox"]
                if tx <= center_x <= tx + tw and ty <= center_y <= ty + th:
                    table["cells"].append(bbox)
                    found = True
                    break
            
            if not found:
                QMessageBox.warning(self, "Warning", "Cells must be drawn inside an existing Table.")

    def next_image(self):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def clear_page(self):
        if str(self.current_idx) in self.annotations:
            del self.annotations[str(self.current_idx)]
            self.canvas.update()

    def save_annotations(self):
        with open(self.output_json, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Saved to {self.output_json}")
        self.statusBar().showMessage(f"Saved to {self.output_json}", 2000)

    def load_annotations(self):
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.annotations = json.load(f)

# Wrapper to run the tool
def run_annotator(image_list, output_path):
    app = QApplication(sys.argv)
    window = AnnotatorWindow(image_list, output_path)
    window.show()
    sys.exit(app.exec_())

