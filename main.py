import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt 
from PIL import ImageGrab
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
import numpy as np
import torch
import easyocr
from PIL import ImageDraw, ImageFont

class scripts:
    def function1(self):
        pass

    def function2(self):
        pass

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("untitled.ui", self)

        self.screen_capture_pushButton.clicked.connect(self.display_screenshot)
        self.add_scripts_methods()
        self.run_model_pushButton.clicked.connect(self.run_model)
        self.ai_module_comboBox.currentIndexChanged.connect(self.on_ai_module_changed)
        self.on_ai_module_changed()

    def run_model(self):
        current_model = self.ai_module_comboBox.currentText()
        screenshot = ImageGrab.grab()

        if current_model == "yolov5":
            weight_fileName = self.custom_train_data_ListWidget.currentItem().text()
            screenshot_np = np.array(screenshot)
            self.process_with_yolov5(screenshot_np, weight_fileName)

        elif current_model == "easy_ocr":
            self.process_with_easyocr(screenshot)

    def process_with_yolov5(self, np_image, weight_fileName):
        model_path = f'E:/ProjectGit/Detect_Inappropriate_Image/weights/exp1/{weight_fileName}'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        model.conf = 0.25  # confidence threshold
        model.imgsz = 640  # image size

        results = model(np_image)
        result_np  = results.render()[0]
        h, w, ch = result_np.shape
        bytes_per_line = ch * w
        result_qimage = QImage(result_np.data, w, h, bytes_per_line, QImage.Format_RGB888)

        result_pixmap = QPixmap.fromImage(result_qimage)
        max_size = self.graphicsView.maximumSize()
        scaled_pixmap = result_pixmap.scaled(max_size, Qt.KeepAspectRatio)

        pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

        scene = QGraphicsScene()
        scene.addItem(pixmap_item)
        self.graphicsView.setScene(scene)
    
    def process_with_easyocr(self, pil_image):
        reader = easyocr.Reader(['en'])
        np_image = np.array(pil_image)

        results = reader.readtext(np_image)
        draw = ImageDraw.Draw(pil_image)

        for result in results:
            top_left = tuple(result[0][0])
            bottom_right = tuple(result[0][2])
            text = result[1]
            draw.rectangle([top_left, bottom_right], outline='red')
            draw.text(top_left, text, fill='red')
        
        result_qimage = self.convert_pil_to_qt(pil_image)
        result_pixmap = QPixmap.fromImage(result_qimage)

        max_width = self.graphicsView.maximumWidth()
        max_height = self.graphicsView.maximumHeight()
        scaled_pixmap = result_pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)

        pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

        scene = QGraphicsScene()
        scene.addItem(pixmap_item)

        self.graphicsView.setScene(scene)
 

    def on_ai_module_changed(self):
        if self.ai_module_comboBox.currentText() == "yolov5":
            self.load_weights_files()
        else:
            self.custom_train_data_ListWidget.clear()

    def load_weights_files(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        weights_folder = os.path.join(current_directory, 'weights', 'exp1')

        try:
            files = os.listdir(weights_folder)
            for file_name in files:
                self.custom_train_data_ListWidget.addItem(file_name)
        except FileNotFoundError:
            print("weights 폴더를 찾을 수 없습니다.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F10:
            self.display_screenshot()
        else:
            super().keyPressEvent(event)

    def add_scripts_methods(self):
        methods = [func for func in dir(scripts) if callable(getattr(scripts, func)) and not func.startswith("__")]
        for method in methods:
            self.scripts_comboBox.addItem(method)

    def display_screenshot(self):
        screenshot = ImageGrab.grab()
        screenshot_qt = self.convert_pil_to_qt(screenshot)

        pixmap = QPixmap.fromImage(screenshot_qt)

        max_size = self.graphicsView.maximumSize()
        scaled_pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio)

        scene = QtWidgets.QGraphicsScene(self)
        scene.addPixmap(scaled_pixmap)
        self.graphicsView.setScene(scene)

    def convert_pil_to_qt(self, image):
        data = image.tobytes("raw", "RGB")
        qimage = QImage(data, image.size[0], image.size[1], QImage.Format_RGB888)
        return qimage


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.show()
    sys.exit(app.exec_())