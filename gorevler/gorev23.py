import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantik Segmentasyon Arayüzü")
        self.setGeometry(100, 100, 1000, 500)

        self.model_path = ""
        self.image_path = ""
        self.original_label = QLabel("Orijinal Görsel")
        self.segmented_label = QLabel("Segmentasyon Sonucu")

        self.original_label.setAlignment(Qt.AlignCenter)
        self.segmented_label.setAlignment(Qt.AlignCenter)

        # Layout’lar
        button_layout = QHBoxLayout()
        image_layout = QHBoxLayout()
        main_layout = QVBoxLayout()

        # Butonlar
        self.model_button = QPushButton("Model Seç")
        self.image_button = QPushButton("Görsel Seç")
        self.process_button = QPushButton("Segmentasyonu Başlat")

        self.model_button.clicked.connect(self.load_model)
        self.image_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.run_segmentation)

        # Butonları yerleştir
        button_layout.addWidget(self.model_button)
        button_layout.addWidget(self.image_button)
        button_layout.addWidget(self.process_button)

        # Görsel yerleşimi
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.segmented_label)

        # Ana yerleşim
        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

    def load_model(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, "Model Dosyasını Seç", "", "Model Files (*.pth *.onnx)")
        if self.model_path:
            self.model_button.setText(os.path.basename(self.model_path))

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Image Files (*.jpg *.png *.jpeg)")
        if self.image_path:
            self.display_image(self.image_path, self.original_label)

    def display_image(self, image_path, label):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def run_segmentation(self):
        if not self.image_path:
            return

        # Görseli oku
        image = cv2.imread(self.image_path)

        # --- Dummy model çıktısı (Gerçek modelle değiştirilebilir) ---
        segmented = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        segmented = cv2.applyColorMap(segmented, cv2.COLORMAP_JET)

        # -------------------------------------------------------------

        h, w, ch = segmented.shape
        qt_image = QImage(segmented.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.segmented_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())
