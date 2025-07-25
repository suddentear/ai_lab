import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kamera ve Nesne ID Atama")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Kamerayı Başlat")
        self.stop_button = QPushButton("Kamerayı Durdur")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.object_id_counter = 0

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image_label.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Görüntüyü işle ve nesne ID'lerini ata
        processed_frame = self.detect_objects_and_assign_ids(frame)

        # OpenCV görüntüsünü QImage'e dönüştür
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, channel * width, QImage.Format_RGB888)

        # QLabel'e görüntüyü yerleştir
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def detect_objects_and_assign_ids(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                object_id = i + 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {object_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
