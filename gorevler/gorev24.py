# gorev24.py
# -*- coding: utf-8 -*-
import sys
import argparse
import time
from typing import Union

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QStatusBar,
    QShortcut,
    QMessageBox,
)

from depth_model import DepthEstimationV2


def np_to_qimage(img: np.ndarray) -> QImage:
    """NumPy görüntüsünü (RGB ya da tek kanallı) QImage'e çevirir."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    elif img.ndim == 3:
        h, w, c = img.shape
        assert c == 3
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()
    else:
        raise ValueError("Desteklenmeyen görüntü boyutu")


class VideoThread(QThread):
    frame_rgb = pyqtSignal(QImage)
    depth_rgb = pyqtSignal(QImage)
    error = pyqtSignal(str)
    fps = pyqtSignal(float)

    def __init__(self, source: Union[int, str], model: DepthEstimationV2, parent=None):
        super().__init__(parent)
        self.source = source
        self.model = model
        self._running = False
        self._cap = None

    def run(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self.error.emit("Kaynak açılamadı: {}".format(self.source))
            return

        self._running = True
        last_t = time.time()
        frames = 0

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.error.emit("Kare okunamadı. Kaynak bitti veya kamera erişilemiyor.")
                break

            try:
                depth8, depth_col = self.model.predict(frame)
            except Exception as e:
                self.error.emit(f"Model hatası: {e}")
                break

            # QImage (RGB) yayınla
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_rgb = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)
            self.frame_rgb.emit(np_to_qimage(rgb))
            self.depth_rgb.emit(np_to_qimage(depth_rgb))

            frames += 1
            now = time.time()
            if now - last_t >= 1.0:
                self.fps.emit(frames / (now - last_t))
                frames = 0
                last_t = now

        if self._cap is not None:
            self._cap.release()

    def stop(self):
        self._running = False
        self.wait(1000)


class MainWindow(QMainWindow):
    def __init__(self, source: Union[int, str]):
        super().__init__()
        self.setWindowTitle("DepthEstimationV2 Görüntüleyici (q ile çık)")
        self.resize(1200, 600)

        # Model
        self.model = DepthEstimationV2()

        # Görüntü label'ları
        self.orig_label = QLabel("Orijinal")
        self.depth_label = QLabel("Derinlik")
        for lab in (self.orig_label, self.depth_label):
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")

        # Yerleşim
        content = QWidget()
        hl = QHBoxLayout(content)
        hl.addWidget(self.orig_label, 1)
        hl.addWidget(self.depth_label, 1)

        # Kontroller
        self.btn_start = QPushButton("Başlat")
        self.btn_stop = QPushButton("Durdur")
        self.btn_open = QPushButton("Video Aç…")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open.clicked.connect(self.open_file)

        controls = QWidget()
        cl = QHBoxLayout(controls)
        cl.addWidget(self.btn_start)
        cl.addWidget(self.btn_stop)
        cl.addWidget(self.btn_open)
        cl.addStretch(1)

        wrapper = QWidget()
        vl = QVBoxLayout(wrapper)
        vl.addWidget(content, 1)
        vl.addWidget(controls, 0)
        self.setCentralWidget(wrapper)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Kısayollar: q ve Esc -> çıkış
        self._sc_q = QShortcut(QKeySequence("q"), self)
        self._sc_q.activated.connect(self.close)
        self._sc_Q = QShortcut(QKeySequence("Q"), self)
        self._sc_Q.activated.connect(self.close)
        self._sc_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._sc_esc.activated.connect(self.close)

        # Video iş parçacığı yönetimi
        self.source = source
        self.vthread = None

    def start(self):
        if self.vthread and self.vthread.isRunning():
            return
        self.vthread = VideoThread(self.source, self.model)
        self.vthread.frame_rgb.connect(self.update_orig)
        self.vthread.depth_rgb.connect(self.update_depth)
        self.vthread.error.connect(self.on_error)
        self.vthread.fps.connect(self.on_fps)
        self.vthread.start()
        self.status.showMessage("Akış başladı…")

    def stop(self):
        if self.vthread:
            self.vthread.stop()
            self.vthread = None
            self.status.showMessage("Akış durdu.")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video dosyası seç", "", "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.source = path
            self.stop()
            self.start()

    def update_orig(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.orig_label.setPixmap(pix.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_depth(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.depth_label.setPixmap(pix.scaled(self.depth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_error(self, msg: str):
        self.status.showMessage(msg)
        QMessageBox.critical(self, "Hata", msg)
        self.stop()

    def on_fps(self, fps: float):
        self.status.showMessage(f"FPS: {fps:.1f}")

    def keyPressEvent(self, event):
        # Yedek: q / Esc ile kapanış
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stop()
        return super().closeEvent(event)


def parse_args():
    p = argparse.ArgumentParser(description="DepthEstimationV2 PyQt5 görüntüleyici")
    p.add_argument("--source", default=0, help="Kamera indeksi (örn. 0) veya video yolu")
    args = p.parse_args()
    # Kamera indeksi sayıya çevrilebiliyorsa int yap
    src = args.source
    try:
        src = int(src)
    except Exception:
        pass
    return src


def main():
    source = parse_args()
    app = QApplication(sys.argv)
    win = MainWindow(source)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
