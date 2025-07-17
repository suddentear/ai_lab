import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resim Gösterici")
        self.setGeometry(100, 100, 400, 400)

        # Ana düzen
        layout = QVBoxLayout()

        # QLabel oluştur ve resmi yükle
        self.label = QLabel(self)
        pixmap = QPixmap("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev22/ornek.jpg")  # Buraya kendi görselinizin yolunu yazın
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Resmi pencereye sığdır

        # Label'ı layout'a ekle
        layout.addWidget(self.label)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec_())
