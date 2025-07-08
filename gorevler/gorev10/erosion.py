import cv2
import matplotlib.pyplot as plt
import numpy as np

# Görseli yükle ve griye çevir
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev10/ornek.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kernel tanımla
kernel = np.ones((5, 5), np.uint8)

# Erosion uygula
eroded = cv2.erode(gray, kernel, iterations=1)

# Sonucu göster
plt.imshow(eroded, cmap='gray')
plt.title("Erosion")
plt.axis("off")
plt.show()
