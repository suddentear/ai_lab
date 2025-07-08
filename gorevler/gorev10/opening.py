import cv2
import matplotlib.pyplot as plt
import numpy as np

# Görseli yükle ve griye çevir
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev10/ornek.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kernel tanımla
kernel = np.ones((5, 5), np.uint8)

# Opening uygula
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# Sonucu göster
plt.imshow(opened, cmap='gray')
plt.title("Opening")
plt.axis("off")
plt.show()
