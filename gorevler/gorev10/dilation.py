import cv2
import matplotlib.pyplot as plt
import numpy as np

# Görseli yükle ve griye çevir
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev10/ornek.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kernel tanımla
kernel = np.ones((5, 5), np.uint8)

# Dilation uygula
dilated = cv2.dilate(gray, kernel, iterations=1)

# Sonucu göster
plt.imshow(dilated, cmap='gray')
plt.title("Dilation")
plt.axis("off")
plt.show()
