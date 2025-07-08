import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görseli yükle
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev8/ornek.jpg")

# Griye çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi köşe tespiti
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.intp(corners)  # Köşe koordinatlarını tam sayıya çevir

# Köşeleri çiz
for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

# Görseli BGR'den RGB'ye çevir ve göster
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Corner Detection (Shi-Tomasi)")
plt.axis("off")
plt.show()
