import cv2
import matplotlib.pyplot as plt

# Görseli yükle
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev11/ornek.jpg")

# 1. Dikey yansıtma (flipCode = 0)
flip_vertical = cv2.flip(image, 0)

# 2. Yatay yansıtma (flipCode = 1)
flip_horizontal = cv2.flip(image, 1)

# 3. Hem yatay hem dikey yansıtma (flipCode = -1)
flip_both = cv2.flip(image, -1)

# Sonuçları göster
cv2.imshow("Original", image)
cv2.imshow("Vertical Flip", flip_vertical)
cv2.imshow("Horizontal Flip", flip_horizontal)
cv2.imshow("Both Flip", flip_both)

cv2.waitKey(0)
cv2.destroyAllWindows()
