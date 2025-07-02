import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Görseli oku
img = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev7/yaprakyesil.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Threshold ile ikili görüntü oluştur
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Gürültüyü temizlemek için morfolojik açma (erosion + dilation)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Arka planı belirle (dilate edilmiş alan)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 5. Ön planı (nesneleri) bulmak için distance transform kullan
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 6. Bilinmeyen alanları bul
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 7. Marker'ları (etiketli bölgeler) oluştur
ret, markers = cv2.connectedComponents(sure_fg)

# 8. Bilinmeyen alanları 0 yap
markers = markers + 1
markers[unknown == 255] = 0

# 9. Watershed uygula
markers = cv2.watershed(img, markers)

# 10. Sınır çizgilerini kırmızı yap (piksel değeri -1 olanlar)
img[markers == -1] = [0, 0, 255]

# 11. Görseli göster
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite("watershed_sonucu.jpg", img)
plt.title('Watershed Sonucu')
plt.axis('off')
plt.show()
