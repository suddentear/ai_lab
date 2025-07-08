import cv2
import matplotlib.pyplot as plt

# 1. Görseli yükle
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev3/ornek.jpg")

# 2. Görseli yeniden boyutlandır (örneğin 300x300 piksel)
resized = cv2.resize(image, (300, 300))

# 3. Ortadaki 100x100 piksellik alanı seç (ROI - Region of Interest)
x, y, w, h = 100, 100, 100, 100  # sol üst köşe (x=100, y=100) ve genişlik-yükseklik
roi = resized[y:y+h, x:x+w]

# 4. Renkleri BGR'den RGB'ye çevir (Matplotlib doğru renk göstersin diye)
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

# 5. Görselleri göster
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(resized_rgb)
plt.title("Yeniden Boyutlandirilmis Gorsel (300x300)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(roi_rgb)
plt.title("Secilen Alan (100x100)")
plt.axis("off")

plt.tight_layout()
plt.show()
