import cv2
import numpy as np

# Görseli yükle
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev3/ornek.jpg")

# Orijinal görselin boyut ve kanal bilgisi
print("Orijinal görüntü şekli:", image.shape)  # (yükseklik, genişlik, 3)

# Griye çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gri görüntünün boyut bilgisi
print("Gri görüntü şekli:", gray.shape)  # (yükseklik, genişlik)

# Gri resmi 3 kanallı hale getir (karşılaştırmak için)
gray_3ch = cv2.merge([gray, gray, gray])

# Dizi farkını hesapla
difference = cv2.absdiff(image, gray_3ch)  # fark mutlak değerle alınır

# Farkın istatistiksel analizi
print("Fark - minimum değer:", np.min(difference))
print("Fark - maksimum değer:", np.max(difference))
print("Fark - ortalama değer:", np.mean(difference))

# Opsiyonel: fark görüntüsünü göster
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Orijinal Görsel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Gri Görsel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
plt.title("Fark (AbsDiff)")
plt.axis("off")

plt.tight_layout()
plt.show()
