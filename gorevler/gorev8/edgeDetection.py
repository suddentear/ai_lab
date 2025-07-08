import cv2
import matplotlib.pyplot as plt

# Görseli yükle
image = cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev8/ornek.jpg")

# Gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny kenar algılama
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Sonucu görselleştir
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Canny)")
plt.axis("off")
plt.show()
