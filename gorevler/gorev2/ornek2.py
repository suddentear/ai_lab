import cv2

# Resmi yükle
image = cv2.imread('C:/Users/Asus/Desktop/ai_lab/gorevler/gorev2/ornek.jpg')  # Dosya adını kendi dosyanızla değiştirin

# Gri tonlamaya dönüştür
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# İsteğe bağlı: Gaussian Blur filtresi uygula
filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Görüntüyü göster
cv2.imshow('Gri Filtreli Goruntu', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kaydetmek isterseniz:
cv2.imwrite('gri_filtreli_resim.jpg', filtered_image)