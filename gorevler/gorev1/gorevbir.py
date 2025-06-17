import cv2
import os

# En güvenli yol: ileri eğik çizgi kullan
resim_yolu = "C://Users/Asus/Desktop/gorevler/gorev1/ornek.jpg"

# Resmi oku
resim = cv2.imread(resim_yolu)

if resim is None:
    print("Resim yüklenemedi. Dosya yolunu kontrol edin.")
else:
    cv2.imshow("Goruntu", resim)

    kayit_yolu = "C://Users/Asus/Desktop/gorevler/gorev1/"
    dosya_adi = "yeni_dosya.jpg"
    cv2.imwrite(os.path.join(kayit_yolu, dosya_adi), resim)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
