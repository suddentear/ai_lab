import cv2

# Kameraya eriş (0: varsayılan kamera)
cap = cv2.VideoCapture(0)

# Kamera açılmış mı kontrol et
if not cap.isOpened():
    print("Kameraya erişilemedi.")
    exit()

# Sonsuz döngü ile sürekli görüntü al
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Görüntü alınamadı.")
        break

    # Görüntüyü pencere içinde göster
    cv2.imshow("Kamera Goruntusu", frame)

    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
