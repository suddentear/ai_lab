import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kameraya erişilemedi.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Siyah renk aralığı (HSV formatında)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Parlaklığı düşük olan renkler

    # Maske oluştur (siyah alanlar 1, diğerleri 0)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturlar üzerinden geç ve dikdörtgen çiz
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Gürültüleri engellemek için minimum alan
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Siyah Nesne", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow("Kamera", frame)
    cv2.imshow("Maske", mask)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
