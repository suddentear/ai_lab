import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralıkları (HSV'de kırmızı iki ayrı aralıkta yer alır)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Maske oluştur
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Maskeyi orijinal görüntüye uygula
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Görüntüleri göster
    cv2.imshow("Orijinal", frame)
    cv2.imshow("Kırmızı Maske", mask)
    cv2.imshow("Kırmızı Renk Tespiti", result)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
