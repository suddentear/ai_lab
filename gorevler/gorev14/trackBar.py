import cv2
import numpy as np

def nothing(x):
    pass  # Trackbar callback'i için boş fonksiyon

# Pencere oluştur
cv2.namedWindow("Trackbars")

# Trackbar'ları oluştur
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HSV'ye dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar'lardan değerleri oku
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # HSV aralığına göre maske oluştur
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Maske ile görüntüyü filtrele
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Sonuçları göster
    cv2.imshow("Orijinal", frame)
    cv2.imshow("Maske", mask)
    cv2.imshow("Filtreli Goruntu", result)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
