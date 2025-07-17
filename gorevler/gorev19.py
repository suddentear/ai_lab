import cv2
import numpy as np
import time

# Dosya yolları
config_path = 'C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev19/yolov4.cfg'
weights_path = 'C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev19/yolov4.weights'
names_path = 'C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev19/coco.names'

# Sınıf isimlerini oku
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Yalnızca "person" sınıfına odaklan
target_class = "person"

# Ağ modeli yükle
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Kamera başlat
cap = cv2.VideoCapture(0)

# FPS hesaplama için zaman takibi
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Giriş blob'u oluştur
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Çıktı katman isimleri
    ln = net.getUnconnectedOutLayersNames()
    outputs = net.forward(ln)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == target_class and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = f"Person: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS hesapla
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    # FPS ekranın sağ üst köşesine yaz
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Görüntüyü göster
    cv2.imshow('YOLOv4 Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu ile çık
        break

cap.release()
cv2.destroyAllWindows()
