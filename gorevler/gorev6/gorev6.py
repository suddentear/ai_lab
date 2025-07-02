import cv2
image=cv2.imread("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev6/yaprak.jpg",0)
ret,thresh1=cv2.threshold(image,127,255, cv2.THRESH_BINARY)
cv2.imshow("original",image)
cv2.imshow("thresh1",thresh1)
cv2.imwrite("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev6/yaprak_gray.jpg", image)
cv2.imwrite("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev6/yaprak_thresh.jpg", thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
