from PIL import Image

# 1. Resmi yükle
image_path = "ornek.jpg"
image = Image.open("C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev2/ornek.jpg")

# 2. Yeniden boyutlandır (örneğin 800x600 piksel)
new_size = (800, 600)
resized_image = image.resize(new_size)

# 3. Belirli bir alanı seç (sol, üst, sağ, alt)
crop_box = (200, 150, 600, 450)  # (left, top, right, bottom)
cropped_image = resized_image.crop(crop_box)

# 4. Kayıt et
resized_image.save("yeniden_boyutlandirilmis.jpg")
cropped_image.save("secili_alan.jpg")
