from PIL import Image, ImageDraw, ImageFilter

# 1. Resmi yükle
image_path = "C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev4/sari_dikdortgen_alan.jpg"
image = Image.open(image_path).convert("RGB")

# 2. Dikdörtgen alan tanımı (sol, üst, sağ, alt)
box = (100, 100, 300, 300)  # örnek: kare alan

# 3. Belirli alanı kırp
region = image.crop(box)

# 4. Grayscale (gri) yap
gray_region = region.convert("L").convert("RGB")

# 5. Bulanıklaştır
blurred_region = gray_region.filter(ImageFilter.GaussianBlur(radius=4))

# 6. Orijinal görüntüye bu bölgeyi yapıştır
image.paste(blurred_region, box)

# 7. Sarı çerçeve çiz
draw = ImageDraw.Draw(image)
draw.rectangle(box, outline="yellow", width=4)

# 8. Sonuç görselini kaydet
image.save("islenmis_goruntu.jpg")
