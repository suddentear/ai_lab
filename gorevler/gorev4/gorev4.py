from PIL import Image, ImageDraw

# 1. Resmi yükle
image_path = "C:/Users/Asus/Desktop/KURSLARIM/BTK/Git_GitHub/ai_lab/gorevler/gorev3/ornek.jpg"
image = Image.open(image_path).convert("RGB")

# 2. Çizim nesnesi oluştur
draw = ImageDraw.Draw(image)

# 3. Dikdörtgenin koordinatları (left, top, right, bottom)
box = (100, 100, 300, 250)  # örnek koordinatlar

# 4. Sarı dikdörtgeni çiz (içi dolu)
draw.rectangle(box, fill="yellow", outline="yellow")

# 5. Sonucu kaydet
image.save("sari_dikdortgen_alan.jpg")
