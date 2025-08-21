# gorev20.py - Basit, parametresiz YOLOv5 realtime (Windows/PowerShell uyumlu)
import os, sys, cv2, time
from pathlib import Path

# --- YOLOv5 klasörünü bul ve sys.path'e ekle ---
FILE = Path(__file__).resolve()
CWD = FILE.parent
candidates = [
    CWD,                              # script ile aynı klasör
    CWD / "yolov5",                   # script altı
    Path.home() / "Desktop" / "yolov5",  # tipik Windows masaüstü
]
yolov5_root = None
for p in candidates:
    if (p / "models").exists() and (p / "utils").exists():
        yolov5_root = p
        break
if not yolov5_root:
    # Son çare: kullanıcıdan tipik konumu deneyelim
    guess = Path(r"C:\Users\Asus\Desktop\yolov5")
    if (guess / "models").exists():
        yolov5_root = guess
if not yolov5_root:
    raise SystemExit(
        "YOLOv5 klasörü bulunamadı.\n"
        "Çözüm: 'git clone https://github.com/ultralytics/yolov5.git' ile indirin ve\n"
        "yolov5 klasörünü masaüstüne (C:\\Users\\Asus\\Desktop\\yolov5) koyun."
    )

if str(yolov5_root) not in sys.path:
    sys.path.insert(0, str(yolov5_root))

# --- YOLOv5 importları ---
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

import torch

# --- Ağırlık yolunu otomatik seç ---
def pick_weights(root: Path) -> Path:
    # 1) En son eğitim çıktısı
    best1 = root / "runs" / "train" / "exp" / "weights" / "best.pt"
    # 2) Diğer exp klasörlerinde en yeni best.pt'yi ara
    runs_train = root / "runs" / "train"
    best2 = None
    if runs_train.exists():
        exps = sorted([d for d in runs_train.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        for e in exps:
            cand = e / "weights" / "best.pt"
            if cand.exists():
                best2 = cand
                break
    # 3) Hazır küçük model
    ready = root / "yolov5s.pt"

    for w in [best1, best2, ready]:
        if w and Path(w).exists():
            return Path(w)
    raise SystemExit(
        "Ağırlık dosyası bulunamadı.\n"
        f"Şunlardan biri olmalı:\n- {best1}\n- (runs/train/*/weights/best.pt)\n- {ready}\n"
        "Çözüm: Kendi modelinizi eğitin veya yolov5s.pt dosyasını YOLOv5 klasörüne indirin."
    )

def open_capture(src="0"):
    # Windows'ta gecikmeyi azaltmak için CAP_DSHOW
    if str(src).isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Video kaynağı açılamadı: {src}")
    return cap

def main():
    weights = pick_weights(yolov5_root)
    device = select_device("")  # otomatik: varsa GPU, yoksa CPU
    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    show_fps = True
    source = "0"  # webcam

    print(f"[INFO] YOLOv5 kök: {yolov5_root}")
    print(f"[INFO] Ağırlık: {weights}")
    print(f"[INFO] Kaynak: {source} (webcam)")

    model = DetectMultiBackend(str(weights), device=device, dnn=False, fp16=(device.type != "cpu"))
    stride, names = model.stride, model.names
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    cap = open_capture(source)
    prev_t = 0.0

    try:
        while True:
            ok, im0 = cap.read()
            if not ok:
                print("[WARN] Kare alınamadı, akış bitti.")
                break

            # Ön işleme
            im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            im = im.copy()
            im_tensor = torch.from_numpy(im).to(device)
            im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
            im_tensor /= 255.0
            if im_tensor.ndimension() == 3:
                im_tensor = im_tensor.unsqueeze(0)

            # Çıkarım + NMS
            with torch.no_grad():
                pred = model(im_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

            # Çizim
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        c = int(cls.item())
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f"{names[c] if isinstance(names, list) else c} {float(conf):.2f}"
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS
            if show_fps:
                now = time.time()
                fps = 1.0 / (now - prev_t) if prev_t else 0.0
                prev_t = now
                cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Göster
            cv2.imshow("YOLOv5 Realtime (Parametresiz)", im0)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # ESC veya q
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
