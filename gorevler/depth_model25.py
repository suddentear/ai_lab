# depth_model.py (UI uyumlu, cache kurtarmalı)
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, shutil
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _clean_midas_cache():
    hub_dir = torch.hub.get_dir()  # örn: C:\Users\<user>\.cache\torch\hub
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    midas_repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
    for p in (ckpt_dir, midas_repo):
        if os.path.isdir(p):
            try:
                shutil.rmtree(p)
            except Exception:
                pass


class DepthEstimatorV2:
    """
    UI'nin beklediği arayüz:
      - load_model()
      - predict_depth(rgb: np.ndarray[H,W,3] RGB uint8) -> np.ndarray[H,W] float32
    """
    def __init__(self, model_type: str = "DPT_Hybrid", device: Optional[str] = None, optimize: bool = True):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_flag = optimize and (self.device == "cuda")
        self.model = None
        self.transform = None

    def _safe_load_model(self):
        try:
            model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        except Exception as e:
            # bozuk/yarım inmiş cache durumunu toparla ve tekrar dene
            if "PytorchStreamReader failed" in str(e) or "failed finding central directory" in str(e):
                _clean_midas_cache()
                model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            else:
                raise RuntimeError(
                    f"MiDaS modeli yüklenemedi: {e}\n"
                    "İnterneti ve 'timm' kurulumunu kontrol edin: pip install timm"
                )
        return model

    def _safe_load_transforms(self):
        try:
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        except Exception as e:
            if "PytorchStreamReader failed" in str(e) or "failed finding central directory" in str(e):
                _clean_midas_cache()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            else:
                raise RuntimeError(f"MiDaS transforms yüklenemedi: {e}")
        return transforms

    def load_model(self):
        """UI bu fonksiyonu çağırır."""
        self.model = self._safe_load_model().to(self.device).eval()
        transforms = self._safe_load_transforms()
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        if self.optimize_flag:
            # half precision + autocast hız için
            self.model = self.model.half()

    @torch.inference_mode()
    def predict_depth(self, rgb: np.ndarray) -> np.ndarray:
        """
        Girdi: RGB uint8 (H,W,3)
        Çıktı: float32 (H,W) ham derinlik (ölçeklenmemiş, göreli)
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model yüklü değil. Önce load_model() çağırın.")

        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("predict_depth RGB (H,W,3) bekler.")

        # MiDaS transformu PyTorch tensörüne çevirir (NCHW, 1 batch)
        inp = self.transform(rgb).to(self.device)
        inp = inp.half() if self.optimize_flag else inp.float()

        pred = self.model(inp)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)  # (N,1,H,W)

        # Orijinal çözünürlüğe yeniden ölçekle
        H, W = rgb.shape[:2]
        pred = F.interpolate(pred, size=(H, W), mode="bicubic", align_corners=False)
        depth = pred.squeeze().detach().float().cpu().numpy()  # (H,W) float32

        return depth


# Hızlı test (isteğe bağlı): bir görüntü yolu verin
if __name__ == "__main__":
    import sys

    model = DepthEstimatorV2()
    model.load_model()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        bgr = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = model.predict_depth(rgb)
        # sadece görselleştirme
        d = depth.copy()
        d -= d.min()
        d = d / d.max() if d.max() > 1e-8 else d
        d8 = (255 * (1.0 - d)).astype(np.uint8)
        dcm = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
        show = np.hstack([bgr, dcm])
        cv2.imshow("MiDaS Depth (q/ESC)", show)
        cv2.waitKey(0)
    else:
        # kamera ile deneme
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Kamera açılamadı (source=0).")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = model.predict_depth(rgb)
            d = depth.copy()
            d -= d.min()
            d = d / d.max() if d.max() > 1e-8 else d
            d8 = (255 * (1.0 - d)).astype(np.uint8)
            dcm = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
            show = np.hstack([frame, dcm])
            cv2.imshow("MiDaS Depth (q/ESC)", show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
