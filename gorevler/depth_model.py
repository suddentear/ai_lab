# depth_model.py (cache kurtarma özellikli)
# -*- coding: utf-8 -*-
from __future__ import annotations
import cv2, os, shutil
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def _clean_midas_cache():
    hub_dir = torch.hub.get_dir()  # genelde C:\Users\<user>\.cache\torch\hub
    # checkpoints ve MiDaS repo klasörünü temizle
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    midas_repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
    for p in (ckpt_dir, midas_repo):
        if os.path.isdir(p):
            try:
                shutil.rmtree(p)
            except Exception:
                pass

class DepthEstimationV2:
    def __init__(self, model_type: str = "DPT_Hybrid", device: str | None = None, optimize: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.optimize = optimize and (self.device == "cuda")

        self.model = self._safe_load_model()
        self.model.eval().to(self.device)

        transforms = self._safe_load_transforms()
        if self.model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        if self.optimize:
            self.model = self.model.half()

    def _safe_load_model(self):
        try:
            return torch.hub.load("intel-isl/MiDaS", self.model_type)
        except Exception as e:
            # Bozuk cache vakasını kurtarmayı dene
            if "PytorchStreamReader failed" in str(e) or "failed finding central directory" in str(e):
                _clean_midas_cache()
                return torch.hub.load("intel-isl/MiDaS", self.model_type)
            raise RuntimeError(
                f"MiDaS model yüklenemedi: {e}\n"
                "İnterneti ve 'timm' kurulumunu kontrol edin: pip install timm"
            )

    def _safe_load_transforms(self):
        try:
            return torch.hub.load("intel-isl/MiDaS", "transforms")
        except Exception as e:
            if "PytorchStreamReader failed" in str(e) or "failed finding central directory" in str(e):
                _clean_midas_cache()
                return torch.hub.load("intel-isl/MiDaS", "transforms")
            raise RuntimeError(f"MiDaS transforms yüklenemedi: {e}")

    @torch.inference_mode()
    def predict(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Boş kare alındı (frame).")

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        if self.optimize:
            inp = inp.half()
        else:
            inp = inp.float()

        pred = self.model(inp)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=rgb.shape[:2], mode="bicubic", align_corners=False)
        pred = pred.squeeze().detach().float().cpu().numpy()

        mn, mx = float(pred.min()), float(pred.max())
        denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
        depth_norm = (pred - mn) / denom

        depth_uint8 = np.clip(255.0 * (1.0 - depth_norm), 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return depth_uint8, depth_color

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Kamera açılamadı (source=0).")
    model = DepthEstimationV2()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        depth8, depthc = model.predict(frame)
        show = np.hstack([frame, depthc])
        cv2.imshow("DepthEstimationV2 quick test (q/ESC to quit)", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
