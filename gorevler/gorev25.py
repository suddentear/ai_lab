import sys
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# ----- Import your existing model wrapper -----
# Expecting a module `depth_model.py` with a class `DepthEstimatorV2` exposing:
#   - load_model()
#   - predict_depth(rgb: np.ndarray) -> np.ndarray (H,W) float32 depth map (any scale)
try:
    from depth_model import DepthEstimatorV2
except Exception as e:
    DepthEstimatorV2 = None
    _IMPORT_ERROR = e


# ---------------------------- 3D Utility Code -----------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Intrinsics:
    """Approximate pinhole intrinsics from horizontal FOV (in degrees)."""
    fov_rad = math.radians(max(1e-3, min(179.0, fov_deg)))
    fx = 0.5 * width / math.tan(fov_rad / 2.0)
    # keep square pixels by default
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return Intrinsics(fx, fy, cx, cy)


def depth_to_points(depth: np.ndarray,
                    intr: Intrinsics,
                    rgb: Optional[np.ndarray] = None,
                    stride: int = 1,
                    z_scale: float = 1.0,
                    valid_min: float = 1e-6,
                    valid_max: float = 1e9) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert depth map to point cloud (N,3) and optional colors (N,3 uint8).
    - depth: (H,W) float32, arbitrary scale. We'll multiply by z_scale.
    - rgb: (H,W,3) uint8 or None.
    - stride: subsampling step to reduce point count.
    """
    H, W = depth.shape
    # build grid
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)

    d = depth[::stride, ::stride].astype(np.float32) * float(z_scale)
    mask = np.isfinite(d) & (d > valid_min) & (d < valid_max)

    x = (uu - intr.cx) / intr.fx * d
    y = (vv - intr.cy) / intr.fy * d
    z = d

    pts = np.stack([x, y, z], axis=-1)[mask]

    cols = None
    if rgb is not None:
        if rgb.dtype != np.uint8:
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb8 = rgb
        cols = rgb8[::stride, ::stride][mask]
    return pts, cols


def depth_to_mesh_grid(depth: np.ndarray,
                        intr: Intrinsics,
                        rgb: Optional[np.ndarray] = None,
                        stride: int = 1,
                        z_scale: float = 1.0,
                        max_edge: float = 0.05,
                        valid_min: float = 1e-6,
                        valid_max: float = 1e9) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Triangulate a regular grid from depth.
    Returns (vertices (N,3), faces (M,3), colors (N,3) or None)
    - max_edge: drop triangles whose 3D edge length exceeds this (in depth units after scaling)
    """
    H, W = depth.shape
    intr = intr

    # Subsample grid indices
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    Wg, Hg = len(us), len(vs)

    # Map from (gi,gj) -> vertex index in flattened array
    # Prepare coordinates in vectorized manner
    uu, vv = np.meshgrid(us, vs)
    d = depth[::stride, ::stride].astype(np.float32) * float(z_scale)
    valid = np.isfinite(d) & (d > valid_min) & (d < valid_max)

    X = (uu - intr.cx) / intr.fx * d
    Y = (vv - intr.cy) / intr.fy * d
    Z = d

    V = np.stack([X, Y, Z], axis=-1)

    # Assign vertex indices for valid points; -1 for invalid
    vidx = -np.ones((Hg, Wg), dtype=np.int64)
    vidx[valid] = np.arange(valid.sum())

    verts = V[valid]

    cols = None
    if rgb is not None:
        rgb8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
        cols_full = rgb8[::stride, ::stride]
        cols = cols_full[valid]

    # Build faces by connecting grid quads into two triangles when all 3 vertices are valid and edges short enough
    faces = []
    def add_face(i1, i2, i3):
        faces.append([i1, i2, i3])

    # Precompute neighbor vectors for edge checks
    for j in range(Hg - 1):
        for i in range(Wg - 1):
            a = vidx[j, i]
            b = vidx[j, i + 1]
            c = vidx[j + 1, i]
            d2 = vidx[j + 1, i + 1]
            if a < 0 or b < 0 or c < 0 or d2 < 0:
                continue
            va, vb, vc, vd = verts[a], verts[b], verts[c], verts[d2]

            # edge length filter
            def ok(tri):
                p, q, r = tri
                return (np.linalg.norm(p - q) <= max_edge and
                        np.linalg.norm(q - r) <= max_edge and
                        np.linalg.norm(r - p) <= max_edge)

            # two triangles: (a, b, c) and (b, d, c)
            if ok((va, vb, vc)):
                add_face(a, b, c)
            if ok((vb, vd, vc)):
                add_face(b, d2, c)

    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces, cols


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray, cols: Optional[np.ndarray] = None):
    """Save OBJ (optionally with per-vertex colors as a common non-standard extension)."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# OBJ file generated from depth map\n')
        if cols is not None:
            # write vertex with rgb as floats 0..1 appended; many viewers ignore but Meshlab/CloudCompare handle it
            for (x, y, z), (r, g, b) in zip(verts, cols):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r/255.0:.6f} {g/255.0:.6f} {b/255.0:.6f}\n")
        else:
            for x, y, z in verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            # OBJ is 1-indexed
            f.write(f"f {a+1} {b+1} {c+1}\n")


def save_ply(path: str, verts: np.ndarray, faces: Optional[np.ndarray] = None, cols: Optional[np.ndarray] = None):
    """Save ASCII PLY (supports vertex colors)."""
    verts = np.asarray(verts)
    n_vert = verts.shape[0]
    faces = np.asarray(faces) if faces is not None and len(faces) > 0 else None
    n_face = 0 if faces is None else faces.shape[0]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {n_vert}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        if cols is not None:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        if faces is not None:
            f.write(f'element face {n_face}\n')
            f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        if cols is not None:
            for (x, y, z), (r, g, b) in zip(verts, cols):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in verts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        if faces is not None:
            for a, b, c in faces:
                f.write(f"3 {a} {b} {c}\n")


# ---------------------------- UI Application ------------------------------
class DepthApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DepthEstimationV2 - 3D Export UI')
        self.resize(1200, 800)

        self.model = None
        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Left: controls
        controls = QVBoxLayout()

        self.btn_load = QPushButton('Resim Aç')
        self.btn_infer = QPushButton('Derinlik Çıkar')
        self.btn_export = QPushButton('3D Model Dışa Aktar (.obj/.ply)')

        self.btn_infer.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_infer.clicked.connect(self.on_infer)
        self.btn_export.clicked.connect(self.on_export)

        # Params group
        params = QGroupBox('3D Parametreleri')
        gp = QVBoxLayout(params)

        row1 = QHBoxLayout()
        lbl_fov = QLabel('Yatay FOV (°):')
        self.spn_fov = QDoubleSpinBox()
        self.spn_fov.setRange(1.0, 179.0)
        self.spn_fov.setValue(60.0)
        self.spn_fov.setDecimals(1)
        row1.addWidget(lbl_fov)
        row1.addWidget(self.spn_fov)

        row2 = QHBoxLayout()
        lbl_z = QLabel('Z Ölçeği:')
        self.spn_z = QDoubleSpinBox()
        self.spn_z.setRange(0.01, 100.0)
        self.spn_z.setValue(1.0)
        self.spn_z.setDecimals(2)
        row2.addWidget(lbl_z)
        row2.addWidget(self.spn_z)

        row3 = QHBoxLayout()
        lbl_stride = QLabel('Örnekleme (stride):')
        self.spn_stride = QSpinBox()
        self.spn_stride.setRange(1, 16)
        self.spn_stride.setValue(1)
        row3.addWidget(lbl_stride)
        row3.addWidget(self.spn_stride)

        row4 = QHBoxLayout()
        lbl_edge = QLabel('Maks. Kenar Uzunluğu:')
        self.spn_edge = QDoubleSpinBox()
        self.spn_edge.setRange(0.001, 10.0)
        self.spn_edge.setValue(0.05)
        self.spn_edge.setDecimals(3)
        row4.addWidget(lbl_edge)
        row4.addWidget(self.spn_edge)

        gp.addLayout(row1)
        gp.addLayout(row2)
        gp.addLayout(row3)
        gp.addLayout(row4)

        controls.addWidget(self.btn_load)
        controls.addWidget(self.btn_infer)
        controls.addWidget(self.btn_export)
        controls.addWidget(params)
        controls.addStretch(1)

        # Right: image/depth previews
        previews = QVBoxLayout()
        self.lbl_img = QLabel('Görüntü')
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setMinimumHeight(300)
        self.lbl_img.setStyleSheet('border:1px solid #aaa;')

        self.lbl_depth = QLabel('Derinlik')
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setMinimumHeight(300)
        self.lbl_depth.setStyleSheet('border:1px solid #aaa;')

        previews.addWidget(self.lbl_img)
        previews.addWidget(self.lbl_depth)

        main_layout.addLayout(controls, 0)
        main_layout.addLayout(previews, 1)

        # Lazy-load model
        self.load_model_if_needed()

    # ----------------------- Utility methods -----------------------
    def load_model_if_needed(self):
        if self.model is not None:
            return
        if DepthEstimatorV2 is None:
            QMessageBox.critical(self, 'Model Yüklenemedi',
                                 f'depth_model.py içe aktarılırken hata oluştu:\n{_IMPORT_ERROR}')
            return
        try:
            self.model = DepthEstimatorV2()
            self.model.load_model()
        except Exception as e:
            QMessageBox.critical(self, 'Model Yüklenemedi', f'Model yükleme hatası: {e}')
            self.model = None

    def np_to_qpix(self, img_bgr_or_rgb: np.ndarray) -> QPixmap:
        if img_bgr_or_rgb.ndim == 2:
            # grayscale
            h, w = img_bgr_or_rgb.shape
            qimg = QImage(img_bgr_or_rgb.data, w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimg)
        else:
            # assume BGR or RGB uint8
            if img_bgr_or_rgb.shape[2] == 3:
                # OpenCV default is BGR
                rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                return QPixmap.fromImage(qimg)
            elif img_bgr_or_rgb.shape[2] == 4:
                h, w, ch = img_bgr_or_rgb.shape
                qimg = QImage(img_bgr_or_rgb.data, w, h, ch * w, QImage.Format_RGBA8888)
                return QPixmap.fromImage(qimg)
            else:
                raise ValueError('Desteklenmeyen görüntü formatı')

    def show_depth_colormap(self, depth: np.ndarray) -> np.ndarray:
        # Normalize for visualization only
        d = depth.copy().astype(np.float32)
        d[np.isnan(d)] = 0
        if np.all(d == 0):
            d_vis = np.zeros_like(d, dtype=np.uint8)
        else:
            d -= d.min()
            if d.max() > 0:
                d /= d.max()
            d_vis = (d * 255.0).astype(np.uint8)
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
        return d_vis

    # ----------------------- Slots -----------------------
    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Resim Aç', '', 'Görüntüler (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.warning(self, 'Hata', 'Görüntü okunamadı.')
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.depth = None
        self.lbl_img.setPixmap(self.np_to_qpix(bgr).scaled(self.lbl_img.width(), self.lbl_img.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_depth.setText('Derinlik')
        self.btn_infer.setEnabled(True)
        self.btn_export.setEnabled(False)

    def on_infer(self):
        if self.model is None:
            self.load_model_if_needed()
            if self.model is None:
                return
        if self.rgb is None:
            QMessageBox.information(self, 'Bilgi', 'Önce bir görüntü açın.')
            return
        try:
            depth = self.model.predict_depth(self.rgb)  # (H,W) float32
            if depth is None or depth.ndim != 2:
                raise ValueError('predict_depth geçersiz çıktı üretti')
            self.depth = depth.astype(np.float32)
            depth_vis = self.show_depth_colormap(self.depth)
            self.lbl_depth.setPixmap(self.np_to_qpix(depth_vis).scaled(self.lbl_depth.width(), self.lbl_depth.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_export.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, 'Hata', f'Derinlik çıkarma başarısız: {e}')

    def on_export(self):
        if self.depth is None or self.rgb is None:
            QMessageBox.information(self, 'Bilgi', 'Önce derinliği çıkarın.')
            return

        # Ask path & format
        save_path, sel_filter = QFileDialog.getSaveFileName(
            self,
            '3D Model Dışa Aktar',
            '',
            'OBJ (*.obj);;PLY (mesh) (*.ply);;PLY (point cloud) (*.ply)'
        )
        if not save_path:
            return

        # Intrinsics from FOV
        H, W = self.depth.shape
        intr = intrinsics_from_fov(W, H, float(self.spn_fov.value()))

        stride = int(self.spn_stride.value())
        z_scale = float(self.spn_z.value())
        max_edge = float(self.spn_edge.value())

        try:
            if 'point cloud' in sel_filter:
                pts, cols = depth_to_points(self.depth, intr, self.rgb, stride=stride, z_scale=z_scale)
                # save as PLY without faces
                if not save_path.lower().endswith('.ply'):
                    save_path += '.ply'
                save_ply(save_path, pts, faces=None, cols=cols)
            else:
                # mesh (OBJ or PLY mesh)
                verts, faces, cols = depth_to_mesh_grid(self.depth, intr, self.rgb, stride=stride, z_scale=z_scale, max_edge=max_edge)
                if verts.shape[0] == 0 or faces.shape[0] == 0:
                    raise ValueError('Geçerli üçgen üretilemedi; parametreleri değiştirin (stride/edge/z_scale).')
                if sel_filter.startswith('OBJ'):
                    if not save_path.lower().endswith('.obj'):
                        save_path += '.obj'
                    save_obj(save_path, verts, faces, cols)
                else:
                    if not save_path.lower().endswith('.ply'):
                        save_path += '.ply'
                    save_ply(save_path, verts, faces, cols)
            QMessageBox.information(self, 'Tamamlandı', f'3D çıktı kaydedildi:\n{save_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Dışa Aktarma Hatası', str(e))


# ------------------------------ main --------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = DepthApp()
    w.show()
    sys.exit(app.exec_())
