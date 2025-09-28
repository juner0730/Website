#predict.py
import os, cv2, numpy as np, torch
from ultralytics import YOLO
from typing import Optional, Tuple, List
from config import MODEL_PATH

for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "1")

USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else "cpu"

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.cuda.set_device(DEVICE if isinstance(DEVICE, int) else 0)
    except Exception:
        pass

print(f"torch.cuda.is_available() = {USE_CUDA}")
if USE_CUDA:
    print(f"CUDA device count = {torch.cuda.device_count()}")
    print(f"Using device index = {DEVICE}")
    print(f"GPU name = {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

model = YOLO(MODEL_PATH)

PRED_KW = dict(device=DEVICE, imgsz=224, verbose=False)
if USE_CUDA:
    PRED_KW["half"] = True  # 僅 CUDA

_dummy = np.zeros((224, 224, 3), dtype=np.uint8)
warm = model(_dummy, **PRED_KW)
try:
    core = getattr(model, "model", None) or model
    print("Model device =", next(core.parameters()).device)
except Exception as e:
    print("Cannot read model device:", e)

def _safe_crop_bgr(frame: np.ndarray, bbox: Tuple[int, int, int, int], max_side=512):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
    return np.ascontiguousarray(crop)

@torch.inference_mode()
def detect_jersey_number(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], List[float]]:
    crop = _safe_crop_bgr(frame, bbox, max_side=512)
    if crop is None:
        return None, []
    h_person = crop.shape[0]
    cut_h = int(h_person * 0.3)
    crop_no_bottom = crop[: max(1, h_person - cut_h), :]
    gray = cv2.cvtColor(crop_no_bottom, cv2.COLOR_BGR2GRAY)
    crop_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    crop_gray_bgr = np.ascontiguousarray(crop_gray_bgr)

    results = model(crop_gray_bgr, **PRED_KW)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, []

    items: List[Tuple[float, str]] = []
    confidences: List[float] = []
    for box in boxes:
        conf = float(box.conf[0].item())
        if conf < 0.25:
            continue
        xy = box.xyxy[0].detach().cpu().numpy()
        x_center = float((xy[0] + xy[2]) / 2)
        digit = str(int(box.cls[0].item()))
        
        items.append((x_center, digit))
        confidences.append(conf)

    items.sort(key=lambda x: x[0])
    number = ''.join([d for _, d in items])
    confidences.sort(reverse=True)
    return (number if number else None), confidences
