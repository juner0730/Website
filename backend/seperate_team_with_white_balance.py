import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tqdm import tqdm
import json

# =============================================================================
# ===== 新增：白平衡函式 (Gray World Algorithm) =====
# =============================================================================
def apply_white_balance(image):
    """
    使用灰度世界演算法對圖像進行白平衡處理
    :param image: 輸入的 BGR 圖像
    :return: 白平衡後的 BGR 圖像
    """
    # 將圖像轉換為 float32 以進行精確計算
    img_float = image.astype(np.float32)
    
    # 分離 B, G, R 通道
    b, g, r = cv2.split(img_float)
    
    # 計算每個通道的平均值
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # 計算所有通道的總平均值
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # 計算每個通道的縮放因子
    # 避免除以零的錯誤
    scale_b = avg_gray / (avg_b + 1e-6)
    scale_g = avg_gray / (avg_g + 1e-6)
    scale_r = avg_gray / (avg_r + 1e-6)
    
    # 應用縮放因子
    b_balanced = np.clip(b * scale_b, 0, 255)
    g_balanced = np.clip(g * scale_g, 0, 255)
    r_balanced = np.clip(r * scale_r, 0, 255)
    
    # 合併通道並轉換回 uint8
    balanced_img = cv2.merge([b_balanced, g_balanced, r_balanced]).astype(np.uint8)
    
    return balanced_img

# =============================================================================
# ===== 新增：球隊顏色分類函式 (Team Color Classification) =====
# =============================================================================
# ===== 可調參數 =====
_H_BIN = 10        # 色相量化的步長(hue180)
S_MIN = 30         # 有效彩度下限
V_MIN = 40         # 有效亮度下限
BLACK_V_MAX = 30   # 黑色上限
WHITE_S_MAX = 40   # 白色彩度上限
ROI_RESIZE = 60    # ROI 取樣大小

def _crop_torso_roi(image, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: 
        return None
    h_step = h / 10.0
    roi_y1 = int(y1 + 2 * h_step)
    roi_y2 = int(y1 + 5 * h_step)
    w_step = w / 5.0
    roi_x1 = int(x1 + 1 * w_step)
    roi_x2 = int(x1 + 4 * w_step)

    roi_x1 = max(x1, min(roi_x1, x2 - 1))
    roi_x2 = max(roi_x1 + 1, min(roi_x2, x2))
    roi_y1 = max(y1, min(roi_y1, y2 - 1))
    roi_y2 = max(roi_y1 + 1, min(roi_y2, y2))

    crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
    return crop if crop.size else None

def _palette_from_h(h_bin_center):
    h = h_bin_center #BGR
    if (h < 15) or (h >= 160):   
        return (0, 0, 255)     # Red
    if 15 <= h < 35:
        return (0, 255, 255)   # Yellow
    if 35 <= h < 85:
        return (0, 255, 0)     # Green
    if 85 <= h < 135:
        return (255, 0, 0)     # Blue
    return (255, 165, 0)       # Orange (or other)

def classify_team_color_by_hist(image, bbox):
    crop = _crop_torso_roi(image, bbox)
    if crop is None:
        return (0, 0, 0) # Default to black if crop fails

    crop = cv2.resize(crop, (ROI_RESIZE, ROI_RESIZE), interpolation=cv2.INTER_AREA)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)   # 0..179
    S = hsv[:, :, 1].astype(np.int32)   # 0..255
    V = hsv[:, :, 2].astype(np.int32)   # 0..255

    total = H.size
    if total == 0:
        return (0, 0, 0)

    black_mask = (V <= BLACK_V_MAX) 
    white_mask = (S <= WHITE_S_MAX)
    valid_color = (S >= S_MIN) & (V >= V_MIN) & (~black_mask) & (~white_mask)
    black_prop = float(black_mask.sum()) / total
    white_prop = float(white_mask.sum()) / total
    color_prop = 0.0
    palette_bgr = None
    if valid_color.any():
        h_valid = H[valid_color]
        bins = h_valid // _H_BIN
        counts = np.bincount(bins, minlength=180 // _H_BIN)
        best_bin = int(np.argmax(counts))
        color_prop = float(counts[best_bin]) / total
        h_center = int(best_bin * _H_BIN + _H_BIN // 2)
        palette_bgr = _palette_from_h(h_center)

    if black_prop >= white_prop and black_prop >= color_prop:
        return (0, 0, 0) # Black
    if white_prop >= black_prop and white_prop >= color_prop:
        return (255, 255, 255) # White
    
    return palette_bgr if palette_bgr is not None else (0, 0, 0)


def detect(save_img=False):
    source, weights, save_txt, imgsz, trace = opt.source, opt.weights, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    input_video = source.endswith('.mp4') or source.endswith('.avi') or source.endswith('.mkv')
    
    #!check the video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open the video")
    else:
        print("----影片可以成功讀取-------")
   
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'image' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    if half:
        model.half()  # to FP16
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
 
    if opt.write_log:
        with open('%s.txt' % opt.write_log, 'a') as f:
            f.write('Frame Person Ball Ball_Pos\n')
            
    results = {}
    
    if input_video:
        for _, _, _, vid_cap in dataset:
            vid_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=int(vid_len))
            break
    
    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        print(f'processing frame {idx}.....')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
               
        # Inference
        pred = model(img, augment=opt.augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            # =================== 新增：應用白平衡 ===================
            # 對 im0 (原始幀) 應用白平衡，以提高顏色分類的準確性
            # 注意：我們在 im0.copy() 上操作，以避免修改原始數據
            im0 = apply_white_balance(im0.copy())
            # =======================================================

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            img_path = str(save_dir / 'image' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            
            save_result = {}
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    
                    # =================== 修改：根據物件類別決定框線顏色 ===================
                    if names[int(cls)] == 'person':
                        # 如果是人，就進行球衣顏色分析來決定顏色
                        bbox = [int(c) for c in xyxy] # 轉換 bbox 格式
                        team_color = classify_team_color_by_hist(im0, bbox)
                        plot_one_box(xyxy, im0, label=label, color=team_color, line_thickness=2)
                    else:
                        # 如果不是人 (例如球)，就使用預設的類別顏色
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    # ====================================================================

                    # 處理儲存文字檔和結果的部分 (這部分邏輯不變)
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        
                        class_name = names[int(cls)]
                        if class_name not in save_result:
                            save_result[class_name] = []
                        save_result[class_name].append([[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])], conf.item()])
                        
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            results[str(idx)] = save_result
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
        if input_video:
            pbar.update(1)
            
    if input_video:
        pbar.close()
        
    if opt.save_json:
        with open(str(Path(save_path).parent / (Path(save_path).stem + '.json')), "w") as outfile:
            json.dump(results, outfile, indent=4)
            
    print(f"Results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    parser.add_argument('--write-log', default='')
    parser.add_argument('--save-json', action='store_true', help='save results to a JSON file')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()