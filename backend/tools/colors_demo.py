import json

def load_people_bboxes(json_path):
    """
    先嘗試使用「主程式」的解析器（如果你在 app.py 暴露了對應函式）。
    若沒有，使用高度容錯的 fallback。
    輸出格式：frames -> List[List[(x1,y1,x2,y2), ...]]
    """
    # 1) 先試主程式
    app_parser = try_import_app_parser()
    if app_parser is not None:
        frames = app_parser(json_path)
        if frames and isinstance(frames, list):
            return frames

    # 2) fallback：盡量自動適配常見鍵名/結構
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _as_xyxy(box):
        # 吃 [x,y,w,h] 或 [x1,y1,x2,y2] 或 dict 形式
        if box is None: 
            return None
        # dict 形式：{"x1":...,"y1":...,"x2":...,"y2":...} 或 {"x":...,"y":...,"w":...,"h":...}
        if isinstance(box, dict):
            if all(k in box for k in ("x1","y1","x2","y2")):
                return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            if all(k in box for k in ("x","y","w","h")):
                x,y,w,h = box["x"], box["y"], box["w"], box["h"]
                return int(x), int(y), int(x+w), int(y+h)
            if "xyxy" in box and isinstance(box["xyxy"], (list,tuple)) and len(box["xyxy"])==4:
                x1,y1,x2,y2 = box["xyxy"]
                return int(x1),int(y1),int(x2),int(y2)
            if "xywh" in box and isinstance(box["xywh"], (list,tuple)) and len(box["xywh"])==4:
                x,y,w,h = box["xywh"]
                return int(x),int(y),int(x+w),int(y+h)
            # 常見 key 混用
            bb = box.get("bbox") or box.get("box")
            if bb and isinstance(bb, (list,tuple)) and len(bb)==4:
                x,y,w,h = bb
                # 嘗試推斷是 xywh 還是 xyxy
                if w>=x and h>=y and w<4096 and h<4096:
                    return int(x),int(y),int(x+w),int(y+h)
                return int(x),int(y),int(w),int(h)
            return None

        # list/tuple 形式
        if isinstance(box, (list,tuple)) and len(box)==4:
            x,y,w,h = box
            # 嘗試推斷 xywh 或 xyxy
            if w>=x and h>=y and w<4096 and h<4096:
                return int(x),int(y),int(x+w),int(y+h)
            return int(x),int(y),int(w),int(h)
        return None

    def _is_person(d):
        # 支援多種人類類別標記
        cls_name = d.get("cls") or d.get("class") or d.get("label") or d.get("category") or d.get("name")
        cls_id   = d.get("category_id") or d.get("id") or d.get("class_id")
        # 字串類別
        if isinstance(cls_name, str):
            if cls_name.lower() in ("person","player","human"):
                return True
        # 數字類別（COCO: person=0）
        if isinstance(cls_id, int) and cls_id in (0,1):  # 有些模型把 person=1
            # 如果你的主程式是 0=person，改成 {0} 即可
            return True
        return False

    frames = []

    # 形態 1：{"frames":[{"frame_id":0,"detections":[{...}]}, ...]}
    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        for fr in data["frames"]:
            dets = fr.get("detections") or fr.get("objects") or fr.get("players") or fr.get("people") or []
            boxes = []
            for d in dets:
                if not isinstance(d, dict):
                    continue
                if not _is_person(d):
                    # 若來源沒有分類資訊，可放寬直接取所有 bbox
                    pass
                bb = d.get("xyxy") or d.get("xywh") or d.get("bbox") or d.get("box") or d.get("rect")
                xyxy = _as_xyxy(bb)
                if xyxy: boxes.append(xyxy)
            frames.append(boxes)
        if frames:
            return frames

    # 形態 2：{ "0":[{...},...], "1":[{...},...] } 以 frame 索引為 key
    if isinstance(data, dict):
        try:
            keys = sorted([int(k) for k in data.keys()])
            for k in keys:
                dets = data[str(k)]
                boxes = []
                if isinstance(dets, list):
                    for d in dets:
                        if not isinstance(d, dict):
                            continue
                        if not _is_person(d):
                            pass
                        bb = d.get("xyxy") or d.get("xywh") or d.get("bbox") or d.get("box") or d.get("rect")
                        xyxy = _as_xyxy(bb)
                        if xyxy: boxes.append(xyxy)
                frames.append(boxes)
            if frames:
                return frames
        except Exception:
            pass

    # 形態 3：list，每項代表一幀
    if isinstance(data, list):
        for dets in data:
            boxes = []
            if isinstance(dets, dict) and "detections" in dets:
                dets = dets["detections"]
            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    if not _is_person(d):
                        pass
                    bb = d.get("xyxy") or d.get("xywh") or d.get("bbox") or d.get("box") or d.get("rect")
                    xyxy = _as_xyxy(bb)
                    if xyxy: boxes.append(xyxy)
            frames.append(boxes)
        if frames:
            return frames

    raise ValueError("無法解析 TRACKING_JSON 格式；請在 load_people_bboxes() 內對應你的鍵名/結構，或在 app.py 提供解析函式讓我重用。")
