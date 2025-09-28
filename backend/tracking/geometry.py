def bboxes_intersect(b1, b2):
    x1, y1, x2, y2 = b1
    a1, b1y, a2, b2y = b2
    return not (x2 < a1 or a2 < x1 or y2 < (b1y + b2y)//2 or b2y < y1)

def bbox_fully_inside(inner_bbox, outer_bbox):
    x1, y1, x2, y2 = inner_bbox
    gx1, gy1, gx2, gy2 = outer_bbox
    for x, y in [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]:
        if not (gx1 <= x <= gx2 and gy1 <= y <= gy2):
            return False
    return True
