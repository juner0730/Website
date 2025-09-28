def merge_possession_changes(changes, window=13):
    if not changes: return []
    merged = []; i = 0
    while i < len(changes):
        start_f, _from, to = changes[i]
        last_f, last_to = start_f, to; j = i+1
        while j < len(changes) and (changes[j][0] - last_f) <= window:
            last_f, _from2, to2 = changes[j]; last_to = to2; j += 1
        merged.append({"start": start_f, "end": last_f, "to_id": last_to})
        i = j
    return merged

def build_possession_segments(
    merged_groups,
    total_frames,
    pre_margin=60,
    post_margin=60,
    join_window=15,  
    min_len_frames=60
):
    if not merged_groups:
        return []
    clusters = []
    cur = [merged_groups[0]]
    for g in merged_groups[1:]:
        prev = cur[-1]
        if g["start"] - prev["end"] <= join_window:
            cur.append(g)
        else:
            clusters.append(cur)
            cur = [g]
    clusters.append(cur) 
    segments = []
    for grp in clusters:
        first = grp[0]
        last  = grp[-1]
        seg_start = max(first["start"] - pre_margin, 0)
        seg_end   = min(last["end"] + post_margin, total_frames - 1)
        to_id     = last["to_id"] 
        seg_len = seg_end - seg_start + 1
        if seg_len < min_len_frames:
            need = min_len_frames - seg_len
            extend_end = min(seg_end + need, total_frames - 1)
            need -= (extend_end - seg_end)
            seg_end = extend_end
            if need > 0:
                seg_start = max(0, seg_start - need)

        segments.append({"start": seg_start, "end": seg_end, "to_id": to_id})

    return segments


def attach_goal_segments(goal_events, merged_groups, total_frames, margin=50, default_owner=None):
    out = []
    if not goal_events or not merged_groups: return out
    for gframe, _ball_id in goal_events:
        bucket = None
        for g in merged_groups:
            if g["start"] <= gframe <= g["end"]:
                bucket = g; break
        if bucket is not None:
            start = max(bucket["start"]-margin, 0)
            end   = min(gframe+margin, total_frames-1)
            out.append({"start": start, "end": end, "to_id": bucket["to_id"], "tag":"goal"})
        elif default_owner is not None:
            start = max(gframe-margin, 0)
            end   = min(gframe+margin, total_frames-1)
            out.append({"start": start, "end": end, "to_id": default_owner, "tag":"goal"})
    return out
