import csv

def open_jersey_csv(path):
    f = open(path, "w", newline='', encoding='utf-8')
    w = csv.writer(f); w.writerow(["frame_idx","player_id","number","confidences"])
    return f, w

def write_stats_txt(path, jersey_stats):
    with open(path, 'w', encoding='utf-8') as f:
        for pid, num_dict in jersey_stats.items():
            top5 = sorted(num_dict.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            entries = []
            for num, confs in top5:
                avg_conf = sum(confs)/len(confs) if confs else 0
                entries.append(f"{num}(avg_conf={avg_conf:.2f})")
            f.write(f"ID{pid} 前五名: " + ", ".join(entries) + "\n")
