import csv
import os

def save_tracking_csv(results, save_path):
    """
    results: list of dict
    Each dict keys:
        - frame_id
        - track_id (or global_id)
        - bbox [x1, y1, x2, y2]
        - cam_id
        - scenario
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "cam_id",
            "frame_id",
            "track_id",
            "x1", "y1", "x2", "y2"
        ])

        for r in results:
            x1,y1,x2,y2 = r["bbox"]
            writer.writerow([
                r["scenario"],
                r["cam_id"],
                r["frame_id"],
                r.get("global_id", r.get("track_id", -1)),
                x1, y1, x2, y2
            ])

    print(f"[EXPORT] CSV saved to {save_path}")


def print_val_summary(summary, title="Validation"):
    """
    summary: MOT metrics DataFrame
    """
    try:
        idf1 = float(summary.loc['acc', 'idf1'])
        mota = float(summary.loc['acc', 'mota'])
        idsw = int(summary.loc['acc', 'num_switches'])
        fn = int(summary.loc['acc', 'num_misses'])
        fp = int(summary.loc['acc', 'num_false_positives'])
    except:
        print(f"[{title}] Invalid summary structure:")
        print(summary)
        return

    print(f"\n========== {title} Summary ==========")
    print(f"IDF1 : {idf1:.3f}")
    print(f"MOTA : {mota:.3f}")
    print(f"IDSW : {idsw}")
    print(f"FN   : {fn}")
    print(f"FP   : {fp}")
    print("=====================================\n")
