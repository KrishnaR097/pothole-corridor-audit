import argparse
import json
import os
import re
import shutil
from datetime import datetime
import pandas as pd


def safe_slug(name_str):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name_str or "route").strip()).strip("_")
    return slug or "route"


def create_complaint(route_name, distance_km, audit_dir="outputs", complaints_dir="complaints"):
    slug = safe_slug(route_name)
    bundle_dir = os.path.join(complaints_dir, slug)
    os.makedirs(bundle_dir, exist_ok=True)

    src_csv = os.path.join(audit_dir, "municipal_work_order.csv")
    src_vid = os.path.join(audit_dir, "annotated_output.mp4")
    src_chart = os.path.join(audit_dir, "road_safety_analytics.png")

    records = []
    dst_csv = os.path.join(bundle_dir, f"{slug}_work_order.csv")
    if os.path.exists(src_csv):
        df = pd.read_csv(src_csv)
        df.to_csv(dst_csv, index=False)
        records = df.to_dict(orient="records")

    dst_vid = os.path.join(bundle_dir, f"{slug}_annotated_video.mp4") if os.path.exists(src_vid) else None
    if dst_vid:
        shutil.copy2(src_vid, dst_vid)

    dst_chart = os.path.join(bundle_dir, f"{slug}_analytics_pie.png") if os.path.exists(src_chart) else None
    if dst_chart:
        shutil.copy2(src_chart, dst_chart)

    payload = {
        "complaint_name": route_name,
        "route_distance_km": float(distance_km),
        "raised_at": datetime.now().isoformat(),
        "saved_files": {
            "csv_work_order": dst_csv if os.path.exists(dst_csv) else None,
            "annotated_video": dst_vid,
            "analytics_chart": dst_chart,
        },
        "pothole_records": records,
    }

    dst_json = os.path.join(bundle_dir, f"{slug}_complaint.json")
    with open(dst_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved complaint bundle to: {bundle_dir}")


def main():
    parser = argparse.ArgumentParser(description="Package audit artifacts into a route-named complaint folder.")
    parser.add_argument("--route", required=True, help="Route name")
    parser.add_argument("--distance-km", type=float, default=2.5, help="Route distance in km")
    parser.add_argument("--audit-dir", default="outputs", help="Directory containing audit outputs")
    parser.add_argument("--complaints-dir", default="complaints", help="Target directory for saved complaints")
    args = parser.parse_args()
    create_complaint(args.route, args.distance_km, args.audit_dir, args.complaints_dir)


if __name__ == "__main__":
    main()