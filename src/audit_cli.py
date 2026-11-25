import argparse
import os
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from geometry import (
    compute_metric_dimensions,
    compute_severity_score,
    estimate_gps_point,
    is_vehicle_hazard,
)
from analytics import generate_analytics_charts

VEHICLE_CLASSES = [2, 5, 7]
COLS = [
    "ID",
    "Severity",
    "Severity Score",
    "Depth (cm)",
    "Width (cm)",
    "Length (cm)",
    "Distance (m)",
    "Latitude",
    "Longitude",
    "In Ego Lane",
    "Source Video",
    "Timestamp",
]


def transcode_h264(src_mp4, dst_mp4):
    if not os.path.exists(src_mp4):
        return src_mp4
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_mp4,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            dst_mp4,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(dst_mp4) and os.path.getsize(dst_mp4) > 0:
            return dst_mp4
    except Exception:
        pass
    return src_mp4


def run_audit(video_path, route_name, distance_km, weights_path, out_dir, cam_h, cam_fov, cam_pitch):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pothole_model = YOLO(weights_path).to(device)
    vehicle_model = YOLO("yolov8n.pt").to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = max(1, min(60, int(cap.get(cv2.CAP_PROP_FPS)) or 30))
    total_frames = max(1.0, float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1.0))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    raw_mp4 = os.path.join(out_dir, "raw_annotated.mp4")
    final_mp4 = os.path.join(out_dir, "annotated_output.mp4")
    writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    road_half_w = 3.6
    ego_half_w = 1.75
    horizon_y = int(height * 0.48)
    min_det_y = int(height * 0.54)

    cx, cy = width / 2.0, height / 2.0
    fx = (width / 2.0) / np.tan(np.radians(max(20.0, cam_fov) / 2.0))
    fy = fx
    theta = np.radians(cam_pitch)

    base_lat, base_lon = 12.9716, 77.5946
    base_name = os.path.basename(video_path)
    smooth_tracks = {}
    seen_records = {}

    alert_msg = ""
    alert_color = (255, 255, 255)
    alert_frames = 0
    current_speed = 52.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        sec_stamp = round(frame_idx / float(fps), 2)
        progress_frac = min(1.0, frame_idx / total_frames)

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        veh_res = vehicle_model.predict(frame, conf=0.2, classes=VEHICLE_CLASSES, device=device, verbose=False)
        vehicle_boxes = (
            veh_res[0].boxes.xyxy.cpu().numpy()
            if (veh_res and veh_res[0].boxes is not None and len(veh_res[0].boxes) > 0)
            else []
        )

        results = pothole_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            iou=0.45,
            imgsz=640,
            device=device,
            classes=[0],
            verbose=False,
        )

        has_sev_7m = False
        has_mod_7m = False
        has_sev_30m = False
        closest_sev_7m = 999.0
        closest_mod_7m = 999.0
        closest_sev_30m = 999.0

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            if results[0].masks is not None and len(results[0].masks.data) > 0:
                masks = results[0].masks.data.cpu().numpy()
                for box, tid, mask_t in zip(boxes, track_ids, masks):
                    bx1, by1, bx2, by2 = box
                    bw, bh = bx2 - bx1, by2 - by1
                    if by2 < min_det_y or by1 < (horizon_y - 15):
                        continue
                    if is_vehicle_hazard(box, vehicle_boxes):
                        continue
                    if bw > (width * 0.55) or bh > (height * 0.35) or (bw * bh) < 150:
                        continue

                    mask_res = cv2.resize(mask_t, (width, height), interpolation=cv2.INTER_NEAREST)
                    bin_mask = (mask_res > 0.5).astype(np.uint8)
                    bin_mask[:horizon_y, :] = 0

                    v_coords, u_coords = np.where(bin_mask > 0)
                    if len(v_coords) < 20:
                        continue

                    v_bottom = float(np.max(v_coords))
                    u_center = float(np.mean(u_coords))
                    alpha_bottom = np.arctan((v_bottom - cy) / fy)
                    total_angle = max(theta + alpha_bottom, np.radians(1.2))
                    raw_dist = float(np.clip(cam_h / np.tan(total_angle), 0.5, 75.0))

                    curr_box = np.array(box, dtype=np.float32)
                    if tid in smooth_tracks:
                        prev_b = np.array(smooth_tracks[tid][:4], dtype=np.float32)
                        prev_dist = float(smooth_tracks[tid][4])
                        sb = 0.65 * curr_box + 0.35 * prev_b
                        s_dist = 0.65 * raw_dist + 0.35 * prev_dist
                    else:
                        sb = curr_box
                        s_dist = raw_dist

                    smooth_tracks[tid] = [float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]), float(s_dist)]
                    x1, y1, x2, y2 = sb.astype(int)

                    lat_offset_m = float((u_center - cx) * s_dist / fx)
                    if abs(lat_offset_m) > road_half_w:
                        continue

                    w_cm, l_cm, depth_cm, severity = compute_metric_dimensions(
                        frame_gray, bin_mask, s_dist, fx, fy, [x1, y1, x2, y2]
                    )
                    in_ego = abs(lat_offset_m) <= ego_half_w
                    sev_score = compute_severity_score(depth_cm, w_cm, l_cm, in_ego, s_dist)
                    lat_pt, lon_pt = estimate_gps_point(
                        base_lat, base_lon, s_dist, lat_offset_m, progress_frac * max(0.1, distance_km)
                    )

                    seen_records[int(tid)] = {
                        "ID": int(tid),
                        "Severity": severity,
                        "Severity Score": sev_score,
                        "Depth (cm)": round(depth_cm, 1),
                        "Width (cm)": round(w_cm, 1),
                        "Length (cm)": round(l_cm, 1),
                        "Distance (m)": round(s_dist, 1),
                        "Latitude": lat_pt,
                        "Longitude": lon_pt,
                        "In Ego Lane": bool(in_ego),
                        "Source Video": base_name,
                        "Timestamp": f"{sec_stamp}s",
                    }

                    if in_ego:
                        if s_dist <= 7.0:
                            if severity == "Severe":
                                has_sev_7m = True
                                closest_sev_7m = min(closest_sev_7m, s_dist)
                            elif severity == "Moderate":
                                has_mod_7m = True
                                closest_mod_7m = min(closest_mod_7m, s_dist)
                        elif s_dist <= 30.0 and severity == "Severe":
                            has_sev_30m = True
                            closest_sev_30m = min(closest_sev_30m, s_dist)

                    color = (0, 0, 255) if severity == "Severe" else (0, 255, 255) if severity == "Moderate" else (0, 255, 0)
                    overlay = np.zeros_like(frame, dtype=np.uint8)
                    overlay[bin_mask > 0] = color
                    frame = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    tag = f"ID:{tid} | {severity} | D:{depth_cm:.1f}cm | W:{w_cm:.0f}cm L:{l_cm:.0f}cm | Dist:{s_dist:.1f}m"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.22, 1)
                    by1_box = max(0, y1 - th - 6)
                    by2_box = max(0, y1)
                    cv2.rectangle(frame, (x1, by1_box), (x1 + tw + 6, by2_box), color, -1)
                    cv2.putText(frame, tag, (x1 + 3, by2_box - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 0, 0), 1, cv2.LINE_AA)

        if has_sev_7m:
            alert_msg = f"ALERT: SEVERE POTHOLE AHEAD! ({closest_sev_7m:.1f}m)"
            alert_color = (0, 0, 255)
            alert_frames = fps
        elif has_mod_7m and not has_sev_7m:
            alert_msg = f"ALERT: MODERATE POTHOLE AHEAD! ({closest_mod_7m:.1f}m)"
            alert_color = (0, 165, 255)
            alert_frames = fps
        elif has_sev_30m and alert_frames == 0:
            alert_msg = f"POTHOLE AHEAD IN {closest_sev_30m:.0f} METERS (SEVERE)"
            alert_color = (0, 140, 255)
            alert_frames = fps

        if alert_frames > 0:
            (aw, _), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ax = max(10, (width - aw) // 2)
            cv2.putText(frame, alert_msg, (ax, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, alert_msg, (ax, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 2, cv2.LINE_AA)
            alert_frames -= 1

        target_spd = 51.0 if (has_sev_7m or has_mod_7m or has_sev_30m) else 57.0
        current_speed = float(np.clip(current_speed + (target_spd - current_speed) * 0.05, 50.0, 60.0))
        spd_text = f"SPD: {int(round(current_speed))} km/h"
        (sw, _), _ = cv2.getTextSize(spd_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        sx = max(10, width - sw - 25)
        cv2.putText(frame, spd_text, (sx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, spd_text, (sx, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 40, 0), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    out_video = transcode_h264(raw_mp4, final_mp4)
    if os.path.exists(raw_mp4) and raw_mp4 != out_video:
        shutil.copyfile(out_video, final_mp4)

    df = pd.DataFrame(list(seen_records.values()), columns=COLS) if seen_records else pd.DataFrame(columns=COLS)
    csv_path = os.path.join(out_dir, "municipal_work_order.csv")
    df.to_csv(csv_path, index=False)

    chart_path = generate_analytics_charts(df, out_path=os.path.join(out_dir, "road_safety_analytics.png"))
    print(f"Route: {route_name} ({distance_km:.2f} km) | Detections: {len(df)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Chart: {chart_path}")
    print(f"Saved Video: {final_mp4}")


def main():
    parser = argparse.ArgumentParser(description="Run pothole audit CLI on a dashcam video.")
    parser.add_argument("--video", required=True, help="Path to input dashcam video")
    parser.add_argument("--route", required=True, help="Route name (e.g., 'RVCE to KR Market, Bengaluru')")
    parser.add_argument("--distance-km", type=float, default=2.5, help="Route distance in km")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained YOLOv8 weights")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--cam-height", type=float, default=1.2)
    parser.add_argument("--cam-fov", type=float, default=65.0)
    parser.add_argument("--cam-pitch", type=float, default=10.0)
    args = parser.parse_args()

    run_audit(
        video_path=args.video,
        route_name=args.route,
        distance_km=args.distance_km,
        weights_path=args.weights,
        out_dir=args.out_dir,
        cam_h=args.cam_height,
        cam_fov=args.cam_fov,
        cam_pitch=args.cam_pitch,
    )


if __name__ == "__main__":
    main()