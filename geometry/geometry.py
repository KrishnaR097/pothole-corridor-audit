import cv2
import numpy as np


def is_vehicle_hazard(pothole_box, vehicle_boxes, margin_x=20, shadow_margin_y=40):
    px1, py1, px2, py2 = pothole_box
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    for vx1, vy1, vx2, vy2 in vehicle_boxes:
        if (vx1 - margin_x <= pcx <= vx2 + margin_x) and (vy1 - 10 <= pcy <= vy2 + shadow_margin_y):
            return True
    return False


def compute_metric_dimensions(frame_gray, binary_mask, ground_dist_m, fx, fy, box):
    bx1, by1, bx2, by2 = box
    bw_px = max(1.0, float(bx2 - bx1))
    bh_px = max(1.0, float(by2 - by1))

    width_cm = float((bw_px * ground_dist_m / fx) * 100.0)
    length_cm = float((bh_px * ground_dist_m / fy) * 100.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    rim_mask = (dilated > 0) & (binary_mask == 0)

    crater_pixels = frame_gray[binary_mask > 0]
    rim_pixels = frame_gray[rim_mask]

    if len(crater_pixels) < 15 or len(rim_pixels) < 15:
        return width_cm, length_cm, 3.2, "Mild"

    rim_lum = float(np.median(rim_pixels))
    crater_darkness = float(np.percentile(crater_pixels, 20))
    optical_deficit = max(0.0, (rim_lum - crater_darkness) / max(rim_lum, 1.0))

    effective_diameter_cm = float(np.sqrt(max(0.1, width_cm * length_cm)))
    depth_cm = 2.0 + (effective_diameter_cm * 0.12) + (optical_deficit * 9.5)
    depth_cm = float(np.clip(depth_cm, 2.2, 12.0))

    if depth_cm < 4.5:
        severity = "Mild"
    elif depth_cm < 7.5:
        severity = "Moderate"
    else:
        severity = "Severe"

    return width_cm, length_cm, depth_cm, severity


def compute_severity_score(depth_cm, width_cm, length_cm, in_ego_lane, distance_m):
    area_cm2 = max(1.0, width_cm * length_cm)
    depth_pts = min(50.0, (depth_cm / 12.0) * 50.0)
    size_pts = min(25.0, (np.sqrt(area_cm2) / 80.0) * 25.0)
    lane_pts = 15.0 if in_ego_lane else 6.0
    prox_pts = max(0.0, 10.0 * (1.0 - min(distance_m, 30.0) / 30.0))
    return round(float(np.clip(depth_pts + size_pts + lane_pts + prox_pts, 5.0, 100.0)), 1)


def estimate_gps_point(base_lat, base_lon, distance_m, lateral_offset_m, step_fraction=0.0):
    forward_m = max(1.0, float(distance_m)) + (step_fraction * 500.0)
    d_lat = forward_m / 111320.0
    d_lon = float(lateral_offset_m) / max(1.0, 111320.0 * np.cos(np.radians(base_lat)))
    return round(base_lat + d_lat, 6), round(base_lon + d_lon, 6)