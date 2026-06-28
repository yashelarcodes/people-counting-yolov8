"""
People Counting System using YOLOv8 and OpenCV
Bharati Vidyapeeth College of Engineering - Mini Project
Supports: Webcam + Video File | ROI Counting + Line Crossing Counting
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these as needed)
# ─────────────────────────────────────────────

# Input source: 0 = webcam, or give a video file path like "video.mp4"
SOURCE = 0

# Counting mode: "roi" or "line"
MODE = "line"

# Line-crossing config (fraction of frame: 0.0 = top, 1.0 = bottom)
LINE_POSITION = 0.5          # horizontal line at 50% height
LINE_ORIENTATION = "horizontal"  # "horizontal" or "vertical"

# ROI polygon – will be set dynamically based on frame size
# You can hard-code pixel coordinates here if you want, e.g.:
# ROI_POINTS = np.array([[100, 200], [500, 200], [500, 450], [100, 450]])
ROI_POINTS = None            # None = auto-generate centered rectangle

# YOLOv8 model – 'yolov8n.pt' is smallest/fastest; use 'yolov8s.pt' for better accuracy
MODEL_NAME = "yolov8n.pt"

# Detection confidence threshold
CONFIDENCE = 0.4

# ─────────────────────────────────────────────


def get_centroid(box):
    """Return (cx, cy) centre of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_dashed_line(frame, pt1, pt2, color, thickness=2, dash_len=15):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = max(1, int(np.hypot(dx, dy)))
    for i in range(0, length, dash_len * 2):
        s = i / length
        e = min((i + dash_len) / length, 1.0)
        sx, sy = int(x1 + s * dx), int(y1 + s * dy)
        ex, ey = int(x1 + e * dx), int(y1 + e * dy)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness)


def draw_roi(frame, roi_pts, color=(0, 255, 128)):
    """Draw a semi-transparent ROI polygon on the frame."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_pts], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [roi_pts], True, color, 2)


def draw_label(frame, text, pos, bg_color=(30, 30, 30), text_color=(255, 255, 255), scale=0.6):
    """Draw a filled-background text label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, text_color, 2, cv2.LINE_AA)


def draw_hud(frame, counts, mode):
    """Draw the top-left HUD showing live counts."""
    h, w = frame.shape[:2]
    # Semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (280, 160), (0, 200, 100), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "PEOPLE COUNTER", (20, 38), font, 0.65, (0, 220, 120), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 45), (270, 45), (60, 60, 60), 1)

    if mode == "line":
        cv2.putText(frame, f"Entries  : {counts['entries']}", (20, 72), font, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Exits    : {counts['exits']}", (20, 100), font, 0.6, (100, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Occupancy: {counts['occupancy']}", (20, 128), font, 0.6, (255, 220, 50), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detected : {counts['detected']}", (20, 156), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"In ROI   : {counts['in_roi']}", (20, 80), font, 0.65, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detected : {counts['detected']}", (20, 112), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def run(source, mode, model_name, confidence, line_pos, line_orient, roi_pts_config):
    print(f"\n  [INFO] Loading model: {model_name}")
    model = YOLO(model_name)
    print(f"  [INFO] Model loaded. Starting capture from: {source}")
    print(f"  [INFO] Mode: {mode.upper()}")
    print(f"  [INFO] Press 'Q' to quit | Press 'S' to save screenshot\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open source: {source}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("  [ERROR] Cannot read from source.")
        sys.exit(1)

    H, W = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Build ROI polygon ──
    if mode == "roi":
        if roi_pts_config is not None:
            roi_pts = roi_pts_config
        else:
            # Default: centred rectangle covering ~60% of frame
            margin_x, margin_y = int(W * 0.2), int(H * 0.2)
            roi_pts = np.array([
                [margin_x,         margin_y],
                [W - margin_x,     margin_y],
                [W - margin_x,     H - margin_y],
                [margin_x,         H - margin_y]
            ], dtype=np.int32)

    # ── Build counting line ──
    if mode == "line":
        if line_orient == "horizontal":
            line_y = int(H * line_pos)
            line_pt1 = (0, line_y)
            line_pt2 = (W, line_y)
        else:
            line_x = int(W * line_pos)
            line_pt1 = (line_x, 0)
            line_pt2 = (line_x, H)

    # ── Tracking state ──
    prev_positions = {}      # id -> (cx, cy)
    crossed_ids    = set()   # ids that already crossed once
    counts = {"entries": 0, "exits": 0, "occupancy": 0, "detected": 0, "in_roi": 0}

    screenshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video file back to start
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # ── YOLOv8 detection + tracking ──
        results = model.track(
            frame,
            persist=True,
            classes=[0],          # 0 = person
            conf=confidence,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detected_count = 0
        in_roi_count   = 0

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes  = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids    = results[0].boxes.id.cpu().numpy().astype(int)
            confs  = results[0].boxes.conf.cpu().numpy()
            detected_count = len(boxes)

            for box, tid, conf_val in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                cx, cy = get_centroid(box)

                # ── Draw bounding box ──
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                # Centroid dot
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)
                # ID label
                draw_label(frame, f"ID:{tid}", (x1, y1 - 8), bg_color=(0, 100, 180))

                # ── ROI logic ──
                if mode == "roi":
                    inside = cv2.pointPolygonTest(roi_pts, (cx, cy), False)
                    if inside >= 0:
                        in_roi_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)

                # ── Line-crossing logic ──
                if mode == "line":
                    if tid in prev_positions:
                        pcx, pcy = prev_positions[tid]

                        if line_orient == "horizontal":
                            crossed = (pcy < line_y <= cy) or (pcy > line_y >= cy)
                            going_down = cy > pcy
                        else:
                            crossed = (pcx < line_x <= cx) or (pcx > line_x >= cx)
                            going_down = cx > pcx

                        if crossed and tid not in crossed_ids:
                            crossed_ids.add(tid)
                            if going_down:
                                counts["entries"] += 1
                            else:
                                counts["exits"] += 1
                            counts["occupancy"] = max(0, counts["entries"] - counts["exits"])
                            # Flash effect on crossing
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    prev_positions[tid] = (cx, cy)

        # Clean up stale IDs from prev_positions
        if mode == "line" and results[0].boxes is not None and results[0].boxes.id is not None:
            active_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
            stale = [k for k in prev_positions if k not in active_ids]
            for k in stale:
                del prev_positions[k]
            # Allow re-crossing after person leaves frame
            crossed_ids -= (crossed_ids - active_ids)

        counts["detected"] = detected_count
        counts["in_roi"]   = in_roi_count

        # ── Draw overlays ──
        if mode == "roi":
            draw_roi(frame, roi_pts)
            draw_label(frame, "ROI ZONE", (roi_pts[0][0], roi_pts[0][1] - 10),
                       bg_color=(0, 160, 80))

        if mode == "line":
            draw_dashed_line(frame, line_pt1, line_pt2, (0, 255, 255), 3)
            mid = (W // 2, line_y if line_orient == "horizontal" else H // 2)
            draw_label(frame, "COUNTING LINE", (mid[0] - 60, mid[1] - 12),
                       bg_color=(0, 160, 160))
            # Direction arrows
            if line_orient == "horizontal":
                cv2.arrowedLine(frame, (W - 60, line_y - 25), (W - 60, line_y - 5),
                                (100, 255, 100), 2, tipLength=0.4)
                cv2.putText(frame, "IN", (W - 45, line_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                cv2.arrowedLine(frame, (W - 60, line_y + 5), (W - 60, line_y + 25),
                                (100, 180, 255), 2, tipLength=0.4)
                cv2.putText(frame, "OUT", (W - 45, line_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 255), 1)

        draw_hud(frame, counts, mode)

        # ── FPS display ──
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            cv2.putText(frame, f"FPS: {fps:.0f}", (W - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("People Counting System | Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n}.png"
            cv2.imwrite(fname, frame)
            print(f"  [INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  ── Final Count ──")
    if mode == "line":
        print(f"  Entries  : {counts['entries']}")
        print(f"  Exits    : {counts['exits']}")
        print(f"  Occupancy: {counts['occupancy']}")
    else:
        print(f"  Max in ROI (last frame): {counts['in_roi']}")
    print()


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Counting System – YOLOv8 + OpenCV")
    parser.add_argument("--source", default=None,
                        help="Video source: 0 for webcam, or path to video file (default: webcam)")
    parser.add_argument("--mode", choices=["roi", "line"], default=MODE,
                        help="Counting mode: roi or line (default: line)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help="YOLOv8 model name (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help="Confidence threshold 0-1 (default: 0.4)")
    parser.add_argument("--line-pos", type=float, default=LINE_POSITION,
                        help="Counting line position as fraction 0-1 (default: 0.5)")
    parser.add_argument("--line-orient", choices=["horizontal", "vertical"],
                        default=LINE_ORIENTATION,
                        help="Line orientation (default: horizontal)")
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = SOURCE
    elif source.isdigit():
        source = int(source)

    run(
        source      = source,
        mode        = args.mode,
        model_name  = args.model,
        confidence  = args.conf,
        line_pos    = args.line_pos,
        line_orient = args.line_orient,
        roi_pts_config = ROI_POINTS
    )
