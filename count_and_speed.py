"""Vehicle counting by direction + speed estimation from a fixed camera.

Pipeline (one frame at a time):
  OpenCV reads the frame -> YOLO detects vehicles -> ByteTrack (built into
  Ultralytics) gives each vehicle a stable ID -> a counting line records which direction it crossed ->
  a perspective transform (homography, OpenCV) maps pixels to metres on the
  road, so the distance a track moves per second becomes km/h.

The default video is the public "vehicles" sample from Roboflow Supervision
(highway, camera above the road). For an Astana intersection, record your
own video and pass --video plus new --source points / road size.

Usage:
    python count_and_speed.py                       # sample video
    python count_and_speed.py --video my.mp4 --source "x1,y1 x2,y2 x3,y3 x4,y4" --road-w 12 --road-l 40
"""
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

HERE = Path(__file__).parent
OUT = HERE / "results"
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids

# Calibration for the sample video (3840x2160): four corners of a road section
# in pixels and its real size. Values from the Roboflow speed-estimation tutorial.
SAMPLE_SOURCE = "1252,787 2298,803 5039,2159 -550,2159"
SAMPLE_ROAD_W, SAMPLE_ROAD_L = 25, 250


class PixelToMeters:
    """Homography: maps image points onto a top-down view of the road in metres."""

    def __init__(self, source, width_m, length_m):
        target = np.array([[0, 0], [width_m - 1, 0], [width_m - 1, length_m - 1], [0, length_m - 1]], dtype=np.float32)
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target)

    def __call__(self, points):
        if len(points) == 0:
            return points
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2)


def parse_points(text):
    return np.array([[float(v) for v in p.split(",")] for p in text.split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="path to video; default = Supervision sample")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--source", default=SAMPLE_SOURCE, help="4 road corners in pixels: 'x,y x,y x,y x,y'")
    ap.add_argument("--road-w", type=float, default=SAMPLE_ROAD_W, help="road section width, m")
    ap.add_argument("--road-l", type=float, default=SAMPLE_ROAD_L, help="road section length, m")
    ap.add_argument("--line-y", type=float, default=0.55, help="counting line height as share of frame")
    ap.add_argument("--out-width", type=int, default=1280, help="width of the saved annotated video")
    ap.add_argument("--device", default=None, help="'cpu' or GPU index; default = auto")
    a = ap.parse_args()

    if a.video is None:
        import os
        from supervision.assets import VideoAssets, download_assets
        data = Path(os.environ.get("CV_DATA", Path.home() / "cv-data")) / "videos"
        data.mkdir(parents=True, exist_ok=True)
        os.chdir(data)  # download_assets saves into the current folder
        download_assets(VideoAssets.VEHICLES)
        a.video = str(data / VideoAssets.VEHICLES.value)

    info = sv.VideoInfo.from_video_path(a.video)
    model = YOLO(a.weights)
    source = parse_points(a.source)
    to_m = PixelToMeters(source, a.road_w, a.road_l)
    zone = sv.PolygonZone(polygon=source.astype(int))
    y = int(info.height * a.line_y)
    line = sv.LineZone(start=sv.Point(0, y), end=sv.Point(info.width, y))

    history = defaultdict(lambda: deque(maxlen=int(info.fps)))  # last second of positions, metres
    speeds = defaultdict(list)
    classes = {}
    scale = a.out_width / info.width
    out_size = (a.out_width, int(info.height * scale))
    OUT.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(OUT / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, out_size)
    thickness = sv.calculate_optimal_line_thickness(info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(info.resolution_wh)
    box_ann = sv.BoxAnnotator(thickness=thickness)
    label_ann = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
    trace_ann = sv.TraceAnnotator(thickness=thickness, trace_length=int(info.fps * 2))
    line_ann = sv.LineZoneAnnotator(thickness=thickness, text_scale=text_scale)

    for frame in sv.get_video_frames_generator(a.video):
        # YOLO detection + ByteTrack tracking (persist=True keeps IDs between frames)
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3,
                             classes=list(VEHICLES), device=a.device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        if det.tracker_id is None:  # no vehicles in this frame
            det = sv.Detections.empty()
            det.tracker_id = np.array([], dtype=int)
        det = det[zone.trigger(det)]
        line.trigger(det)

        anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        labels = []
        for tid, cid, pt in zip(det.tracker_id, det.class_id, to_m(anchors)):
            classes[tid] = VEHICLES[int(cid)]
            history[tid].append(pt[1])
            label = f"#{tid}"
            if len(history[tid]) >= info.fps / 2:
                dist = abs(history[tid][-1] - history[tid][0])
                kmh = dist / ((len(history[tid]) - 1) / info.fps) * 3.6
                speeds[tid].append(kmh)
                label = f"#{tid} {kmh:.0f} km/h"
            labels.append(label)

        frame = sv.draw_polygon(frame, source.astype(int), sv.Color.YELLOW, thickness)
        frame = trace_ann.annotate(frame, det)
        frame = box_ann.annotate(frame, det)
        frame = label_ann.annotate(frame, det, labels)
        frame = line_ann.annotate(frame, line)
        writer.write(cv2.resize(frame, out_size))
    writer.release()

    rows = [{"track_id": int(t), "class": classes[t], "median_speed_kmh": round(float(np.median(s)), 1)}
            for t, s in speeds.items() if len(s) >= 3]
    with open(OUT / "vehicles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "class", "median_speed_kmh"])
        w.writeheader()
        w.writerows(rows)
    by_class = defaultdict(int)
    for r in rows:
        by_class[r["class"]] += 1
    summary = {
        "video": Path(a.video).name, "frames": info.total_frames, "fps": info.fps,
        "crossed_line_in": line.in_count, "crossed_line_out": line.out_count,
        "tracked_vehicles_with_speed": len(rows), "by_class": dict(by_class),
        "mean_speed_kmh": round(float(np.mean([r["median_speed_kmh"] for r in rows])), 1) if rows else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
