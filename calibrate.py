"""Helper for a new video: saves the first frame with a pixel grid.

Open results/calibration_grid.jpg, pick four corners of a road section whose
real size you know (e.g. lane markings, a pedestrian crossing: lane ~3.5 m,
dashed line + gap ~9-12 m), and pass them to count_and_speed.py:

    python calibrate.py --video my_intersection.mp4
    python count_and_speed.py --video my_intersection.mp4 \
        --source "x1,y1 x2,y2 x3,y3 x4,y4" --road-w 10.5 --road-l 30

Point order: top-left, top-right, bottom-right, bottom-left (as seen in the frame).
"""
import argparse
from pathlib import Path

import cv2

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--step", type=int, default=100, help="grid step in pixels")
    a = ap.parse_args()
    ok, frame = cv2.VideoCapture(a.video).read()
    if not ok:
        raise SystemExit("cannot read the video")
    h, w = frame.shape[:2]
    for x in range(0, w, a.step):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for y in range(0, h, a.step):
        cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(frame, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "calibration_grid.jpg"), frame)
    print(f"{w}x{h}, saved {OUT / 'calibration_grid.jpg'}")


if __name__ == "__main__":
    main()
