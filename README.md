# People Counting System — YOLOv8 + OpenCV

Real-time people counting using YOLOv8 and ByteTrack.  
**Bharati Vidyapeeth College of Engineering, Navi Mumbai | SE Sem IV Mini Project**

---

## What it does
- Detects people in real time using YOLOv8
- Tracks each person with a unique ID (ByteTrack)
- **Line Crossing mode** — counts entries, exits, and live occupancy
- **ROI mode** — counts people inside a defined zone
- Works with webcam and video files

## Setup

```bash
pip install -r requirements.txt
```

> YOLOv8 model weights are downloaded automatically on first run.

## How to run

```bash
# Line crossing with webcam
python main.py --source 0 --mode line

# Line crossing with video file
python main.py --source video.mp4 --mode line

# ROI mode with webcam
python main.py --source 0 --mode roi
```

## Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Save screenshot |

## Tech Stack
- Python 3.10+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy