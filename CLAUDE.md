# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CourtSense MVP is a tennis recovery speed analyzer that uses computer vision (SAM2) to track players on court and analyze their recovery patterns after shots. It outputs coaching feedback based on recovery times.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with default video (data/input_video.mp4)
python main.py

# Run headless (no display windows)
python main.py --no-display --auto-court --auto-bbox

# Run with automatic court detection (Hough-based)
python main.py --auto-court-detector

# Run with manual bounding box
python main.py --bbox 100,200,150,300

# Limit frames processed (avoid OOM on long videos)
python main.py --max-frames 200

# Skip Ollama LLM feedback (use heuristic advice instead)
python main.py --no-ollama
```

## Architecture

The pipeline processes tennis video through these stages:

1. **Court Calibration** (`modules/calibration.py`, `modules/auto_calibrate.py`)
   - Interactive: User clicks 4 singles court corners (TL, TR, BR, BL order)
   - Auto modes: `--auto-court` (frame bounds) or `--auto-court-detector` (Hough/color detection)
   - Computes homography matrix mapping pixels to real-world meters (8.23m x 23.77m court)

2. **Player Tracking** (`modules/tracker.py`)
   - Uses SAM2 (Segment Anything 2) video predictor with MPS/CPU support
   - Patches SAM2's video loader to use OpenCV instead of decord
   - Returns bottom-center centroid of player mask per frame

3. **Motion Analysis** (`modules/geometry.py`)
   - Converts pixel coordinates to meters via homography
   - Detects shots by velocity-x sign flips combined with speed dips
   - Computes recovery time: frames until player returns within 1.5m of ideal position (4.11m, 23.77m)

4. **Coaching Feedback** (`modules/coach.py`)
   - Sends recovery stats to Ollama (llama3) for LLM-generated advice
   - Falls back to heuristic advice if Ollama unavailable

## Key Data Flow

```
Video -> First Frame -> Calibration (homography) -> SAM2 Tracking -> Pixel Centroids
                                                                          |
                                                                          v
                                    Meter Coords <- homography transform <- Fill missing frames
                                          |
                                          v
                            Shot Detection -> Recovery Stats -> Coach Feedback
```

## Output Files (in `data/`)

- `coords_m.npy`: Player positions in meters per frame
- `coords_px.npy`: Raw pixel centroids
- `coords_stats.txt`: Summary statistics
- `topdown_path.png`: Bird's-eye view visualization
- `auto_court_debug.png`: Court detection debug image
- `manual_court_debug.png`: Manual calibration confirmation

## Device Support

The tracker automatically selects MPS (Apple Silicon) if available, falling back to CPU. The SAM2 model downloads from HuggingFace on first run.

## Court Constants

- Singles court: 8.23m wide x 23.77m long
- Doubles court: 10.97m wide (used to derive singles from doubles detection)
- Ideal recovery position: (4.11m, 23.77m) - center of baseline
