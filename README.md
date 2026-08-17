# telemouse

Control the mouse pointer with your hand, through a webcam. One file,
[`telemouse.py`](telemouse.py).

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger up, middle down | Move the pointer |
| ...thumb touches index | Left click; keep them together to drag |
| ...two quick pinches | Double click |
| Index + middle up | Scroll (locks to the axis you start on) |
| ...thumb touches middle | Right click |
| Open palm, fist, or hand out of view | Idle — the pointer stays put |

Keys in the preview window: `q`/`esc` quit, `p` pause, `h` help, `d` dry-run,
`r` recentre, `[` / `]` smoothing.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The 7.8 MB landmark model is already in the repo; if it is missing, the script
downloads it on first run.

On macOS, grant your terminal (or IDE) two permissions in System Settings →
Privacy & Security, or nothing will happen:

- **Camera** — otherwise the script exits with "Could not open camera 0".
- **Accessibility** — otherwise pointer events are silently swallowed: the
  preview tracks your hand perfectly and the cursor never moves.

Grant them to the **app you launch from** (Terminal, iTerm, VS Code…), not to
"python", and fully quit that app afterwards — macOS only re-reads the
permission when the process restarts.

`--doctor` tells you which of the two is missing:

```bash
.venv/bin/python telemouse.py --doctor
```

It nudges the cursor eight points, reads the position back, and puts it where
it found it. If the nudge doesn't land, the events are being dropped and it
prints exactly what to switch on.

## Run

```bash
.venv/bin/python telemouse.py
```

Check the camera and model without touching the pointer:

```bash
.venv/bin/python telemouse.py --selftest
```

Watch the tracking with the pointer disabled — the best way to tune thresholds,
but note it will never move the cursor:

```bash
.venv/bin/python telemouse.py --dry-run
```

`--help` lists every knob. The ones worth touching:

| Flag | Default | Effect |
| --- | --- | --- |
| `--min-cutoff` | 1.5 | Lower = steadier pointer at rest, more lag |
| `--beta` | 0.01 | Higher = less lag during fast movement |
| `--pad` | 0.12 | How much of the frame edge to trim from the active area |
| `--pinch-on` / `--pinch-off` | 0.42 / 0.60 | Pinch sensitivity, as a fraction of your palm length |
| `--scroll-gain` | 2600 | Scroll pixels per unit of hand travel |
| `--invert-scroll` | off | Flip the scroll direction |
| `--camera` | 0 | Try `1` if the wrong camera opens |

## How it works

Frames come off the webcam on a background thread that keeps only the newest
one, so the pointer never lags behind buffered frames. MediaPipe's
HandLandmarker runs in `VIDEO` mode, which tracks between frames instead of
re-detecting from scratch. The 21 landmarks then feed a small gesture layer:

- **Everything is measured in units of palm length** (wrist to middle knuckle),
  in pixels rather than normalised coordinates, so thresholds hold whether your
  hand is close to the lens or across the room, and the frame's aspect ratio
  doesn't skew distances.
- **Finger extension is judged by distance from the wrist**, not by screen `y`,
  so a tilted or rotated hand still reads correctly.
- **Every threshold latches** (Schmitt trigger): a value hovering on the edge
  cannot chatter between states, and the mode has to hold for two frames before
  it switches. A drag also pins the mode, so one misread frame can't drop what
  you are dragging.
- **The pointer is smoothed by a 1-euro filter** — heavy smoothing when your
  hand is still, light when it moves fast, and frame-rate independent, so it
  feels the same at 15 fps and at 60.
- **The active area is letterboxed to your screen's aspect ratio** and inset
  from the frame edge, where tracking degrades.
- On macOS, pointer events go straight to Quartz: pixel-precise scrolling, real
  double-click events, and much lower latency than pyautogui. Other platforms
  fall back to pyautogui, with sub-notch scrolling accumulated rather than lost.

A held button is always released — when the hand leaves the frame, when you
pause, when the mode changes, and on exit — so the script cannot leave your
mouse stuck down.

## Notes

- `mediapipe` is pinned below `1.0`. On macOS, 1.0.x aborts with
  `graph_service.h: Check failed: service_ Service is unavailable` inside
  `DrishtiMetalHelper`, on both the CPU and GPU delegate.
- The pointer maps to the main display only.
- `test.py` and `test2.py` are the earlier prototypes, superseded by
  `telemouse.py`.
