#!/usr/bin/env python3
"""telemouse - control the mouse pointer with your hand, via a webcam.

Gestures (right or left hand, mirrored preview):

    index up, middle down .............. move the pointer
    ... + thumb touches index .......... left click / hold to drag
    ... + two quick pinches ............ double click
    index + middle up .................. scroll (axis locks on first motion)
    ... + thumb touches middle ......... right click
    open palm / fist / no hand ......... idle, pointer parked

Keys in the preview window:  q or esc quit  .  p pause  .  h help  .  d dry-run
                             r recentre     .  [ / ] smoothing

Run `python telemouse.py --selftest` to check the camera and model without
touching the pointer.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(path: str) -> str:
    """Return a path to the landmark model, downloading it once if needed."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    tmp = path + ".part"
    print(f"Downloading hand landmark model to {path} ...", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 // total:3d}%", end="", flush=True)
        print("\r  done." if total else "  done.")
    except (urllib.error.URLError, OSError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"Could not download the model ({exc}).\n"
            f"Download it manually from {MODEL_URL} and save it as {path}."
        ) from exc

    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# Pointer back ends
# --------------------------------------------------------------------------


class PointerBackend:
    """Minimal pointer interface. Coordinates are screen points, origin top-left."""

    name = "none"

    def size(self) -> tuple[int, int]:
        raise NotImplementedError

    def position(self) -> tuple[float, float] | None:
        """Where the pointer actually is, or None if the back end cannot say.

        Reading needs no permission, so comparing this against a move we just
        made is how --doctor tells "events sent" from "events delivered".
        """
        return None

    def move(self, x: float, y: float, dragging: bool = False) -> None: ...
    def button_down(self, button: str = "left", clicks: int = 1) -> None: ...
    def button_up(self, button: str = "left", clicks: int = 1) -> None: ...
    def scroll(self, dx_px: float, dy_px: float) -> None: ...

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self.button_down(button, clicks)
        self.button_up(button, clicks)


class QuartzBackend(PointerBackend):
    """Native macOS back end. Far lower latency than pyautogui and supports
    pixel-precise (smooth) scrolling and real double-click events."""

    name = "quartz"

    def __init__(self) -> None:
        import Quartz  # noqa: PLC0415 - optional, platform specific

        self.q = Quartz
        self._buttons = {
            "left": (
                Quartz.kCGEventLeftMouseDown,
                Quartz.kCGEventLeftMouseUp,
                Quartz.kCGEventLeftMouseDragged,
                Quartz.kCGMouseButtonLeft,
            ),
            "right": (
                Quartz.kCGEventRightMouseDown,
                Quartz.kCGEventRightMouseUp,
                Quartz.kCGEventRightMouseDragged,
                Quartz.kCGMouseButtonRight,
            ),
        }
        self._held: str | None = None
        self._pos = (0.0, 0.0)

    def size(self) -> tuple[int, int]:
        bounds = self.q.CGDisplayBounds(self.q.CGMainDisplayID())
        return int(bounds.size.width), int(bounds.size.height)

    def position(self) -> tuple[float, float] | None:
        point = self.q.CGEventGetLocation(self.q.CGEventCreate(None))
        return float(point.x), float(point.y)

    def _post(self, event_type, pos, button, clicks: int = 1) -> None:
        event = self.q.CGEventCreateMouseEvent(None, event_type, pos, button)
        if clicks > 1:
            self.q.CGEventSetIntegerValueField(event, self.q.kCGMouseEventClickState, clicks)
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def move(self, x: float, y: float, dragging: bool = False) -> None:
        self._pos = (x, y)
        if self._held:
            down, up, drag, btn = self._buttons[self._held]
            self._post(drag, self._pos, btn)
        else:
            self._post(self.q.kCGEventMouseMoved, self._pos, self.q.kCGMouseButtonLeft)

    def button_down(self, button: str = "left", clicks: int = 1) -> None:
        down, up, drag, btn = self._buttons[button]
        self._post(down, self._pos, btn, clicks)
        self._held = button

    def button_up(self, button: str = "left", clicks: int = 1) -> None:
        down, up, drag, btn = self._buttons[button]
        self._post(up, self._pos, btn, clicks)
        self._held = None

    def scroll(self, dx_px: float, dy_px: float) -> None:
        event = self.q.CGEventCreateScrollWheelEvent(
            None, self.q.kCGScrollEventUnitPixel, 2, int(dy_px), int(dx_px)
        )
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)


class PyAutoGUIBackend(PointerBackend):
    """Cross-platform fallback. Scrolls in notches, so pixels are accumulated."""

    name = "pyautogui"
    PIXELS_PER_NOTCH = 40.0

    def __init__(self) -> None:
        import pyautogui  # noqa: PLC0415 - optional, slow import

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.gui = pyautogui
        self._acc_x = 0.0
        self._acc_y = 0.0

    def size(self) -> tuple[int, int]:
        w, h = self.gui.size()
        return int(w), int(h)

    def position(self) -> tuple[float, float] | None:
        x, y = self.gui.position()
        return float(x), float(y)

    def move(self, x: float, y: float, dragging: bool = False) -> None:
        self.gui.moveTo(int(x), int(y), _pause=False)

    def button_down(self, button: str = "left", clicks: int = 1) -> None:
        self.gui.mouseDown(button=button, _pause=False)

    def button_up(self, button: str = "left", clicks: int = 1) -> None:
        self.gui.mouseUp(button=button, _pause=False)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self.gui.click(button=button, clicks=clicks, interval=0.0, _pause=False)

    def scroll(self, dx_px: float, dy_px: float) -> None:
        self._acc_y += dy_px / self.PIXELS_PER_NOTCH
        self._acc_x += dx_px / self.PIXELS_PER_NOTCH
        notches_y, self._acc_y = int(self._acc_y), math.modf(self._acc_y)[0]
        notches_x, self._acc_x = int(self._acc_x), math.modf(self._acc_x)[0]
        if notches_y:
            self.gui.scroll(notches_y, _pause=False)
        if notches_x and hasattr(self.gui, "hscroll"):
            self.gui.hscroll(notches_x, _pause=False)


class NullBackend(PointerBackend):
    """Reports a screen size but never emits an event (--dry-run, --selftest)."""

    name = "null"

    def __init__(self, size: tuple[int, int] = (1920, 1080)) -> None:
        self._size = size

    def size(self) -> tuple[int, int]:
        return self._size


def make_backend(dry_run: bool) -> PointerBackend:
    real: PointerBackend | None = None
    errors = []
    if platform.system() == "Darwin":
        try:
            real = QuartzBackend()
        except Exception as exc:  # pragma: no cover - depends on host
            errors.append(f"quartz: {exc}")
    if real is None:
        try:
            real = PyAutoGUIBackend()
        except Exception as exc:  # pragma: no cover - depends on host
            errors.append(f"pyautogui: {exc}")
    if real is None:
        raise SystemExit(
            "No pointer back end available. Install pyautogui (or, on macOS, "
            "pyobjc-framework-Quartz).\n  " + "\n  ".join(errors)
        )
    if dry_run:
        return NullBackend(real.size())
    return real


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


class OneEuroFilter:
    """The 1-euro filter: heavy smoothing when still, light when moving fast.

    Unlike a fixed-weight average it is frame-rate independent, so the pointer
    behaves the same at 15 fps and at 60 fps.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class Schmitt:
    """Latching threshold, so a value hovering on the edge cannot chatter.

    `on > off` engages on a rising value (finger extension); `on < off` engages
    on a falling value (a pinch closing).
    """

    def __init__(self, on: float, off: float, state: bool = False):
        self.on = on
        self.off = off
        self.state = state
        self.rising = on > off

    def update(self, value: float) -> bool:
        if self.rising:
            self.state = value >= self.on if not self.state else value > self.off
        else:
            self.state = value <= self.on if not self.state else value < self.off
        return self.state


# --------------------------------------------------------------------------
# Hand geometry
# --------------------------------------------------------------------------

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """(21, 2) array in pixels. Working in pixels keeps distance ratios honest:
    normalised coordinates are stretched by the frame's aspect ratio."""
    return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)


def dist(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


class HandPose:
    """Per-hand feature extraction, all of it scale- and rotation-invariant.

    Every length is divided by the palm length (wrist to middle knuckle), so
    thresholds hold whether the hand is near the lens or across the room, and
    finger extension is judged by distance from the wrist rather than by screen
    y, so a tilted hand still reads correctly.
    """

    def __init__(self) -> None:
        self.fingers = {
            "index": Schmitt(1.10, 1.00),
            "middle": Schmitt(1.10, 1.00),
            "ring": Schmitt(1.10, 1.00),
            "pinky": Schmitt(1.10, 1.00),
            "thumb": Schmitt(1.10, 1.00),
        }
        self.scale = 1.0
        self.up = dict.fromkeys(self.fingers, False)
        self.pinch_index = 1.0
        self.pinch_middle = 1.0

    def reset(self) -> None:
        for latch in self.fingers.values():
            latch.state = False
        self.up = dict.fromkeys(self.fingers, False)

    def update(self, pts: np.ndarray) -> None:
        self.scale = max(dist(pts, WRIST, MIDDLE_MCP), 1e-3)

        for name, tip, pip in (
            ("index", INDEX_TIP, INDEX_PIP),
            ("middle", MIDDLE_TIP, MIDDLE_PIP),
            ("ring", RING_TIP, RING_PIP),
            ("pinky", PINKY_TIP, PINKY_PIP),
        ):
            ratio = dist(pts, tip, WRIST) / max(dist(pts, pip, WRIST), 1e-3)
            self.up[name] = self.fingers[name].update(ratio)

        # The thumb folds sideways, so measure it against the far side of the palm.
        thumb_ratio = dist(pts, THUMB_TIP, PINKY_MCP) / max(dist(pts, THUMB_IP, PINKY_MCP), 1e-3)
        self.up["thumb"] = self.fingers["thumb"].update(thumb_ratio)

        self.pinch_index = dist(pts, THUMB_TIP, INDEX_TIP) / self.scale
        self.pinch_middle = dist(pts, THUMB_TIP, MIDDLE_TIP) / self.scale

    @property
    def fingers_up(self) -> int:
        return sum(self.up[f] for f in ("index", "middle", "ring", "pinky"))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    camera: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    mirror: bool = True

    model: str = MODEL_FILENAME
    detection_confidence: float = 0.6
    presence_confidence: float = 0.6
    tracking_confidence: float = 0.6

    pad: float = 0.12          # fraction of the frame trimmed off each edge
    min_cutoff: float = 1.5    # 1-euro: lower = smoother at rest
    beta: float = 0.01         # 1-euro: higher = less lag when moving fast

    pinch_on: float = 0.42     # thumb-tip distance / palm length
    pinch_off: float = 0.60
    double_click_gap: float = 0.35
    right_click_cooldown: float = 0.6
    mode_hold_frames: int = 2

    scroll_deadzone: float = 0.0035
    scroll_alpha: float = 0.4
    scroll_gain: float = 2600.0
    scroll_max_step: float = 90.0
    invert_scroll: bool = False

    preview: bool = True
    show_help: bool = True
    dry_run: bool = False


MODE_IDLE, MODE_MOVE, MODE_SCROLL = "IDLE", "MOVE", "SCROLL"
MODE_COLOURS = {
    MODE_IDLE: (150, 150, 150),
    MODE_MOVE: (90, 230, 90),
    MODE_SCROLL: (60, 220, 240),
}


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


@dataclass
class Status:
    """What the HUD needs to know about the last frame."""

    mode: str = MODE_IDLE
    message: str = ""
    pinch: float = 1.0
    dragging: bool = False
    scroll_axis: str = ""
    target: tuple[float, float] | None = None  # pointer position, 0..1 within the active area


class Controller:
    """Turns a stream of hand poses into pointer events."""

    def __init__(self, cfg: Config, backend: PointerBackend):
        self.cfg = cfg
        self.backend = backend
        self.screen_w, self.screen_h = backend.size()

        self.fx = OneEuroFilter(cfg.min_cutoff, cfg.beta)
        self.fy = OneEuroFilter(cfg.min_cutoff, cfg.beta)

        self.pinch_index = Schmitt(cfg.pinch_on, cfg.pinch_off)
        self.pinch_middle = Schmitt(cfg.pinch_on, cfg.pinch_off)

        self.mode = MODE_IDLE
        self._candidate = MODE_IDLE
        self._candidate_frames = 0

        self.dragging = False
        self._press_time = 0.0
        self._last_release = 0.0
        self._last_right_click = 0.0
        self._pending_clicks = 1

        self.scroll_prev: tuple[float, float] | None = None
        self.scroll_vel = np.zeros(2, dtype=np.float64)
        self.scroll_axis = ""

        self.status = Status()

    # -- helpers ---------------------------------------------------------

    def set_smoothing(self, min_cutoff: float) -> None:
        self.cfg.min_cutoff = max(0.2, min(8.0, min_cutoff))
        self.fx.min_cutoff = self.fy.min_cutoff = self.cfg.min_cutoff

    def recentre(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def release_all(self) -> None:
        """Never leave a button stuck down - called on idle, pause and exit."""
        if self.dragging:
            self.backend.button_up("left")
            self.dragging = False
        self.pinch_index.state = False
        self.pinch_middle.state = False

    def idle(self) -> None:
        self.release_all()
        self.scroll_prev = None
        self.scroll_vel[:] = 0.0
        self.scroll_axis = ""
        self.recentre()
        self._settle(MODE_IDLE)
        self.status = Status(mode=self.mode)

    def _settle(self, raw_mode: str) -> str:
        """Require the same raw mode for a few frames before switching, so a
        single misread frame cannot flip the pointer into scrolling."""
        if raw_mode == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = raw_mode, 1
        if self._candidate_frames >= self.cfg.mode_hold_frames:
            self.mode = raw_mode
        return self.mode

    # -- main entry point ------------------------------------------------

    def update(self, pose: HandPose, roi: Roi, pts: np.ndarray, dt: float, now: float) -> Status:
        # While dragging the index finger curls a little; hold the mode so a
        # drag is never cut short by a re-classification.
        if self.dragging:
            raw = MODE_MOVE
        elif pose.up["index"] and pose.up["middle"] and not pose.up["ring"] and not pose.up["pinky"]:
            raw = MODE_SCROLL
        elif pose.up["index"] and not pose.up["middle"] and pose.fingers_up <= 2:
            raw = MODE_MOVE
        else:
            raw = MODE_IDLE

        previous = self.mode
        mode = self._settle(raw)
        if mode != previous:
            self._on_mode_left(previous)

        if mode == MODE_MOVE:
            return self._do_move(pose, roi, pts, dt, now)
        if mode == MODE_SCROLL:
            return self._do_scroll(pose, roi, pts, now)
        self.release_all()
        return Status(mode=MODE_IDLE, pinch=pose.pinch_index)

    def _on_mode_left(self, previous: str) -> None:
        if previous == MODE_SCROLL:
            self.scroll_prev = None
            self.scroll_vel[:] = 0.0
            self.scroll_axis = ""
        if previous == MODE_MOVE:
            self.release_all()

    # -- move / click / drag --------------------------------------------

    def _do_move(self, pose: HandPose, roi: Roi, pts: np.ndarray, dt: float, now: float) -> Status:
        nx, ny = roi.normalise(pts[INDEX_TIP])
        target_x = nx * (self.screen_w - 1)
        target_y = ny * (self.screen_h - 1)

        x = min(max(self.fx(target_x, dt), 0.0), self.screen_w - 1.0)
        y = min(max(self.fy(target_y, dt), 0.0), self.screen_h - 1.0)
        self.backend.move(x, y, dragging=self.dragging)

        message = ""
        was_pinched = self.pinch_index.state
        pinched = self.pinch_index.update(pose.pinch_index)

        if pinched and not was_pinched:
            self._pending_clicks = 2 if now - self._last_release <= self.cfg.double_click_gap else 1
            self.backend.button_down("left", self._pending_clicks)
            self.dragging = True
            self._press_time = now
            message = "double click" if self._pending_clicks == 2 else "click"
        elif not pinched and was_pinched and self.dragging:
            self.backend.button_up("left", self._pending_clicks)
            self.dragging = False
            self._last_release = now
        elif self.dragging and now - self._press_time > 0.25:
            message = "drag"

        return Status(
            mode=MODE_MOVE,
            message=message,
            pinch=pose.pinch_index,
            dragging=self.dragging,
            target=(x / (self.screen_w - 1), y / (self.screen_h - 1)),
        )

    # -- scroll / right click -------------------------------------------

    def _do_scroll(self, pose: HandPose, roi: Roi, pts: np.ndarray, now: float) -> Status:
        was_pinched = self.pinch_middle.state
        pinched = self.pinch_middle.update(pose.pinch_middle)
        if pinched and not was_pinched and now - self._last_right_click > self.cfg.right_click_cooldown:
            self.backend.click("right")
            self._last_right_click = now
            self.scroll_prev = None
            return Status(mode=MODE_SCROLL, message="right click", pinch=pose.pinch_middle)
        if pinched:
            self.scroll_prev = None
            return Status(mode=MODE_SCROLL, pinch=pose.pinch_middle)

        centre = roi.normalise((pts[INDEX_TIP] + pts[MIDDLE_TIP]) / 2.0)
        if self.scroll_prev is None:
            self.scroll_prev = centre
            self.scroll_vel[:] = 0.0
            self.scroll_axis = ""
            return Status(mode=MODE_SCROLL, pinch=pose.pinch_middle)

        delta = np.array(centre) - np.array(self.scroll_prev)
        self.scroll_prev = centre
        delta[np.abs(delta) < self.cfg.scroll_deadzone] = 0.0

        a = self.cfg.scroll_alpha
        self.scroll_vel = (1.0 - a) * self.scroll_vel + a * delta

        # Lock to whichever axis the gesture started on; diagonal drift while
        # reading a page should not shove the view sideways.
        vx, vy = abs(self.scroll_vel[0]), abs(self.scroll_vel[1])
        if not self.scroll_axis and max(vx, vy) > self.cfg.scroll_deadzone:
            self.scroll_axis = "h" if vx > vy else "v"

        step = np.clip(self.scroll_vel * self.cfg.scroll_gain, -self.cfg.scroll_max_step, self.cfg.scroll_max_step)
        sign = -1.0 if self.cfg.invert_scroll else 1.0
        dx = -sign * step[0] if self.scroll_axis == "h" else 0.0
        dy = sign * step[1] if self.scroll_axis == "v" else 0.0
        if abs(dx) >= 1.0 or abs(dy) >= 1.0:
            self.backend.scroll(dx, dy)

        return Status(
            mode=MODE_SCROLL,
            pinch=pose.pinch_middle,
            scroll_axis=self.scroll_axis,
        )


# --------------------------------------------------------------------------
# Active area
# --------------------------------------------------------------------------


class Roi:
    """The slice of the frame that maps onto the screen.

    Padding keeps the pointer reachable without stretching to the very edge of
    the camera's view (where tracking degrades), and the box is letterboxed to
    the screen's aspect ratio so hand motion is not distorted on one axis.
    """

    def __init__(self, frame_w: int, frame_h: int, screen_aspect: float, pad: float):
        self.frame_size = (frame_w, frame_h)
        pad = min(max(pad, 0.0), 0.45)
        x1, y1 = frame_w * pad, frame_h * pad
        x2, y2 = frame_w * (1.0 - pad), frame_h * (1.0 - pad)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

        if w / h > screen_aspect:
            new_w = h * screen_aspect
            cx = (x1 + x2) / 2.0
            x1, x2 = cx - new_w / 2.0, cx + new_w / 2.0
        else:
            new_h = w / screen_aspect
            cy = (y1 + y2) / 2.0
            y1, y2 = cy - new_h / 2.0, cy + new_h / 2.0

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.w = max(x2 - x1, 1.0)
        self.h = max(y2 - y1, 1.0)

    def normalise(self, point) -> tuple[float, float]:
        nx = (float(point[0]) - self.x1) / self.w
        ny = (float(point[1]) - self.y1) / self.h
        return min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)

    @property
    def rect(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2))


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


class CameraStream:
    """Grabs frames on a background thread and keeps only the newest one.

    Reading inline makes the pointer lag behind the hand by however many frames
    the driver has buffered; dropping stale frames removes that lag entirely.
    """

    def __init__(self, cfg: Config):
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if platform.system() == "Darwin" else [cv2.CAP_ANY]
        self.cap = None
        for api in backends:
            cap = cv2.VideoCapture(cfg.camera, api)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise SystemExit(
                f"Could not open camera {cfg.camera}. Try --camera 1, close other apps "
                "using the webcam, and check camera permission for your terminal."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        with contextlib.suppress(cv2.error):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        misses = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                misses += 1
                if misses > 90:
                    break
                time.sleep(0.005)
                continue
            misses = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq: int, timeout: float = 2.0) -> tuple[np.ndarray, int] | None:
        """Block until a frame newer than `last_seq` arrives."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    return self._frame, self._seq
            if not self._thread.is_alive():
                return None
            time.sleep(0.001)
        return None

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
HELP_LINES = (
    "index up ................ move",
    "thumb + index ........... click / drag",
    "index + middle .......... scroll",
    "thumb + middle .......... right click",
    "open palm / fist ........ idle",
    "",
    "q quit   p pause   h help   d dry-run",
    "r recentre   [ ] smoothing",
)


def draw_hud(
    frame: np.ndarray,
    cfg: Config,
    roi: Roi,
    status: Status,
    pose: HandPose | None,
    pts: np.ndarray | None,
    fps: float,
    paused: bool,
    backend_name: str,
) -> None:
    h, w = frame.shape[:2]
    colour = MODE_COLOURS[status.mode]

    if pts is not None and pose is not None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (110, 110, 110), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(pts.astype(int)):
            tip = idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
            cv2.circle(frame, (x, y), 6 if tip else 3, colour if tip else (200, 200, 200), -1, cv2.LINE_AA)
        if status.mode == MODE_MOVE:
            a, b, thickness = THUMB_TIP, INDEX_TIP, 2
        elif status.mode == MODE_SCROLL:
            a, b, thickness = INDEX_TIP, MIDDLE_TIP, 3
        else:
            a = None
        if a is not None:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), colour, thickness, cv2.LINE_AA)

    (x1, y1), (x2, y2) = roi.rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70) if paused else colour, 1, cv2.LINE_AA)

    # Where the smoothed pointer sits, drawn inside the active area. The gap
    # between this and the fingertip is exactly the smoothing lag.
    if status.target is not None:
        px = int(roi.x1 + status.target[0] * roi.w)
        py = int(roi.y1 + status.target[1] * roi.h)
        cv2.circle(frame, (px, py), 9, colour, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 2, colour, -1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w, 34), (24, 24, 24), -1)
    label = "PAUSED" if paused else status.mode
    cv2.putText(frame, label, (12, 24), FONT, 0.7, (60, 60, 220) if paused else colour, 2, cv2.LINE_AA)
    if status.message:
        cv2.putText(frame, status.message, (150, 24), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    elif status.scroll_axis:
        axis = "vertical" if status.scroll_axis == "v" else "horizontal"
        cv2.putText(frame, axis, (150, 24), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    right = f"{fps:4.0f} fps  {backend_name}"
    if cfg.dry_run:
        right = "DRY RUN  " + right
    cv2.putText(frame, right, (w - 12 - 9 * len(right), 24), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Pinch meter: full bar means fingers touching.
    if pose is not None:
        span = max(cfg.pinch_off * 1.8, 1e-3)
        level = 1.0 - min(status.pinch / span, 1.0)
        bar_w, bar_h, bx, by = 160, 10, 12, h - 24
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (70, 70, 70), 1)
        cv2.rectangle(frame, (bx, by), (bx + int(bar_w * level), by + bar_h), colour, -1)
        threshold_x = bx + int(bar_w * (1.0 - cfg.pinch_on / span))
        cv2.line(frame, (threshold_x, by - 3), (threshold_x, by + bar_h + 3), (255, 255, 255), 1)

    if cfg.show_help:
        y = 56
        for line in HELP_LINES:
            if line:
                cv2.putText(frame, line, (12, y), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
            y += 18


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    defaults = Config()
    p = argparse.ArgumentParser(
        description="Control the mouse with hand gestures via a webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=defaults.camera, help="camera index")
    p.add_argument("--width", type=int, default=defaults.width, help="capture width")
    p.add_argument("--height", type=int, default=defaults.height, help="capture height")
    p.add_argument("--fps", type=int, default=defaults.fps, help="requested capture fps")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--model", default=defaults.model, help="path to hand_landmarker.task")
    p.add_argument("--pad", type=float, default=defaults.pad, help="fraction trimmed off each frame edge")
    p.add_argument("--min-cutoff", type=float, default=defaults.min_cutoff, help="1-euro min cutoff (lower = smoother)")
    p.add_argument("--beta", type=float, default=defaults.beta, help="1-euro beta (higher = less lag)")
    p.add_argument("--pinch-on", type=float, default=defaults.pinch_on, help="pinch close threshold (x palm length)")
    p.add_argument("--pinch-off", type=float, default=defaults.pinch_off, help="pinch release threshold")
    p.add_argument("--scroll-gain", type=float, default=defaults.scroll_gain,
                   help="scroll pixels per unit of hand travel")
    p.add_argument("--invert-scroll", action="store_true", help="reverse the scroll direction")
    p.add_argument("--no-preview", action="store_true", help="run without the camera window")
    p.add_argument("--no-help", action="store_true", help="hide the gesture cheat sheet")
    p.add_argument("--dry-run", action="store_true", help="track and display, but never move the pointer")
    p.add_argument("--selftest", action="store_true", help="check camera, model and speed, then exit")
    p.add_argument("--doctor", action="store_true",
                   help="selftest plus a pointer-control check (nudges the cursor and puts it back)")
    args = p.parse_args(argv)

    cfg = Config(
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
        model=args.model,
        pad=args.pad,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        pinch_on=args.pinch_on,
        pinch_off=max(args.pinch_off, args.pinch_on + 0.05),
        scroll_gain=args.scroll_gain,
        invert_scroll=args.invert_scroll,
        preview=not args.no_preview,
        show_help=not args.no_help,
        dry_run=args.dry_run or args.selftest or args.doctor,
    )
    return cfg, args


def build_detector(cfg: Config) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model(cfg.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=cfg.detection_confidence,
        min_hand_presence_confidence=cfg.presence_confidence,
        min_tracking_confidence=cfg.tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


ACCESSIBILITY_HELP = """
  macOS is accepting the events but not delivering them, which means the app
  running this script has no Accessibility permission.

  System Settings -> Privacy & Security -> Accessibility, then switch on the
  app you launched this from (Terminal, iTerm, VS Code, ...), not "python".
  If it is not listed, click + and add it from /Applications or /System/
  Applications/Utilities.

  Quit that app completely (cmd-Q, not just closing the window) and reopen it.
  macOS only re-reads the permission when the process restarts.
"""


def check_pointer_control(backend: PointerBackend) -> bool | None:
    """Nudge the pointer a few points and read it back, then put it back.

    Returns True if the move landed, False if it was swallowed, None if this
    back end cannot report the pointer position.
    """
    origin = backend.position()
    if origin is None:
        return None

    # Retry a few times: a real hand on the trackpad during the test would
    # otherwise look exactly like a dropped event.
    for _ in range(3):
        start = backend.position()
        if start is None:
            return None
        target = (start[0] - 8.0 if start[0] > 20 else start[0] + 8.0, start[1])
        backend.move(*target)
        time.sleep(0.2)
        landed = backend.position()
        if landed is None:
            return None
        if abs(landed[0] - target[0]) < 2.0 and abs(landed[1] - target[1]) < 2.0:
            backend.move(*origin)  # put it back where the user left it
            return True
    return False


def selftest(cfg: Config, nudge_pointer: bool = False) -> int:
    print("telemouse doctor" if nudge_pointer else "telemouse selftest")
    print(f"  python   {platform.python_version()} on {platform.system()} {platform.machine()}")
    detector = build_detector(cfg)
    print(f"  model    {cfg.model} ok")
    backend = make_backend(dry_run=False)
    print(f"  pointer  {backend.name}, screen {backend.size()[0]}x{backend.size()[1]}")

    pointer_ok: bool | None = None
    if nudge_pointer:
        pointer_ok = check_pointer_control(backend)
        if pointer_ok is True:
            print("  control  pointer moved on command - Accessibility is granted")
        elif pointer_ok is False:
            print("  control  pointer did NOT move - events are being dropped")
        else:
            print("  control  cannot read the pointer position on this back end")

    # A camera problem must not hide the pointer diagnosis, so keep going.
    try:
        camera = CameraStream(cfg)
    except SystemExit as exc:
        detector.close()
        print(f"  camera   {exc}")
        if pointer_ok is False:
            print(ACCESSIBILITY_HELP)
        return 1

    seq, frames, hands = 0, 0, 0
    start = time.perf_counter()
    try:
        while frames < 60:
            got = camera.read(seq)
            if got is None:
                break
            frame, seq = got
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int((time.perf_counter() - start) * 1000) + frames)
            hands += bool(result.hand_landmarks)
    finally:
        camera.release()
        detector.close()

    elapsed = time.perf_counter() - start
    if frames == 0:
        print("  camera   no frames received", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    print(f"  camera   {w}x{h}, {frames} frames in {elapsed:.1f}s ({frames / elapsed:.0f} fps)")
    print(f"  tracking hand detected in {hands}/{frames} frames")
    if hands == 0:
        print("\n  The camera works but no hand was seen. Hold one open hand in view\n"
              "  and run this again, or use --dry-run to watch the tracking live.")

    if pointer_ok is False:
        print(ACCESSIBILITY_HELP)
        return 1
    print("ok")
    return 0


def run(cfg: Config) -> int:
    detector = build_detector(cfg)
    backend = make_backend(cfg.dry_run)
    controller = Controller(cfg, backend)
    camera = CameraStream(cfg)

    roi: Roi | None = None
    pose = HandPose()
    seq = 0
    paused = False
    fps = 0.0
    last_time = time.perf_counter()
    start = last_time
    stamp = 0

    print(
        f"telemouse running - pointer back end '{backend.name}', "
        f"screen {controller.screen_w}x{controller.screen_h}"
        + ("  [dry run]" if cfg.dry_run else "")
    )
    print("Press q in the preview window (or ctrl-c here) to quit.")

    if cfg.preview:
        cv2.namedWindow("telemouse", cv2.WINDOW_AUTOSIZE)
        with contextlib.suppress(cv2.error, AttributeError):
            cv2.setWindowProperty("telemouse", cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            got = camera.read(seq)
            if got is None:
                print("Camera stopped delivering frames.", file=sys.stderr)
                return 1
            frame, seq = got
            frame = frame.copy()

            now = time.perf_counter()
            dt = min(max(now - last_time, 1.0 / 240.0), 0.25)
            last_time = now
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            if cfg.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if roi is None or roi.frame_size != (w, h):
                roi = Roi(w, h, controller.screen_w / controller.screen_h, cfg.pad)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            stamp = max(stamp + 1, int((now - start) * 1000))
            result = detector.detect_for_video(image, stamp)

            pts = None
            if result.hand_landmarks and not paused:
                pts = landmarks_to_pixels(result.hand_landmarks[0], w, h)
                pose.update(pts)
                status = controller.update(pose, roi, pts, dt, now)
            elif result.hand_landmarks:
                pts = landmarks_to_pixels(result.hand_landmarks[0], w, h)
                pose.update(pts)
                controller.idle()
                status = controller.status
            else:
                pose.reset()
                controller.idle()
                status = controller.status

            if cfg.preview:
                draw_hud(frame, cfg, roi, status, pose if pts is not None else None, pts, fps, paused, backend.name)
                cv2.imshow("telemouse", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                    controller.idle()
                elif key == ord("h"):
                    cfg.show_help = not cfg.show_help
                elif key == ord("d"):
                    cfg.dry_run = not cfg.dry_run
                    controller.release_all()
                    backend = make_backend(cfg.dry_run)
                    controller.backend = backend
                elif key == ord("r"):
                    controller.recentre()
                elif key == ord("["):
                    controller.set_smoothing(cfg.min_cutoff - 0.25)
                elif key == ord("]"):
                    controller.set_smoothing(cfg.min_cutoff + 0.25)
                if cv2.getWindowProperty("telemouse", cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print()
    finally:
        controller.release_all()
        camera.release()
        detector.close()
        if cfg.preview:
            cv2.destroyAllWindows()
        print("telemouse stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg, args = parse_args(argv)
    if args.selftest or args.doctor:
        return selftest(cfg, nudge_pointer=args.doctor)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
