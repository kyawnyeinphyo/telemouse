#!/usr/bin/env python3
"""Checks for telemouse's gesture logic, driven by synthetic hands.

No camera and no real pointer: hands are built from scratch as 21 landmarks and
pointer events are captured by a recording back end, so every gesture, every
safety release and the frame loop itself can be verified offline.

    python test_gestures.py
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import threading
import time

import numpy as np

import telemouse as T

# --------------------------------------------------------------------------
# Synthetic hands
# --------------------------------------------------------------------------

_MCP = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
_XOFF = {"thumb": -0.45, "index": -0.30, "middle": -0.05, "ring": 0.20, "pinky": 0.42}


class _Landmark:
    def __init__(self, x: float, y: float):
        self.x, self.y = float(x), float(y)


def make_hand(cx=0.5, cy=0.5, scale=0.20, fingers=("index",), pinch=None, rot=0.0):
    """21 normalised landmarks for a hand.

    fingers: which fingers are extended. pinch: None | 'index' | 'middle',
    snapping the thumb tip onto that fingertip. rot: palm rotation in radians.
    """
    pts_by_id = {0: (0.0, 0.0)}
    for name, mcp in _MCP.items():
        pts_by_id[mcp] = (_XOFF[name], 0.35 if name == "thumb" else 0.75)

    for name, mcp in _MCP.items():
        if name == "thumb":
            continue
        extended = name in fingers
        for offset, frac in enumerate((0.35, 0.60, 0.85), start=1):
            if extended:
                pts_by_id[mcp + offset] = (_XOFF[name], 0.75 + frac * 0.75)
            else:  # curled back toward the palm
                pts_by_id[mcp + offset] = (_XOFF[name], 1.05 - frac * 0.45)

    if "thumb" in fingers:
        pts_by_id[2], pts_by_id[3], pts_by_id[4] = (-0.62, 0.55), (-0.78, 0.75), (-0.92, 0.92)
    else:
        pts_by_id[2], pts_by_id[3], pts_by_id[4] = (-0.45, 0.50), (-0.30, 0.62), (-0.16, 0.70)

    pts = np.array([pts_by_id[i] for i in range(21)], dtype=np.float64)
    if pinch == "index":
        pts[T.THUMB_TIP] = pts[T.INDEX_TIP] + 0.02
    elif pinch == "middle":
        pts[T.THUMB_TIP] = pts[T.MIDDLE_TIP] + 0.02

    c, s = math.cos(rot), math.sin(rot)
    pts = (pts @ np.array([[c, -s], [s, c]])) * scale
    pts[:, 1] *= -1.0  # screen y grows downward
    pts += np.array([cx, cy])
    return [_Landmark(*p) for p in pts]


class Recorder(T.PointerBackend):
    """Captures pointer events instead of emitting them."""

    name = "recorder"

    def __init__(self):
        self.events: list[tuple] = []

    def size(self):
        return (1728, 1117)

    def move(self, x, y, dragging=False):
        self.events.append(("move", round(x), round(y), dragging))

    def button_down(self, button="left", clicks=1):
        self.events.append(("down", button, clicks))

    def button_up(self, button="left", clicks=1):
        self.events.append(("up", button, clicks))

    def scroll(self, dx, dy):
        self.events.append(("scroll", round(dx), round(dy)))


FRAME_W, FRAME_H = 1280, 720


def play(frames, cfg=None, dt=1 / 30):
    """Feed a list of make_hand() kwargs (or None for 'no hand') to a Controller."""
    cfg = cfg or T.Config()
    rec = Recorder()
    controller = T.Controller(cfg, rec)
    roi = T.Roi(FRAME_W, FRAME_H, rec.size()[0] / rec.size()[1], cfg.pad)
    pose = T.HandPose()
    statuses, now = [], 100.0
    for kwargs in frames:
        now += dt
        if kwargs is None:
            pose.reset()
            controller.idle()
            statuses.append(controller.status)
            continue
        pts = T.landmarks_to_pixels(make_hand(**kwargs), FRAME_W, FRAME_H)
        pose.update(pts)
        statuses.append(controller.update(pose, roi, pts, dt, now))
    return rec, statuses


def settled_pose(**kwargs) -> T.HandPose:
    pose = T.HandPose()
    for _ in range(3):
        pose.update(T.landmarks_to_pixels(make_hand(**kwargs), FRAME_W, FRAME_H))
    return pose


# --------------------------------------------------------------------------
# Test runner
# --------------------------------------------------------------------------

_results: list[bool] = []


def section(title: str) -> None:
    print(f"\n{title}")


def check(label: str, condition: bool, detail=None) -> bool:
    condition = bool(condition)
    _results.append(condition)
    suffix = "" if condition or detail is None else f"   -> {detail}"
    print(f"  {'pass' if condition else 'FAIL'}  {label}{suffix}")
    return condition


# --------------------------------------------------------------------------
# Pose features
# --------------------------------------------------------------------------


def test_finger_classification():
    section("finger classification")
    cases = [
        (("index",), {"index": True, "middle": False, "ring": False, "pinky": False}),
        (("index", "middle"), {"index": True, "middle": True, "ring": False, "pinky": False}),
        (("index", "middle", "ring", "pinky"), dict.fromkeys(("index", "middle", "ring", "pinky"), True)),
        ((), dict.fromkeys(("index", "middle", "ring", "pinky"), False)),
    ]
    for fingers, expected in cases:
        pose = settled_pose(fingers=fingers)
        got = {k: pose.up[k] for k in expected}
        check(f"{fingers or 'fist'}", got == expected, got)


def test_scale_and_rotation_invariance():
    section("scale and rotation invariance")
    for scale in (0.10, 0.20, 0.35):
        for rot in (0.0, 0.4, -0.5):
            pose = settled_pose(fingers=("index",), scale=scale, rot=rot)
            got = (pose.up["index"], pose.up["middle"])
            check(f"scale={scale} rot={rot:+.1f} reads as index only", got == (True, False), got)


def test_pinch_is_scale_invariant():
    section("pinch is scale invariant")
    cfg = T.Config()
    for scale in (0.10, 0.20, 0.35):
        closed = settled_pose(fingers=("index",), scale=scale, pinch="index").pinch_index
        opened = settled_pose(fingers=("index",), scale=scale).pinch_index
        check(f"scale={scale} closed below pinch_on", closed < cfg.pinch_on, round(closed, 3))
        check(f"scale={scale} open above pinch_off", opened > cfg.pinch_off, round(opened, 3))


# --------------------------------------------------------------------------
# Gestures
# --------------------------------------------------------------------------


def test_move():
    section("move")
    rec, statuses = play([dict(fingers=("index",), cx=0.40 + i * 0.01) for i in range(12)])
    moves = [e for e in rec.events if e[0] == "move"]
    check("emits pointer moves", len(moves) >= 8, len(moves))
    check("mode settles on MOVE", statuses[-1].mode == T.MODE_MOVE, statuses[-1].mode)
    check("pointer follows the hand", moves[-1][1] > moves[0][1], (moves[0], moves[-1]))
    check("no button events", not [e for e in rec.events if e[0] in ("down", "up")])


def test_click():
    section("click")
    frames = [dict(fingers=("index",))] * 4 + [dict(fingers=("index",), pinch="index")] * 2 + [dict(fingers=("index",))] * 4
    rec, _ = play(frames)
    buttons = [e for e in rec.events if e[0] in ("down", "up")]
    check("one press then one release", [e[:2] for e in buttons] == [("down", "left"), ("up", "left")], buttons)
    check("counted as a single click", buttons and buttons[0][2] == 1, buttons)


def test_drag():
    section("drag")
    frames = (
        [dict(fingers=("index",))] * 3
        + [dict(fingers=("index",), pinch="index", cx=0.40 + i * 0.01) for i in range(20)]
        + [dict(fingers=("index",))] * 3
    )
    rec, statuses = play(frames)
    buttons = [e for e in rec.events if e[0] in ("down", "up")]
    check("one press, one release", [e[:2] for e in buttons] == [("down", "left"), ("up", "left")], buttons)
    check("moves are flagged as dragging", len([e for e in rec.events if e[0] == "move" and e[3]]) >= 15)
    check("drag reported to the HUD", any(s.message == "drag" for s in statuses))


def test_drag_survives_a_misread_frame():
    section("drag survives a misread frame")
    frames = (
        [dict(fingers=("index",))] * 3
        + [dict(fingers=("index",), pinch="index")] * 3
        + [dict(fingers=(), pinch="index")] * 2  # index momentarily reads as curled
        + [dict(fingers=("index",), pinch="index")] * 3
        + [dict(fingers=("index",))] * 3
    )
    rec, _ = play(frames)
    buttons = [e[0] for e in rec.events if e[0] in ("down", "up")]
    check("exactly one press/release pair", buttons == ["down", "up"], buttons)


def test_double_click():
    section("double click")
    frames = (
        [dict(fingers=("index",))] * 3
        + [dict(fingers=("index",), pinch="index")] * 2
        + [dict(fingers=("index",))] * 2
        + [dict(fingers=("index",), pinch="index")] * 2
        + [dict(fingers=("index",))] * 2
    )
    rec, _ = play(frames)
    downs = [e for e in rec.events if e[0] == "down"]
    check("two presses", len(downs) == 2, downs)
    check("second carries click count 2", len(downs) == 2 and downs[1][2] == 2, downs)


def test_scroll():
    section("scroll")
    cfg = T.Config()
    rec, statuses = play([dict(fingers=("index", "middle"), cy=0.5 + i * 0.012) for i in range(25)])
    scrolls = [e for e in rec.events if e[0] == "scroll"]
    check("emits scroll events", len(scrolls) >= 10, len(scrolls))
    check("axis locked to vertical", statuses[-1].scroll_axis == "v", statuses[-1].scroll_axis)
    check("no horizontal leak", all(e[1] == 0 for e in scrolls), scrolls[:3])
    check("content follows the hand", scrolls[-1][2] > 0, scrolls[-1])
    check("pointer never moves while scrolling", not [e for e in rec.events if e[0] == "move"])

    rec, _ = play([dict(fingers=("index", "middle"), cy=0.5 + i * 0.20) for i in range(6)])
    steps = [abs(e[2]) for e in rec.events if e[0] == "scroll"]
    check("steps are clamped", all(s <= cfg.scroll_max_step for s in steps), steps)

    rec, _ = play([dict(fingers=("index", "middle"), cy=0.5 + i * 0.0002) for i in range(20)])
    check("jitter below the deadzone is ignored", not [e for e in rec.events if e[0] == "scroll"], rec.events[:3])


def test_right_click():
    section("right click")
    frames = (
        [dict(fingers=("index", "middle"))] * 4
        + [dict(fingers=("index", "middle"), pinch="middle")] * 3
        + [dict(fingers=("index", "middle"))] * 3
    )
    rec, _ = play(frames)
    rights = [e for e in rec.events if e[0] in ("down", "up") and e[1] == "right"]
    check("one right press and release", [e[:2] for e in rights] == [("down", "right"), ("up", "right")], rights)
    check("not repeated while held", len([e for e in rights if e[0] == "down"]) == 1, rights)


def test_idle_and_safety():
    section("idle and safety")
    frames = [dict(fingers=("index",))] * 3 + [dict(fingers=("index",), pinch="index")] * 3 + [None] * 3
    rec, _ = play(frames)
    buttons = [e[0] for e in rec.events if e[0] in ("down", "up")]
    check("button released when the hand leaves", buttons == ["down", "up"], buttons)

    rec, statuses = play([dict(fingers=("index", "middle", "ring", "pinky"))] * 5)
    check("open palm is idle", statuses[-1].mode == T.MODE_IDLE, statuses[-1].mode)
    check("open palm emits nothing", rec.events == [], rec.events[:3])

    rec, statuses = play([dict(fingers=())] * 5)
    check("fist is idle and silent", statuses[-1].mode == T.MODE_IDLE and rec.events == [], statuses[-1].mode)


def test_mode_debounce():
    section("mode debounce")
    frames = [dict(fingers=("index",))] * 6 + [dict(fingers=("index", "middle"))] + [dict(fingers=("index",))] * 6
    _, statuses = play(frames)
    modes = [s.mode for s in statuses[2:]]
    check("one stray frame does not switch mode", all(m == T.MODE_MOVE for m in modes), modes)


# --------------------------------------------------------------------------
# Filters and mapping
# --------------------------------------------------------------------------


def test_smoothing_is_frame_rate_independent():
    section("smoothing")

    def destination(dt):
        rec, _ = play([dict(fingers=("index",), cx=0.30)] + [dict(fingers=("index",), cx=0.70)] * 20, dt=dt)
        return [e for e in rec.events if e[0] == "move"][-1][1]

    slow, fast = destination(1 / 15), destination(1 / 60)
    check("same destination at 15 and 60 fps", abs(slow - fast) <= 3, (slow, fast))

    f = T.OneEuroFilter()
    jittered = [f(100.0 + (5 if i % 2 else -5), 1 / 30) for i in range(60)]
    check("suppresses jitter at rest", float(np.std(jittered[20:])) < 2.0, round(float(np.std(jittered[20:])), 2))

    f = T.OneEuroFilter()
    swept = [f(v, 1 / 30) for v in np.linspace(0, 1000, 60)]
    check("keeps up with fast motion", swept[-1] > 900, round(swept[-1], 1))


def test_schmitt():
    section("latching thresholds")
    rising = T.Schmitt(1.10, 1.00)
    check("rising: engages above on", rising.update(1.2))
    check("rising: holds between off and on", rising.update(1.05))
    check("rising: releases below off", not rising.update(0.95))
    falling = T.Schmitt(0.42, 0.60)
    check("falling: engages below on", falling.update(0.30))
    check("falling: holds between on and off", falling.update(0.50))
    check("falling: releases above off", not falling.update(0.70))


def test_roi():
    section("active area")
    roi = T.Roi(FRAME_W, FRAME_H, 1728 / 1117, 0.12)
    check("aspect matches the screen", abs(roi.w / roi.h - 1728 / 1117) < 0.01, roi.w / roi.h)
    corners_ok = roi.normalise((roi.x1, roi.y1)) == (0.0, 0.0) and roi.normalise((roi.x2, roi.y2)) == (1.0, 1.0)
    check("corners map to 0 and 1", corners_ok)
    check("outside the area clamps", roi.normalise((-500, 5000)) == (0.0, 1.0))
    check("is inset from the frame edge", roi.x1 > 0 and roi.y1 > 0)


def test_backends():
    section("pointer back ends")
    null = T.NullBackend((100, 100))
    null.move(5, 5)
    null.click("left", 2)
    null.scroll(3, 3)
    check("null back end emits nothing and does not raise", null.size() == (100, 100))

    if sys.platform == "darwin":
        try:
            import Quartz

            backend = T.QuartzBackend()
            w, h = backend.size()
            check(f"quartz reports a screen ({w}x{h})", w > 0 and h > 0)
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, (10.0, 10.0), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, 2)
            wheel = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, 3, 0)
            check("mouse and scroll events build (never posted here)", event is not None and wheel is not None)
        except ImportError:
            print("  skip  quartz not installed")


# --------------------------------------------------------------------------
# Frame loop
# --------------------------------------------------------------------------


def test_frame_loop():
    section("frame loop (real detector, synthetic frames)")
    total = 45
    cameras = []

    class FakeCamera:
        def __init__(self, cfg):
            rng = np.random.default_rng(0)
            self.frames = [rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(4)]
            self.n = 0
            self.released = False

        def read(self, last_seq, timeout=2.0):
            if self.n >= total:
                return None
            self.n += 1
            return self.frames[self.n % 4], self.n

        def release(self):
            self.released = True

    original = T.CameraStream
    T.CameraStream = lambda cfg: (cameras.append(FakeCamera(cfg)), cameras[-1])[1]
    try:
        cfg, _ = T.parse_args(["--no-preview", "--dry-run"])
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = T.run(cfg)
        elapsed = time.perf_counter() - started
    finally:
        T.CameraStream = original

    check("returns 1 when the camera stops", rc == 1, rc)
    check("camera released", cameras[0].released)
    check(f"processed every frame ({cameras[0].n / elapsed:.0f} fps)", cameras[0].n == total, cameras[0].n)
    check("no camera thread left behind", not [t for t in threading.enumerate() if t.name == "camera"])


def test_hud_renders():
    section("hud")
    cfg = T.Config()
    roi = T.Roi(FRAME_W, FRAME_H, 1512 / 982, cfg.pad)
    for label, kwargs in (
        ("move", dict(fingers=("index",), pinch="index")),
        ("scroll", dict(fingers=("index", "middle"))),
        ("idle", dict(fingers=("index", "middle", "ring", "pinky"))),
    ):
        frame = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
        rec = Recorder()
        controller = T.Controller(cfg, rec)
        pose = T.HandPose()
        status = None
        for i in range(6):
            pts = T.landmarks_to_pixels(make_hand(cy=0.45 + i * 0.01, scale=0.22, **kwargs), FRAME_W, FRAME_H)
            pose.update(pts)
            status = controller.update(pose, roi, pts, 1 / 30, 100.0 + i / 30)
        before = frame.copy()
        T.draw_hud(frame, cfg, roi, status, pose, pts, 58.0, False, "quartz")
        check(f"{label} hud draws without raising and changes pixels", not np.array_equal(frame, before))


# --------------------------------------------------------------------------


def main() -> int:
    for test in (
        test_finger_classification,
        test_scale_and_rotation_invariance,
        test_pinch_is_scale_invariant,
        test_move,
        test_click,
        test_drag,
        test_drag_survives_a_misread_frame,
        test_double_click,
        test_scroll,
        test_right_click,
        test_idle_and_safety,
        test_mode_debounce,
        test_smoothing_is_frame_rate_independent,
        test_schmitt,
        test_roi,
        test_backends,
        test_frame_loop,
        test_hud_renders,
    ):
        test()

    failed = _results.count(False)
    print(f"\n{len(_results) - failed}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
