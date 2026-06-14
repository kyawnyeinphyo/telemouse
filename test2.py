import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time

# --- Configuration ---
model_path = "hand_landmarker.task"
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Disable pyautogui failsafe for smoother control (optional)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Pointer smoothing
p_loc_x, p_loc_y = 0, 0 
smoothening = 5 

# Scroll logic
scroll_prev_y = None
scroll_velocity = 0.0
scroll_deadzone = 0.0025  # normalized dy threshold to ignore jitter
scroll_alpha = 0.35       # smoothing factor for scroll velocity
scroll_speed = 1500       # scale normalized dy to scroll units
scroll_max_step = 120     # clamp each scroll step to avoid spikes

# Active area padding (reduce usable camera area to avoid edge loss)
active_pad_x = 0.08  # 8% left/right padding
active_pad_y = 0.08  # 8% top/bottom padding
draw_active_area = True

# Click debounce
last_click_time = 0
click_cooldown = 0.3  # seconds
pinch_down_thresh = 0.035  # normalized distance
pinch_up_thresh = 0.05     # normalized distance (hysteresis)
pinch_active = False
drag_active = False

# --- Model Setup ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

def is_finger_extended(landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is extended (tip is above pip joint in y-axis)"""
    # For webcam flipped view, extended finger has tip.y < pip.y
    return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y

def normalize_to_active(x, y, roi_x1, roi_y1, roi_w, roi_h, frame_w, frame_h):
    """Map full-frame normalized coords into active-area normalized coords (clamped)."""
    px = x * frame_w
    py = y * frame_h
    nx = (px - roi_x1) / roi_w
    ny = (py - roi_y1) / roi_h
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return nx, ny

print("Program started! Show index + middle fingers to scroll. Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success: 
        break
    
    frame = cv2.flip(frame, 1)  # Mirror for natural control
    h, w, _ = frame.shape

    # Compute active area (crop) to avoid edges
    pad_x = min(max(active_pad_x, 0.0), 0.45)
    pad_y = min(max(active_pad_y, 0.0), 0.45)
    roi_x1 = int(w * pad_x)
    roi_y1 = int(h * pad_y)
    roi_x2 = int(w * (1.0 - pad_x))
    roi_y2 = int(h * (1.0 - pad_y))
    roi_w = max(1, roi_x2 - roi_x1)
    roi_h = max(1, roi_y2 - roi_y1)

    # Match active area aspect ratio to display ratio
    screen_aspect = screen_w / screen_h
    roi_aspect = roi_w / roi_h
    if roi_aspect > screen_aspect:
        new_w = max(1, int(roi_h * screen_aspect))
        cx = (roi_x1 + roi_x2) // 2
        roi_x1 = max(0, cx - new_w // 2)
        roi_x2 = min(w, roi_x1 + new_w)
    elif roi_aspect < screen_aspect:
        new_h = max(1, int(roi_w / screen_aspect))
        cy = (roi_y1 + roi_y2) // 2
        roi_y1 = max(0, cy - new_h // 2)
        roi_y2 = min(h, roi_y1 + new_h)
    roi_w = max(1, roi_x2 - roi_x1)
    roi_h = max(1, roi_y2 - roi_y1)

    if draw_active_area:
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Finger landmarks
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            
            # Pixel coordinates
            ix, iy = index_tip.x * w, index_tip.y * h
            mx, my = middle_tip.x * w, middle_tip.y * h
            tx, ty = thumb_tip.x * w, thumb_tip.y * h

            # Check which fingers are extended
            index_up = is_finger_extended(hand_landmarks, 8, 6)
            middle_up = is_finger_extended(hand_landmarks, 12, 10)
            ring_up = is_finger_extended(hand_landmarks, 16, 14)
            pinky_up = is_finger_extended(hand_landmarks, 20, 18)
            
            # Scroll Mode: Index + Middle extended, Ring + Pinky down
            scroll_mode = index_up and middle_up and not ring_up and not pinky_up
            
            if scroll_mode:
                # Use midpoint between index and middle for scroll tracking (normalized)
                _, idx_ny = normalize_to_active(index_tip.x, index_tip.y, roi_x1, roi_y1, roi_w, roi_h, w, h)
                _, mid_ny = normalize_to_active(middle_tip.x, middle_tip.y, roi_x1, roi_y1, roi_w, roi_h, w, h)
                scroll_y = (idx_ny + mid_ny) / 2.0
                
                cv2.putText(frame, "SCROLL MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.line(frame, (int(ix), int(iy)), (int(mx), int(my)), (0, 255, 255), 3)
                
                if scroll_prev_y is None:
                    scroll_prev_y = scroll_y
                dy = scroll_prev_y - scroll_y
                scroll_prev_y = scroll_y

                # Ignore tiny movements
                if abs(dy) < scroll_deadzone:
                    dy = 0.0

                # Smooth the scroll velocity
                scroll_velocity = (1 - scroll_alpha) * scroll_velocity + scroll_alpha * dy
                scroll_amount = int(scroll_velocity * scroll_speed)
                if scroll_amount != 0:
                    scroll_amount = max(-scroll_max_step, min(scroll_max_step, scroll_amount))
                    pyautogui.scroll(scroll_amount)

                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False
                pinch_active = False
                continue  # Skip mouse movement while scrolling
            else:
                scroll_prev_y = None
                scroll_velocity = 0.0  # Reset scroll state

            # Mouse Moving: Only index finger extended
            if index_up and not middle_up:
                cv2.putText(frame, "MOVE MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mapped_x, mapped_y = normalize_to_active(index_tip.x, index_tip.y, roi_x1, roi_y1, roi_w, roi_h, w, h)
                target_x = screen_w * mapped_x
                target_y = screen_h * mapped_y
                target_x = max(0, min(screen_w - 1, target_x))
                target_y = max(0, min(screen_h - 1, target_y))
                
                curr_x = p_loc_x + (target_x - p_loc_x) / smoothening
                curr_y = p_loc_y + (target_y - p_loc_y) / smoothening
                
                pyautogui.moveTo(curr_x, curr_y, _pause=False)
                p_loc_x, p_loc_y = curr_x, curr_y
                
                # Click: Thumb touches index finger
                click_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                current_time = time.time()
                if not pinch_active and click_dist < pinch_down_thresh:
                    if current_time - last_click_time > click_cooldown:
                        cv2.circle(frame, (int(ix), int(iy)), 20, (0, 255, 0), cv2.FILLED)
                        pyautogui.mouseDown()
                        drag_active = True
                        last_click_time = current_time
                    pinch_active = True
                elif pinch_active and click_dist > pinch_up_thresh:
                    pinch_active = False
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
            else:
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False
                pinch_active = False

            # Draw hand landmarks
            for landmark in hand_landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, (255, 0, 255), -1)
    else:
        scroll_prev_y = None
        scroll_velocity = 0.0
        pinch_active = False
        if drag_active:
            pyautogui.mouseUp()
            drag_active = False

    cv2.imshow("Virtual Mouse + Scroll", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
