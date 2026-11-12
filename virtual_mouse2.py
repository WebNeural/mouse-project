import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

CAM_W, CAM_H = 640, 480
MARGIN = 100                
CONTROL_LANDMARK = 5        
PINCH_ON = 0.30            
PINCH_OFF = 0.45             
RIGHT_CLICK_COOLDOWN = 0.5
SCROLL_GAIN = 4000
SCROLL_DEADZONE = 0.008
MIN_ALPHA, MAX_ALPHA = 0.10, 0.60
STABLE_FRAMES = 3

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class Stable:
    def __init__(self, n):
        self.n, self.value, self.count = n, False, 0

    def update(self, raw):
        if raw == self.value:
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.n:
                self.value, self.count = raw, 0
        return self.value


def finger_extended(pts, tip, pip):
    wrist = pts[0]
    return np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist) * 1.1


def pinch_state(prev, dist, size):
    return dist < size * (PINCH_OFF if prev else PINCH_ON)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("خطا: وبکم باز نشد!")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    fist_f, scroll_f = Stable(STABLE_FRAMES), Stable(STABLE_FRAMES)
    smooth = None
    dragging = False
    left_pinched = right_pinched = False
    last_right = 0.0
    prev_scroll_y = None

    print("🖱️ Virtual Mouse v8 | q = خروج")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mode = "NO HAND"
            now = time.time()

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                pts = np.array([[l.x * w, l.y * h] for l in hand.landmark])
                size = np.linalg.norm(pts[0] - pts[9]) 

                idx, mid, ring, pinky = [
                    finger_extended(pts, t, p)
                    for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))
                ]
                is_fist = fist_f.update(not (idx or mid or ring or pinky))
                is_scroll = scroll_f.update(idx and mid and not ring and not pinky) and not is_fist

           
                d_left = np.linalg.norm(pts[4] - pts[8])
                d_right = np.linalg.norm(pts[4] - pts[16])
                allow = not is_fist and not is_scroll
                new_left = allow and pinch_state(left_pinched, d_left, size) and d_left <= d_right
                new_right = allow and not new_left and pinch_state(right_pinched, d_right, size)

                
                if new_left and not left_pinched:
                    pyautogui.click()
                if new_right and not right_pinched and now - last_right > RIGHT_CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_right = now
                left_pinched, right_pinched = new_left, new_right

                
                if is_fist and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                elif not is_fist and dragging:
                    pyautogui.mouseUp()
                    dragging = False

              
                if is_scroll:
                    y = (pts[8][1] + pts[12][1]) / 2 / h
                    if prev_scroll_y is not None:
                        delta = prev_scroll_y - y
                        if abs(delta) > SCROLL_DEADZONE:
                            pyautogui.scroll(int(delta * SCROLL_GAIN))
                    prev_scroll_y = y
                else:
                    prev_scroll_y = None

               
                if not (left_pinched or right_pinched or is_scroll):
                    cx, cy = pts[CONTROL_LANDMARK]
                    target = np.array([
                        np.interp(cx, (MARGIN, w - MARGIN), (0, SCREEN_W - 1)),
                        np.interp(cy, (MARGIN, h - MARGIN), (0, SCREEN_H - 1)),
                    ])
                    if smooth is None:
                        smooth = target
                    else:
                        alpha = np.clip(np.linalg.norm(target - smooth) / 300, MIN_ALPHA, MAX_ALPHA)
                        smooth = smooth + (target - smooth) * alpha
                    pyautogui.moveTo(smooth[0], smooth[1])

               
                mode = ("DRAG" if is_fist else "SCROLL" if is_scroll else
                        "LEFT CLICK" if left_pinched else
                        "RIGHT CLICK" if right_pinched else "MOVE")
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, tuple(pts[CONTROL_LANDMARK].astype(int)), 10, (0, 255, 0), -1)
            else:
               
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                smooth = prev_scroll_y = None
                left_pinched = right_pinched = False
                fist_f.value = scroll_f.value = False

            cv2.rectangle(frame, (MARGIN, MARGIN), (w - MARGIN, h - MARGIN), (255, 255, 0), 1)
            cv2.putText(frame, mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Virtual Mouse v8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if dragging:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()