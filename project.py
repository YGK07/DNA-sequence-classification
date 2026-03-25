"""
Air Character Recognition - DNA String Matcher
================================================
Uses webcam + MediaPipe to track index fingertip.
Write characters in the air to enter DNA sequence and pattern,
then runs naive string matching in C.

Requirements:
    pip install opencv-python mediapipe numpy

Also compile the C program first:
    gcc naive_match.c -o naive_match

Controls:
    - Raise index finger to draw in the air
    - Fold index finger (or keep all fingers down) to PAUSE drawing
    - Press SPACE  → confirm current character (adds to input)
    - Press ENTER  → submit the full input (DNA first, then pattern)
    - Press BACKSPACE → delete last character
    - Press C      → clear current stroke
    - Press Q      → quit
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import sys
from collections import deque

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CANVAS_SIZE   = (480, 640)          # height, width  (matches typical webcam)
STROKE_COLOR  = (0, 255, 0)         # green strokes
STROKE_WIDTH  = 3
TRAIL_LENGTH  = 64                  # how many recent points to keep for drawing
C_EXECUTABLE = "naive_match.exe"     # path to compiled C binary

# ──────────────────────────────────────────────
# Simple stroke → character mapping using
# bounding-box aspect ratio + stroke count heuristic.
# For a class demo, we use on-screen keyboard overlay
# so the user selects a letter by hovering.
# ──────────────────────────────────────────────

# On-screen keyboard rows
KEYBOARD = [
    list("ACGT"),          # DNA bases (top row, quick access)
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]

KEY_W, KEY_H = 55, 45           # key dimensions in pixels
KEY_START_X,  KEY_START_Y = 10, 220  # top-left of keyboard area
HOVER_FRAMES  = 20              # frames to hover over key to select it


def build_key_rects():
    """Return list of (char, x1, y1, x2, y2) for every key."""
    rects = []
    for row_i, row in enumerate(KEYBOARD):
        y1 = KEY_START_Y + row_i * (KEY_H + 5)
        y2 = y1 + KEY_H
        for col_i, ch in enumerate(row):
            x1 = KEY_START_X + col_i * (KEY_W + 5)
            x2 = x1 + KEY_W
            rects.append((ch, x1, y1, x2, y2))
    return rects


KEY_RECTS = build_key_rects()


def draw_keyboard(frame, hovered_char, typed_text, stage):
    """Overlay the on-screen keyboard and current typed text onto frame."""
    overlay = frame.copy()

    for (ch, x1, y1, x2, y2) in KEY_RECTS:
        color = (0, 200, 255) if ch == hovered_char else (50, 50, 50)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(overlay, ch, (x1 + 12, y2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Backspace key
    bx1, by1, bx2, by2 = KEY_START_X + 8 * (KEY_W + 5), KEY_START_Y + 3 * (KEY_H + 5), \
                          KEY_START_X + 8 * (KEY_W + 5) + KEY_W + 30, KEY_START_Y + 3 * (KEY_H + 5) + KEY_H
    bcolor = (0, 200, 255) if hovered_char == "⌫" else (80, 0, 0)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bcolor, -1)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    cv2.putText(overlay, "DEL", (bx1 + 5, by2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    KEY_RECTS_EXTRA = [("⌫", bx1, by1, bx2, by2)]

    # ENTER key
    ex1 = KEY_START_X
    ey1 = KEY_START_Y + 4 * (KEY_H + 5)
    ex2 = ex1 + 120
    ey2 = ey1 + KEY_H
    ecolor = (0, 200, 255) if hovered_char == "↵" else (0, 100, 0)
    cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), ecolor, -1)
    cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (200, 200, 200), 1)
    cv2.putText(overlay, "ENTER", (ex1 + 15, ey2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Space key
    sx1 = ex2 + 10
    sy1, sy2 = ey1, ey2
    sx2 = sx1 + 200
    scolor = (0, 200, 255) if hovered_char == " " else (60, 60, 60)
    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), scolor, -1)
    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (200, 200, 200), 1)
    cv2.putText(overlay, "SPACE", (sx1 + 50, sy2 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # blend overlay
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # typed text display
    label = "DNA Sequence" if stage == 0 else "Gene Pattern"
    cv2.rectangle(frame, (0, 0), (640, 60), (20, 20, 20), -1)
    cv2.putText(frame, f"{label}: {typed_text}_",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 180), 2)

    return KEY_RECTS_EXTRA


def get_hovered_key(finger_x, finger_y, extra_rects):
    """Return the character whose key the fingertip is hovering over."""
    all_rects = KEY_RECTS + extra_rects

    # Add ENTER and SPACE manually (they were drawn in draw_keyboard)
    # We reconstruct them here for hit-testing
    ex1 = KEY_START_X
    ey1 = KEY_START_Y + 4 * (KEY_H + 5)
    ex2 = ex1 + 120
    ey2 = ey1 + KEY_H
    all_rects.append(("↵", ex1, ey1, ex2, ey2))

    sx1 = ex2 + 10
    sy1, sy2 = ey1, ey2
    sx2 = sx1 + 200
    all_rects.append((" ", sx1, sy1, sx2, sy2))

    for (ch, x1, y1, x2, y2) in all_rects:
        if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
            return ch
    return None


def is_index_up(hand_landmarks):
    """Return True if index finger is raised (tip above PIP joint)."""
    tip   = hand_landmarks.landmark[8]   # INDEX_FINGER_TIP
    pip   = hand_landmarks.landmark[6]   # INDEX_FINGER_PIP
    return tip.y < pip.y                 # y increases downward


def run_c_program(dna, pattern):
    """Compile (if needed) and run the C naive match program."""
    if not os.path.exists(C_EXECUTABLE):
        print("[INFO] Compiling naive_match.c ...")
        ret = os.system("gcc naive_match.c -o naive_match")
        if ret != 0:
            print("[ERROR] Compilation failed. Make sure naive_match.c is in the same folder.")
            return
    result = subprocess.run(
        [C_EXECUTABLE, dna, pattern],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)


def main():
    mp_hands   = mp.solutions.hands
    mp_draw    = mp.solutions.drawing_utils
    hands_sol  = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cv2.namedWindow("Air Character Recognition - DNA Matcher", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Character Recognition - DNA Matcher", 1280, 720)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    trail        = deque(maxlen=TRAIL_LENGTH)
    typed_texts  = ["", ""]   # [dna, pattern]
    stage        = 0          # 0 = entering DNA, 1 = entering pattern
    hover_char   = None
    hover_count  = 0
    last_selected = None

    print("=" * 50)
    print(" Air Character Recognition - DNA String Matcher")
    print("=" * 50)
    print("Hover your index fingertip over a key to select it.")
    print("Hold for ~20 frames to register the character.")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_sol.process(rgb)

        finger_x, finger_y = -1, -1
        index_up = False

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                tip = hand_lm.landmark[8]
                h, w, _ = frame.shape
                finger_x = int(tip.x * w)
                finger_y = int(tip.y * h)
                index_up = is_index_up(hand_lm)

                # draw fingertip circle
                cv2.circle(frame, (finger_x, finger_y), 10,
                           (0, 255, 0) if index_up else (0, 0, 255), -1)

        # Draw keyboard overlay
        extra_rects = draw_keyboard(frame, hover_char,
                                    typed_texts[stage], stage)

        # Hover detection (only when index is up)
        if index_up and finger_x > 0:
            ch = get_hovered_key(finger_x, finger_y, extra_rects)
            if ch == hover_char:
                hover_count += 1
            else:
                hover_char  = ch
                hover_count = 0

            # Draw hover progress bar on the hovered key
            if hover_char and hover_count > 0:
                progress = min(hover_count / HOVER_FRAMES, 1.0)
                for (c, x1, y1, x2, y2) in KEY_RECTS + extra_rects:
                    if c == hover_char:
                        bar_w = int((x2 - x1) * progress)
                        cv2.rectangle(frame, (x1, y2 - 5),
                                      (x1 + bar_w, y2), (0, 255, 255), -1)
                        break

            if hover_count >= HOVER_FRAMES and hover_char != last_selected:
                # Character selected!
                selected = hover_char
                last_selected = selected
                hover_count = 0

                if selected == "⌫":
                    typed_texts[stage] = typed_texts[stage][:-1]
                elif selected == "↵":
                    if stage == 0 and typed_texts[0]:
                        print(f"[DNA]     {typed_texts[0]}")
                        stage = 1
                    elif stage == 1 and typed_texts[1]:
                        print(f"[Pattern] {typed_texts[1]}")
                        cap.release()
                        cv2.destroyAllWindows()
                        run_c_program(typed_texts[0], typed_texts[1])
                        return
                else:
                    typed_texts[stage] += selected
        else:
            # Reset hover when finger is down
            if not index_up:
                hover_char    = None
                hover_count   = 0
                last_selected = None

        # Stage indicator
        stage_msg = "Step 1/2: Enter DNA Sequence  →  raise finger & hover keys" \
                    if stage == 0 else \
                    "Step 2/2: Enter Gene Pattern  →  hover keys, then ENTER"
        cv2.putText(frame, stage_msg, (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("Air Character Recognition - DNA Matcher", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
