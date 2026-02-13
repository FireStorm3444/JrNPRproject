import re
import os
import sqlite3
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
VIDEO_PATH = "test_videos/test_9.1.mp4"
OUTPUT_PATH = "output_videos/output_9.1_easy.mp4"
DB_PATH = "vehicles.db"
FRAME_SKIP = 1
FRAME_RESIZE_WIDTH = None
SHOW_WINDOW = True
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
TEXT_SCALE = 1.5
TEXT_THICKNESS = 3
BUFFER_SIZE = 10

def clean_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Heuristic fixes for common OCR confusions.
    corrections = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    text_list = list(text)
    # Fix the last 4 characters (usually numbers in many formats)
    if len(text_list) >= 4:
        for i in range(len(text_list) - 4, len(text_list)):
            char = text_list[i]
            if char in corrections:
                text_list[i] = corrections[char]

    return "".join(text_list)

def load_known_plates(db_path: str) -> set:
    if not os.path.exists(db_path):
        return set()

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT text FROM plates")
        rows = cur.fetchall()
    except sqlite3.Error:
        return set()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    cleaned = {clean_plate_text(row[0]) for row in rows if row and row[0]}
    return {plate[-4:] for plate in cleaned if len(plate) >= 4}

def get_plate_label_and_color(text: str, known_plates: set):
    tail = text[-4:] if len(text) >= 4 else ""
    if tail and tail in known_plates:
        return "KNOWN", (0, 255, 0)
    return "UNKNOWN", (0, 0, 255)

def resize_for_detection(frame):
    if not FRAME_RESIZE_WIDTH:
        return frame, 1.0, 1.0

    h, w = frame.shape[:2]
    if w <= FRAME_RESIZE_WIDTH:
        return frame, 1.0, 1.0

    scale = FRAME_RESIZE_WIDTH / w
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame, (FRAME_RESIZE_WIDTH, new_h))
    return resized, w / FRAME_RESIZE_WIDTH, h / new_h

# --- IMPROVED PREPROCESSING (Ported from test_paddle.py) ---
def preprocess_plate(plate_crop):
    if plate_crop.size == 0:
        return plate_crop

    # 1. Grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Smart Upscale (Force height to 64px)
    h, w = enhanced.shape
    target_height = 64
    scale = target_height / h

    # INTER_CUBIC is better for regenerating details from blur
    upscaled = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 4. Strong Sharpening Kernel
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)

    # 5. Denoise (Clean up artifacts from sharpening)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)

    # EasyOCR works fine with Gray or BGR, but returning BGR ensures compatibility
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def resize_for_display(frame):
    if not SHOW_WINDOW:
        return frame

    h, w = frame.shape[:2]
    if w <= DISPLAY_MAX_WIDTH and h <= DISPLAY_MAX_HEIGHT:
        return frame

    scale_w = DISPLAY_MAX_WIDTH / w
    scale_h = DISPLAY_MAX_HEIGHT / h
    scale = min(scale_w, scale_h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h))

def draw_detections(target_frame, detections):
    for (x1, y1, x2, y2, text, color) in detections:
        cv2.rectangle(target_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            target_frame,
            text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS,
        )

def read_easyocr_text(reader, image):
    # detail=0 returns just the text list
    # paragraph=False is often better for plates to prevent merging separate lines erroneously,
    # but paragraph=True is okay if the crop is tight.
    try:
        lines = reader.readtext(image, detail=0)
        if not lines:
            return ""
        # Join lines but prioritize the longest valid-looking string if multiple exist
        return "".join(lines)
    except Exception:
        return ""

# --- MAIN ---
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False) # Change gpu=True if you have CUDA
known_plates = load_known_plates(DB_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 25.0

frame_idx = 0
last_detections = []
plate_history = {}
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if out is None:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (int(frame.shape[1]), int(frame.shape[0])))

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        if last_detections:
            draw_detections(frame, last_detections)
        out.write(frame)
        if SHOW_WINDOW:
            cv2.imshow("result", resize_for_display(frame))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        continue

    det_frame, scale_x, scale_y = resize_for_detection(frame)
    results = model.track(det_frame, verbose=False, persist=True, tracker="bytetrack.yaml", device="cpu")

    frame_detections = []
    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # --- USE IMPROVED PREPROCESSING ---
            processed = preprocess_plate(plate_crop)

            try:
                raw_text = read_easyocr_text(reader, processed)
                text = clean_plate_text(raw_text)
            except Exception:
                text = ""

            display_text = text

            # --- VOTING LOGIC ---
            if track_id is not None and len(text) > 3:
                track_id = int(track_id)
                if track_id not in plate_history:
                    plate_history[track_id] = []

                plate_history[track_id].append(text)
                if len(plate_history[track_id]) > BUFFER_SIZE:
                    plate_history[track_id].pop(0)

                if plate_history[track_id]:
                    display_text = Counter(plate_history[track_id]).most_common(1)[0][0]

            if len(display_text) > 3:
                label, color = get_plate_label_and_color(display_text, known_plates)
                frame_detections.append((x1, y1, x2, y2, label, color))

    last_detections = frame_detections
    if last_detections:
        draw_detections(frame, last_detections)

    out.write(frame)
    if SHOW_WINDOW:
        cv2.imshow("result", resize_for_display(frame))
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()