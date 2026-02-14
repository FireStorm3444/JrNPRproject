import re
from ultralytics import YOLO
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import Counter
import sqlite3

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# --- CONFIGURATION (CPU-ORIENTED DEFAULTS) ---
MODEL_PATH = "best.pt"
VIDEO_PATH = "test_videos/test_6.mp4"
OUTPUT_PATH = "output_videos/output_6.3.mp4"
DB_PATH = "vehicles.db"
FRAME_SKIP = 2
FRAME_RESIZE_WIDTH = None  # set to int (e.g., 1280) to speed up detection
SHOW_WINDOW = True
DISPLAY_MAX_WIDTH = 1260
DISPLAY_MAX_HEIGHT = 1080
OCR_UPSCALE = 2
OCR_USE_OTSU = False
TEXT_SCALE = 1
TEXT_THICKNESS = 2
BUFFER_SIZE = 10

def clean_plate_text(text: str) -> str:
    text = text.upper()
    # Remove special characters
    text = re.sub(r"[^A-Z0-9]", "", text)

    # --- HEURISTIC FIXES ---
    # Common OCR confusions
    corrections = {
        'O': '0', 'Q': '0', 'D': '0',  # Letters detected as Numbers
        'I': '1', 'L': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6'
    }


    text_list = list(text)

    # Try to fix the last 4 characters (usually numbers)
    if len(text) >= 4:
        for i in range(len(text) - 4, len(text)):
            char = text_list[i]
            if char in ['O', 'Q', 'D', 'S', 'Z', 'B']:  # If letter found in number spot
                # Swap letter to number
                if char == 'O' or char == 'D' or char == 'Q': text_list[i] = '0'
                if char == 'S': text_list[i] = '5'
                if char == 'Z': text_list[i] = '2'
                if char == 'B': text_list[i] = '8'

    final_text = "".join(text_list).replace("IND", "")
    return final_text

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

def extract_text_lines(result):
    if isinstance(result, list) and result:
        first = result[0]

        # New PaddleOCR API result format
        if isinstance(first, dict) and "rec_texts" in first:
            return first.get("rec_texts", [])

        # Old format fallback
        if isinstance(first, list):
            lines = []
            for line in first:
                if len(line) >= 2:
                    lines.append(line[1][0])
            return lines

    return []

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

def preprocess_plate(plate_crop):
    if plate_crop.size == 0:
        return plate_crop

    # 1. Grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (Crucial for the dull lighting in your image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Smart Upscale
    h, w = enhanced.shape
    target_height = 64
    scale = target_height / h

    # INTER_CUBIC is better for regenerating details from blur
    upscaled = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 4. Sharpening Kernel (Fixes the "M" looking like "N" blur)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)

    # 5. Denoise (Optional, but helps after sharpening)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)

    # Convert back to BGR for Paddle
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
        cv2.putText(target_frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, TEXT_THICKNESS)

model = YOLO(MODEL_PATH)

known_plates = load_known_plates(DB_PATH)

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en"
)

video_path = VIDEO_PATH
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {video_path}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS) or 25.0,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

FRAME_SKIP = FRAME_SKIP
frame_idx = 0
last_detections = []
plate_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        # Check if boxes exist
        if r.boxes is None: continue

        # We need boxes AND IDs. If tracking fails to assign an ID, skip or treat as new.
        boxes = r.boxes.xyxy.cpu().numpy()

        # Get IDs if available (YOLO might return None for IDs briefly)
        track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # --- SCALE COORDINATES BACK ---
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            # --- PADDING ---
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1: continue

            # --- CROP & OCR ---
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            thr_bgr = preprocess_plate(plate_crop)

            # Run OCR
            try:
                result = ocr.predict(thr_bgr)  # Using your existing method
                lines = extract_text_lines(result)
                raw_text = "".join(lines)
                text = clean_plate_text(raw_text)
            except Exception:
                text = ""

            display_text = text  # Default to current text
            print(display_text, track_id, end="\r")

            # --- LOGIC: PER-ID VOTING ---
            # Only use history if we have a valid Track ID and text is decent
            if track_id is not None and len(text) > 3:
                track_id = int(track_id)

                # Create a list for this specific car if it doesn't exist
                if track_id not in plate_history:
                    plate_history[track_id] = []

                plate_history[track_id].append(text)

                # Limit size
                if len(plate_history[track_id]) > BUFFER_SIZE:
                    plate_history[track_id].pop(0)

                # Vote specifically for THIS car
                if plate_history[track_id]:
                    most_common, count = Counter(plate_history[track_id]).most_common(1)[0]
                    display_text = most_common

            # --- DRAWING ---
            # FIXED: Only append ONCE.

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
out.release()
cv2.destroyAllWindows()
