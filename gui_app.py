import os
import re
import sqlite3
import threading
import uuid
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import easyocr
from paddleocr import PaddleOCR

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
DB_PATH = "vehicles.db"
TEST_VIDEOS_DIR = "test_videos"
OUTPUT_DIR = "output_videos"
FRAME_SKIP = 1
FRAME_RESIZE_WIDTH = None
TEXT_SCALE = 1.5
TEXT_THICKNESS = 3

BUFFER_SIZE = 10

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class StreamSession:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.stop_event = threading.Event()
        self.finished = False
        self.completed = False
        self.stopped_by_user = False


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_plates_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS plates (text TEXT NOT NULL)")
    conn.commit()


def refresh_known_plates() -> None:
    global known_plates
    known_plates = load_known_plates(DB_PATH)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def clean_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    corrections = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    text_list = list(text)
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


def preprocess_plate(plate_crop):
    if plate_crop.size == 0:
        return plate_crop

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    h, _ = enhanced.shape
    target_height = 64
    scale = target_height / h
    upscaled = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


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
    try:
        lines = reader.readtext(image, detail=0)
        if not lines:
            return ""
        return "".join(lines)
    except Exception:
        return ""


def extract_text_lines(result):
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "rec_texts" in first:
            return first.get("rec_texts", [])
        if isinstance(first, list):
            lines = []
            for line in first:
                if len(line) >= 2:
                    lines.append(line[1][0])
            return lines
    return []


def list_test_videos() -> List[str]:
    if not os.path.isdir(TEST_VIDEOS_DIR):
        return []
    videos = []
    for name in os.listdir(TEST_VIDEOS_DIR):
        _, ext = os.path.splitext(name)
        if ext.lower() in ALLOWED_VIDEO_EXTS:
            videos.append(name)
    return sorted(videos)


def resolve_video_path(video_name: str) -> str:
    videos = set(list_test_videos())
    if video_name not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return os.path.join(TEST_VIDEOS_DIR, video_name)


known_plates = load_known_plates(DB_PATH)
yolo_model = YOLO(MODEL_PATH)
easy_reader = easyocr.Reader(["en"], gpu=False)
paddle_reader = PaddleOCR(use_textline_orientation=True, lang="en")
sessions: Dict[str, StreamSession] = {}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    videos = list_test_videos()
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})


@app.on_event("startup")
def startup() -> None:
    conn = get_db_connection()
    try:
        ensure_plates_table(conn)
    finally:
        conn.close()


@app.get("/plates")
def list_plates():
    conn = get_db_connection()
    try:
        ensure_plates_table(conn)
        rows = conn.execute("SELECT rowid as id, text FROM plates ORDER BY rowid DESC").fetchall()
    finally:
        conn.close()

    plates = [{"id": row["id"], "text": row["text"]} for row in rows]
    return JSONResponse({"plates": plates})


@app.post("/plates")
def add_plate(payload: Dict[str, str]):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Plate text required")

    conn = get_db_connection()
    try:
        ensure_plates_table(conn)
        conn.execute("INSERT INTO plates (text) VALUES (?)", (text,))
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    finally:
        conn.close()

    refresh_known_plates()
    return JSONResponse({"id": new_id, "text": text})


@app.put("/plates/{plate_id}")
def update_plate(plate_id: int, payload: Dict[str, str]):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Plate text required")

    conn = get_db_connection()
    try:
        ensure_plates_table(conn)
        cur = conn.execute("UPDATE plates SET text = ? WHERE rowid = ?", (text, plate_id))
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Plate not found")

    refresh_known_plates()
    return JSONResponse({"id": plate_id, "text": text})


@app.delete("/plates/{plate_id}")
def delete_plate(plate_id: int):
    conn = get_db_connection()
    try:
        ensure_plates_table(conn)
        cur = conn.execute("DELETE FROM plates WHERE rowid = ?", (plate_id,))
        conn.commit()
    finally:
        conn.close()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Plate not found")

    refresh_known_plates()
    return JSONResponse({"deleted": True})


@app.post("/start")
def start_stream(payload: Dict[str, str]):
    video = payload.get("video", "")
    mode = payload.get("mode", "")
    if mode not in {"easy", "paddle"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    _ = resolve_video_path(video)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_id = uuid.uuid4().hex
    base_name = os.path.splitext(video)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{mode}_{session_id}.mp4")
    sessions[session_id] = StreamSession(output_path)
    stream_url = f"/stream?video={video}&mode={mode}&session_id={session_id}"
    return JSONResponse({"session_id": session_id, "stream_url": stream_url})


@app.post("/stop")
def stop_stream(payload: Dict[str, str]):
    session_id = payload.get("session_id", "")
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.stopped_by_user = True
    session.stop_event.set()
    return JSONResponse({"download_url": f"/download/{session_id}"})


@app.get("/status/{session_id}")
def stream_status(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    download_url = None
    if session.completed and os.path.exists(session.output_path):
        download_url = f"/download/{session_id}"

    return JSONResponse({
        "finished": session.finished,
        "completed": session.completed,
        "download_url": download_url,
    })


@app.get("/download/{session_id}")
def download_video(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not os.path.exists(session.output_path):
        raise HTTPException(status_code=404, detail="Processed video not found")
    return FileResponse(session.output_path, filename=os.path.basename(session.output_path))


@app.get("/stream")
def stream(video: str, mode: str, session_id: str):
    if mode not in {"easy", "paddle"}:
        raise HTTPException(status_code=400, detail="Invalid mode")

    video_path = resolve_video_path(video)
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video")

        frame_idx = 0
        last_detections: List[Tuple[int, int, int, int, str, Tuple[int, int, int]]] = []
        plate_history: Dict[int, List[str]] = {}
        out = None
        completed = False

        while True:
            if session.stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                completed = True
                break

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    session.output_path,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS) or 25.0,
                    (int(frame.shape[1]), int(frame.shape[0])),
                )

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                if last_detections:
                    draw_detections(frame, last_detections)
            else:
                det_frame, scale_x, scale_y = resize_for_detection(frame)
                results = yolo_model.track(
                    det_frame,
                    verbose=False,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device="cpu",
                )

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

                        processed = preprocess_plate(plate_crop)

                        if mode == "easy":
                            raw_text = read_easyocr_text(easy_reader, processed)
                        else:
                            result = paddle_reader.predict(processed)
                            lines = extract_text_lines(result)
                            raw_text = "".join(lines)

                        text = clean_plate_text(raw_text)
                        display_text = text

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

            if out is not None:
                out.write(frame)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()
        if out is not None:
            out.release()
        session.finished = True
        session.completed = completed and not session.stopped_by_user

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

