# NPR Project (PaddleOCR)

Minimal CPU-friendly runner for `test_paddle.py`.

## Quick Start

```bash
python run_paddle.py
```

## Notes

- Tune `FRAME_RESIZE_WIDTH`, `FRAME_SKIP`, and `OCR_UPSCALE` in `test_paddle.py` for speed.
- Set `SHOW_WINDOW = False` to save CPU during batch runs.

## GUI (FastAPI)

Run the GUI to select a test video and OCR mode and stream processed frames.

```bash
python run_gui.py
```

Then open `http://localhost:8000` in your browser.

Notes:
- GUI reads videos from `test_videos/`.
- Streamed frames are processed live with YOLO + OCR and show KNOWN/UNKNOWN labels.
