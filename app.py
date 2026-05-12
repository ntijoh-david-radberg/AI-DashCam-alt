from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import re
import time
import os

app = Flask(__name__)

MODEL_PATH = "best.pt"
PI_CAMERA_URL = "http://172.20.10.5:5001/video_feed"

latest_plate = "Ingen skylt hittad"
latest_confidence = 0
latest_time = "Aldrig"
latest_image_path = "static/latest_plate.jpg"

last_plate_signature = None
last_ocr_result = None
last_ocr_time = 0

os.makedirs("static", exist_ok=True)

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture(PI_CAMERA_URL)


def clean_plate_text(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    match = re.search(r"[A-Z]{3}[0-9]{2}[A-Z0-9]", text)

    if match:
        return match.group(0)

    return None


def read_plate_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    results = reader.readtext(gray)

    for result in results:
        text = result[1]
        cleaned = clean_plate_text(text)

        if cleaned:
            return cleaned

    return None


def get_plate_signature(x1, y1, x2, y2):
    rounded_x1 = round(x1 / 40)
    rounded_y1 = round(y1 / 40)
    rounded_x2 = round(x2 / 40)
    rounded_y2 = round(y2 / 40)

    return f"{rounded_x1}_{rounded_y1}_{rounded_x2}_{rounded_y2}"


def generate_frames():
    global latest_plate, latest_confidence, latest_time
    global last_plate_signature, last_ocr_result, last_ocr_time

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        small_frame = cv2.resize(frame, (640, 360))
        results = model(small_frame, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size == 0:
                    continue

                plate_signature = get_plate_signature(x1, y1, x2, y2)

                reg_number = None
                current_time = time.time()

                if (
                    plate_signature != last_plate_signature
                    and current_time - last_ocr_time > 2
                ):
                    reg_number = read_plate_text(plate_img)
                    last_plate_signature = plate_signature
                    last_ocr_result = reg_number
                    last_ocr_time = current_time
                else:
                    reg_number = last_ocr_result

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = "REG SKYLT"

                if reg_number:
                    latest_plate = reg_number
                    latest_confidence = round(confidence * 100, 1)
                    latest_time = time.strftime("%H:%M:%S")

                    cv2.imwrite(latest_image_path, plate_img)

                    label = f"{reg_number} ({latest_confidence}%)"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/plate_data")
def plate_data():
    biluppgifter_url = ""

    if latest_plate != "Ingen skylt hittad":
        biluppgifter_url = f"https://biluppgifter.se/fordon/{latest_plate}"

    return jsonify({
        "plate": latest_plate,
        "confidence": latest_confidence,
        "time": latest_time,
        "image": "/static/latest_plate.jpg",
        "biluppgifter_url": biluppgifter_url
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)