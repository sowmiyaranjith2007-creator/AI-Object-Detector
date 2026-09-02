from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

model = YOLO("yolo11n.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, "detected_" + file.filename)

    file.save(input_path)

    results = model(input_path)

    result_image = results[0].plot()

    cv2.imwrite(output_path, result_image)

    return render_template(
        "index.html",
        image_path="/" + output_path.replace("\\", "/")
    )

if __name__ == "__main__":
    app.run(debug=True)