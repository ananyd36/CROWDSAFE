from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from image_detect import *
from video_detect import *
from cam_detect import *
import os
import torch

__author__ = 'Anany'
__source__ = ''

app = Flask(__name__)
# UPLOAD_FOLDER = "C:\Users\Arpit Sharma\Desktop\Friendship goals\content\yolov5\static\uploads"
DETECTION_FOLDER = r'./testing/output/images'
detection_folder_video = r'./testing/output/videos'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    print("I about")
    return render_template("about.html")

@app.route("/images",methods = ["GET","POST"])
def images():
    if request.method == 'POST':
        f = request.files['myfile']
        filename = secure_filename(f.filename)
        print(filename)
        file_path = os.path.join("./testing/input/images", filename)
        print(file_path)
        f.save(file_path)
        with torch.no_grad():
            detect_image(file_path, DETECTION_FOLDER)
        return render_template("images.html", fname = filename)

@app.route("/videos",methods = ["POST"])
def videos():
    if request.method == 'POST':
        f = request.files['myvideos']
        filename = secure_filename(f.filename)
        print(filename)
        file_path = os.path.join("./testing/input/videos", filename)
        print(file_path)
        f.save(file_path)
        with torch.no_grad():
            video_detect(file_path, detection_folder_video)
        return render_template("videos.html", fname = filename)

@app.route("/camera",methods = ["POST"])
def camera():
    with torch.no_grad():
            camera_detect()
    return render_template("camera.html")


if __name__ == "__main__":
    app.run(debug = True)
