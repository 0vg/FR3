from flask import Flask, render_template, request, redirect, url_for, make_response, abort, send_from_directory
import os
from werkzeug.utils import secure_filename
import imghdr
import numpy as np
import urllib
import cv2
import base64

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = 'uploads'

face_cascade = cv2.CascadeClassifier('cas/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cas/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cas/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('cas/haarcascade_mcs_mouth.xml')

def detect_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=7)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
        
        # If there is more then 2 change to 19, if 2 pass, if less then 2 change to 7 neighbores
        if len(eyes) > 2:
            eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=19)
        elif len(eyes) < 2:
            eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=7)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
    return img

def detect_nose(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=7)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        nose = nose_cascade.detectMultiScale(roi_gray, minNeighbors=10)
        
        if len(nose) > 1:
            nose = nose_cascade.detectMultiScale(roi_gray, minNeighbors=19)
        elif len(nose) < 1:
            nose = nose_cascade.detectMultiScale(roi_gray, minNeighbors=7)
        
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
            
    return img

def detect_mouth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=7)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        mouth = mouth_cascade.detectMultiScale(roi_gray, minNeighbors=10)
        
        if len(mouth) > 1:
            mouth = mouth_cascade.detectMultiScale(roi_gray, minNeighbors=100)
        elif len(mouth) < 1:
            mouth = mouth_cascade.detectMultiScale(roi_gray, minNeighbors=7)
        
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)
            
    return img

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/out', methods=['POST'])
def success():
    if request.method == 'POST':
        image = request.files['file']
        if image.filename == '':
            return redirect(request.url)
        if image:
            image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(image_array, -1)
            
            img = detect_eyes(img)
            img = detect_nose(img)
            img = detect_mouth(img)
            
            retval, buffer = cv2.imencode('.jpg', img)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/jpeg'
            
            encoded = base64.b64encode(buffer.tobytes())
            return render_template('out.html', image=encoded.decode('utf-8'))
            
            
            

if __name__ == '__main__':
    app.run(debug=True)