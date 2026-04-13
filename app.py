from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import base64
import cv2
from flask import flash, redirect,session, url_for, render_template, request

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = "EmotionDetectionSecretKey"  # ✅ MUST ADD (VERY IMPORTANT)
# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained model
model = load_model("model/emotion_model.hdf5", compile=False)

# Classes
classes = ['angry','disgust','fear','happy','sad','surprise','neutral']

# HOME PAGE
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/detect", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filepath = None   # ✅ MUST ADD (VERY IMPORTANT)

    if request.method == "POST":
        print("Request received")

        # ---------- FILE UPLOAD ----------
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

        # ---------- CAMERA INPUT ----------
        else:
            img_data = request.form.get("image")

            if img_data:
                img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)

                filepath = os.path.join(app.config["UPLOAD_FOLDER"], "capture.png")
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
            else:
                return render_template("index.html")

        # ---------- FACE DETECTION ----------
        img = cv2.imread(filepath)

        if img is None:
            return render_template("index.html", result="Image not readable")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        # ✅ fallback (VERY IMPORTANT)
        if len(faces) == 0:
            print("No face detected, using full image")
            face = gray
        else:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]

        # ---------- PREPROCESS ----------
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        # ---------- PREDICTION ----------
        prediction = model.predict(face)

        confidence = float(np.max(prediction) * 100)
        result = classes[np.argmax(prediction)]

        # 🔥 THRESHOLD LOGIC
        if confidence < 20:
            result = "Invalid / No Face"

        print("Prediction:", result, confidence)

    # ✅ SAFE RETURN
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=filepath
    )
    
    
id="savefile"
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        message = request.form['message']

        # Save to file
        with open("messages.txt", "a") as f:
            f.write(f"Name: {name}\n")
            f.write(f"Mobile: {mobile}\n")
            f.write(f"Email: {email}\n")
            f.write(f"Message: {message}\n")
            f.write("-----\n")

        flash("Message sent successfully! ✅")

        return redirect(url_for('contact'))   # 🔥 IMPORTANT

    return render_template('contact.html')

@app.route('/messages')
def messages():

    if not session.get('admin'):
        return redirect(url_for('login'))   # 🔥 protect page

    # your existing code
    data = []
    with open("messages.txt", "r") as f:
        lines = f.readlines()

    temp = {}
    for line in lines:
        if "Name:" in line:
            temp["name"] = line.split(":")[1].strip()
        elif "Mobile:" in line:
            temp["mobile"] = line.split(":")[1].strip()
        elif "Email:" in line:
            temp["email"] = line.split(":")[1].strip()
        elif "Message:" in line:
            temp["message"] = line.split(":")[1].strip()
        elif "-----" in line:
            data.append(temp)
            temp = {}

    return render_template("messages.html", messages=data)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "1234":
            flash("Login successful! ✅", "success")
            session['admin'] = True
            return redirect(url_for('messages'))
        else:
            flash("Invalid credentials ❌", "error")

    return render_template('login.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route("/test-css")
def test_css():
    return app.send_static_file("css/index.css")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)