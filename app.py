from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load models with correct paths
face_model = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
age_model = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
gender_model = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def detect(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), swapRB=False)
    face_model.setInput(blob)
    detections = face_model.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)
            
            # Gender prediction
            gender_model.setInput(face_blob)
            gender_preds = gender_model.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Age prediction
            age_model.setInput(face_blob)
            age_preds = age_model.forward()
            age = age_list[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
