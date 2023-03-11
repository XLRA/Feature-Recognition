from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np


# Create a face blob from the frame for face recognition
def recognize_emotions(frame, faceNN, emotionNN):
    (height, width) = frame.shape[:2]
    faceBlob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # Set the input for the face recognition model and get recognition
    faceNN.setInput(faceBlob)
    recognition = faceNN.forward()

    # Initialize lists for faces, locations, and predictions
    faces = []
    locations = []
    predictions = []

    # Loop over the recognition
    for i in range(0, recognition.shape[2]):
        # Get the confidence score for the recognition
        confidence = recognition[0, 0, i, 2]

        # If the confidence is higher than 0.5, consider it a face
        if confidence > 0.5:
            # Get the box coordinates for the face
            box = recognition[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the box is within the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

            # Extract the face ROI, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face, location, and prediction to their respective lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # If there is a face, predict their emotions using the emotionNN model
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predictions = emotionNN.predict(faces, batch_size=32)
    return locations, predictions


# Path to the face recognition model files
FACE_PROTO = "face_data/deploy.prototxt"
FACE_MODEL = "face_data/res10_300x300_ssd_iter_140000.caffemodel"

# Load the face recognition model
faceNN = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

# Load the emotion recognition model
print("Loading model for emotion recognition...")
emotionNN = load_model("EmotionsRecognition.h5")

# Start the video stream
print("Video stream starting...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over frames from the video stream
while True:
    # Resize the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detect  different emotions in the frame
    (location, predictions) = recognize_emotions(frame, faceNN, emotionNN)

    # Loop over the face
    for (face_dimension, predict) in zip(location, predictions):
        (startX, startY, endX, endY) = face_dimension
        emotions = ["Happy", "Sad", "Surprised", "Neutral", "Angry"]
        label = emotions[np.argmax(predict)]
        color = (0, 255, 0)

        # Displays the labels and face dimension
        cv2.putText(frame, label, (startX - 50, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Displays frame output
    cv2.imshow("Emotion Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press the 'q' key to exit video stream
    if key == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()
vs.stop()
