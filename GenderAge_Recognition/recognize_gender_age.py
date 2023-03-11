import math
import cv2


def recognize_gender_age(net, frame):
    # Defining a function to highlight the face in the input frame.
    # It takes the neural network, input frame and confidence threshold as parameters.
    frameDNN = frame.copy()

    # Creating a copy of the input frame's height and width.
    height = frameDNN.shape[0]
    width = frameDNN.shape[1]

    # Getting the height and width of the input frame.
    faceBlob = cv2.dnn.blobFromImage(frameDNN, 1.0, (300, 300), [104, 117, 123], True, False)

    # Preparing the image for face recognized by resizing it to 300x300,
    # subtracting mean values, and creating a face blob from the image.
    net.setInput(faceBlob)
    recognitions = net.forward()

    # Looping over the recognitions and filtering out the ones with a confidence above the given threshold.
    # Storing the face boxes in a list and drawing rectangles around the recognized faces on the copied frame.
    faceBoxes = []
    for i in range(recognitions.shape[2]):
        confidence = recognitions[0, 0, i, 2]

        if confidence > 0.7:
            x1 = int(recognitions[0, 0, i, 3] * width)
            y1 = int(recognitions[0, 0, i, 4] * height)
            x2 = int(recognitions[0, 0, i, 5] * width)
            y2 = int(recognitions[0, 0, i, 6] * height)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(height / 150)), 8)

    return frameDNN, faceBoxes


# Returning the modified frame and the list of face boxes.
FACE_PROTOTYPE = "face_data/opencv_face_detector.pbtxt"
FACE_MODEL = "face_data/opencv_face_detector_uint8.pb"
AGE_PROTOTYPE = "age_data/age_deploy.prototxt"
AGE_MODEL = "age_data/age_net.caffemodel"
GENDER_PROTOTYPE = "gender_data/gender_deploy.prototxt"
GENDER_MODEL = "gender_data/gender_net.caffemodel"

# Setting the paths to the face, age and gender recognized models.
MODEL_MEANS = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['[0-2]', '[4-6]', '[8-12]', '[15-20]', '[25-32]', '[38-43]', '[48-53]', '[60-100]']

# Setting the model mean values, age and gender labels.
faceNN = cv2.dnn.readNet(FACE_MODEL, FACE_PROTOTYPE)
ageNN = cv2.dnn.readNet(AGE_MODEL, AGE_PROTOTYPE)
genderNn = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTOTYPE)

# Loading the face, age and gender recognized models using OpenCV's deep neural network module.
video = cv2.VideoCapture(0)

# Setting the padding around the recognized face
padding = 20

while True:
    # Capturing the video frame and displaying box around face
    _, frame = video.read()
    results, faceBoxes = recognize_gender_age(faceNN, frame)

    # Looping over the face boxes and processing each recognized face
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Preparing the face for gender recognition by resizing it to 227x227 and creating a blob from the image
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEANS, swapRB=False)

        # Setting the input to the gender recognition model and getting the predictions
        genderNn.setInput(blob)
        genderPrediction = genderNn.forward()

        # Getting the gender label with the highest probability
        gender = genderList[genderPrediction[0].argmax()]

        # Setting the input to the age recognition model and getting the predictions
        ageNN.setInput(blob)
        agePrediction = ageNN.forward()

        # Getting the age label with the highest probability
        age = ageList[agePrediction[0].argmax()]

        # Creating the text string to display the predicted gender and age
        display_text = "{}:{}".format(gender, age)

        # Adding the text string to the modified frame
        cv2.putText(results, display_text, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)

        # Displaying the modified frame with the added text
        cv2.imshow("Recognizing the gender and age", results)

    # Press the 'q' key to exit video stream
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close all windows
cv2.destroyAllWindows()
video.stop()
