import sys
import os
import cv2

# Check if the required command line arguments are provided
try:
    labeledImages = sys.argv[1]
    numImages = int(sys.argv[2])
except:
    print("Arguments missing.")
    exit(-1)

# Directory path for stored images
IMAGES_PATH = 'emotions_dataset'
CLASSES_PATH = os.path.join(IMAGES_PATH, labeledImages)

# Create the directory to store the images, if it does not exist
try:
    os.mkdir(IMAGES_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(CLASSES_PATH)
except FileExistsError:
    print("{} directory is already created.".format(CLASSES_PATH))
    print("Collected emotions will be saved with other images in the folder.")

capt = cv2.VideoCapture(0)

count = 0
start = False

# Start the loop to capture images from the video stream
while True:
    ret, frame = capt.read()
    if not ret:
        continue

    # Check if the required number of images have been captured
    if count == numImages:
        break

    # If the flag to capture the current frame is set, save the image
    if start:
        save_path = os.path.join(CLASSES_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, frame)
        count += 1

    # Display a message indicating the number of images captured
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
                (5, 10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting emotion images", frame)

    # Press the 'a' key to start taking pictures
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    # Press the 'q' key to exit video stream
    if k == ord('q'):
        break

print("\n{} emotion images saving to {}".format(count, CLASSES_PATH))

# Release the resources used by the video capture object and close all windows
capt.release()
cv2.destroyAllWindows()
