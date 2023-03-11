import os
import sys
import cv2

# Check if the required command line arguments are provided
try:
    labeledImages = sys.argv[1]
    numImages = int(sys.argv[2])
except:
    print("Arguments missing.")
    exit(-1)

# Directory path for stored images
IMG_SAVE_PATH = 'mask_dataset'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, labeledImages)

# Create the directory to store the images, if it does not exist
try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory is already created.".format(IMG_CLASS_PATH))
    print("Collected images with mask and no mask will be saved with other images in the folder.")

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
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, frame)
        count += 1

    # Display a message indicating the number of images captured
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
                (5, 10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting mask images", frame)

    # Press the 'e' key to start taking pictures
    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    # Press the 'q' key to exit video stream
    if k == ord('q'):
        break

print("\n{} mask images saved to {}".format(count, IMG_CLASS_PATH))

# Release the resources used by the video capture object and close all windows
capt.release()
cv2.destroyAllWindows()
