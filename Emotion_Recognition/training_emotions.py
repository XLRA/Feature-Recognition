from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Initialize hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 30
BATCH_SIZE = 32

# Define the path to the dataset directory
EMOTIONS_DATASET = "emotions_dataset"
data = []
labels = []

# Loop over all subdirectories in the emotion images dataset and load the images and labels
for directory in os.listdir(EMOTIONS_DATASET):
    label = os.path.join(EMOTIONS_DATASET, directory)

    if not os.path.isdir(label):
        continue

    for item in os.listdir(label):
        # Prevent unnecessary files
        if item.startswith("."):
            continue

        # Preprocess the input picture (224x224) by loading it.
        image = load_img(os.path.join(label, item), target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # Append the preprocessed image and its corresponding label to the data and labels lists
        data.append(image)
        labels.append(label)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# Convert the data and labels lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the data into 80% training and 20% testing sets
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)

# Augment the training data using transformations
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Load the MobileNetV2 model without the top layers
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Add custom top layers to the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)

# Combine the base model with the top layers
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers of the base model, so they are not trainable
for layer in baseModel.layers:
    layer.trainable = False

# Compiling the model
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# Train the model using the data generator and evaluate it on the validation set
H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
              steps_per_epoch=len(trainX) // BATCH_SIZE,
              validation_data=(testX, testY),
              validation_steps=len(testX) // BATCH_SIZE,
              epochs=EPOCHS)

# Make predictions on the test set based on largest probability
probPred = model.predict(testX, batch_size=BATCH_SIZE)
probPred = np.argmax(probPred, axis=1)

# Formatted report of training results
print(classification_report(testY.argmax(axis=1), probPred, target_names=le.classes_))

# Saving model
print("Saving training model for emotion recognition...")
model.save("EmotionsRecognition.h5")
