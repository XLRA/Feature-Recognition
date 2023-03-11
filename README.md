# Feature Recognition
The Feature Recognition Project is a computer vision and deep learning project that aims to recognize specific characteristics in real-time. The project has three components: Mask Recognition, Gender & Age Recognition, and Emotion Recognition. The Python libraries utilized include TensorFlow, Keras, OpenCV, imutils, and scikit-learn; enabling users to create and train their very own models to recognize unique characteristics utilizing personalized images. The software centers around a GUI that allows users to execute the collection of data, training of the data, and run the recognition all in one interface.

The Mask Recognition component recognizes whether an individual is wearing a mask or not, and users may employ their very own personalized images to construct a model. Gender & Age Recognition utilizes a pre-existing model to predict the gender and age of an individual. For Emotion Recognition, users can design their very own model utilizing personalized images. The program recognizes certain facial expressions to deduce the individual's current emotional state, encompassing a range of emotions: happy, sad, angry, surprised, or neutral.

# Setup
Create virtual environment:
```
Install python 3.7 (version used: 3.7.3)
pip3 install virtualenv
```
In each folder create and activate three separate virtual environments by using the command below:

```
cd folder_name (Mask_Recognition, GenderAge_Recognition, Emotion_Recognition)
virtualenv env_name (ex: env1, env2, and env3)
source env_name/bin/activate
```
Install the required libraries below for each of the virtual environments:

Required Sudo Libraries:
```
sudo apt-get update
sudo apt install -y libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
```
Required Python Libraries:
```
pip3 install numpy==1.20.1 
pip3 install opencv-contrib-python==4.1.0.25
pip3 install imutils==0.5.4
pip3 install tensorflow==2.1.0
pip3 install scikit-learn==1.0.2
pip3 install pillow==9.4.0
pip3 install h5py==2.10.0
pip3 install protobuf==3.20.3
```

# Usage
Configure the variables within the GUI_CONFIG.py file with your own pathway to the directories and there respective enviroments:

Run the Feature Recognition GUI:
```
python3 GUI_Feature_Recognition.py
```
For the Mask Recognition and Emotion Recognition:
```
1. Collect the images
2. Train the images
3. Run the final recognition. 
```
For GenderAge Recognition:
```
1. Run the final recognition as it has pre-trained data.
```


# Potential Errors/Solutions
If there is a problem with the tensorflow version on a raspberry pi buster, run the command below in the virtual enviroment:

```
pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.1.0/tensorflow-2.1.0-cp37-none-linux_armv7l.whl 
```

# Resources Utilized
Pretrained data for GenderAge_Recognition: 
https://talhassner.github.io/home/publication/2015_CVPR 

TensorFlow version:
https://github.com/lhelontra/tensorflow-on-arm/releases 
