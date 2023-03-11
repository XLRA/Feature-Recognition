### Feature Recognition

The Feature Recognition Project is a computer vision project that aims to recognize and classify specific facial features in real-time. The project contains three main features: Mask Recognition, Gender & Age Recognition, and Emotion Recognition. The project is built using several powerful libraries in Python, including TensorFlow, Keras, OpenCV, imutils, and scikit-learn, and allows the user to train their own models to recognize specific features using their own pictures. The project can be used with any version of Python 3.7, making it highly flexible for different system setups. The core of the project is a GUI that allows the user to run the entire project with just a click of a button.

The Mask Recognition feature is designed to recognize if a person is wearing a mask or not, and the user can train their own model using their own pictures. The Gender & Age Recognition feature uses a pre-trained model to predict the gender and age of a person in real-time. For Emotion Recognition, the user can also train their own model using their own pictures, and the project will analyze facial expressions to determine the person's current emotional state, such as happy, sad, angry, surprised, or neutral.


### Setup
```
-Install python 3.7 (version used: 3.7.3)

-Installing the virtual environments
pip3 install virtualenv

In each of the folders create a virtual environment by using the command below:

cd folder_name (Mask_Recognition, GenderAge_Recognition, Emotion_Recognition)

virtualenv env_name (ex: env1, env2, and env3)

Activate the virtual environment using the command below:

source env_name/bin/activate

Install the required libraries below within the 3 different virtual environments:

-Required Sudo Libraries:
sudo apt-get update
sudo apt install -y libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test

-Required Python Libraries:
pip3 install numpy==1.20.1 
pip3 install h5py==2.10.0
pip3 install protobuf==3.20.3
pip3 install tensorflow==2.1.0
pip3 install opencv-contrib-python==4.1.0.25
pip3 install pillow==9.4.0
pip3 install imutils==0.5.4
```

### Usage
Configure the GUI_CONFIG.py file with the directories of the main folders and its respective environments:

Run the Feature Recognition GUI:

python3 GUI_Feature_Recognition.py



###Potential Errors/Solutions

If there is a problem with the tensorflow version on a raspberry pi buster:

pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.1.0/tensorflow-2.1.0-cp37-none-linux_armv7l.whl 


### Resources Utilized:

Pretrained data for GenderAge_Recognition: 
https://talhassner.github.io/home/publication/2015_CVPR 

TensorFlow version:
https://github.com/lhelontra/tensorflow-on-arm/releases 
