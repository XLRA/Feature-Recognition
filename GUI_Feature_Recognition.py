import tkinter as tk
import subprocess
from GUI_CONFIG import *

#GUI window dimensions
window = tk.Tk()
window.title("Feature Recognition GUI")
window.geometry("520x200")

mask_frame = tk.LabelFrame(window, text="Mask Recognition", padx=5, pady=5)
mask_frame.pack(side="left", fill="both", expand=True)

genderage_frame = tk.LabelFrame(window, text="GenderAge Recognition", padx=5, pady=5)
genderage_frame.pack(side="left", fill="both", expand=True)

emotion_frame = tk.LabelFrame(window, text="Emotion Recognition", padx=5, pady=5)
emotion_frame.pack(side="left", fill="both", expand=True)


#GUI button functions
def run_command(command):
    subprocess.Popen(['lxterminal', '-e', 'bash', '-c', f'cd {MASK_PATH} && {MASK_ENV} && {command}'])

def run_virtualenv2():
    subprocess.Popen(['lxterminal', '-e', 'bash', '-c', f'cd {GENDERAGE_PATH} && {GENDERAGE_ENV} && python3 recognize_gender_age.py'])

def run_virtualenv3():
    subprocess.Popen(['lxterminal', '-e', 'bash', '-c', f'cd {EMOTION_PATH} && {EMOTION_ENV} && python3 recognize_emotions.py'])

def gather_MASKON():
    run_command('python3 collect_mask.py MaskON 50')

def gather_MASKOFF():
    run_command('python3 collect_mask.py MaskOFF 50')

def mask_TrainData():
    run_command('python3 training_mask.py')

def Emotion_Collection(emotion):
    subprocess.Popen(['lxterminal', '-e', 'bash', '-c', f'cd {EMOTION_PATH} && {EMOTION_ENV} && python3 collect_emotions.py {emotion} 100'])
    
def train_EmotionData():
    subprocess.Popen(['lxterminal', '-e', 'bash', '-c', f'cd {EMOTION_PATH} && {EMOTION_ENV} && python3 training_emotions.py'])

def collect_images():
    images_window = tk.Toplevel()
    images_window.title("Image Collection")
    images_window.geometry("200x150")
    images_window.focus()

    maskon_button = tk.Button(images_window, text="Gather MaskON Images", command=gather_MASKON)
    maskon_button.pack(pady=10)

    maskoff_button = tk.Button(images_window, text="Gather MaskOFF Images", command=gather_MASKOFF)
    maskoff_button.pack(pady=10)

def collect_emotions():
    emotions_window = tk.Toplevel()
    emotions_window.title("Emotions Collection")
    emotions_window.geometry("200x200")
    emotions_window.focus()

    emotions = {"Happy": "Happy", "Sad": "Sad", "Angry": "Angry", "Surprised": "Surprised", "Neutral": "Neutral"}

    for emotion, emotion_value in emotions.items():
        button = tk.Button(emotions_window, text=emotion, command=lambda emotion_value=emotion_value: Emotion_Collection(emotion_value))
        button.pack(pady=5)

#GUI buttons
images_button = tk.Button(mask_frame, text="Collect Masks", command=collect_images)
images_button.pack(pady=10)

train_button = tk.Button(mask_frame, text="Train Mask Data", command=mask_TrainData)
train_button.pack(pady=10)

facemask_button4 = tk.Button(mask_frame, text="Mask Recognition", command=lambda: run_command('python3 recognize_mask.py'))
facemask_button4.pack(pady=10)

genderage_recognition_button = tk.Button(genderage_frame, text="GenderAge Recognition", command=run_virtualenv2)
genderage_recognition_button.pack(side="top", pady=5)

emotion_button = tk.Button(emotion_frame, text="Collect Emotions", command=collect_emotions)
emotion_button.pack(pady=10)

train_button = tk.Button(emotion_frame, text="Train Emotion Data", command=train_EmotionData)
train_button.pack(pady=10)

detect_button = tk.Button(emotion_frame, text="Emotion Recognition", command=run_virtualenv3)
detect_button.pack(pady=10)

window.mainloop() 
