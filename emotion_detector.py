from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow INFO and WARNING messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the emotion categories
emotion_categories = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Load trained weights if available
model.load_weights('model.h5')

# Define function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((48, 48))  # Resize image to match model input size
    img = img.convert('L')  # Convert to grayscale
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Define function to predict emotion from an image
def predict_emotion(image):
    emotion_prediction = model.predict(image)
    emotion_label = emotion_categories[np.argmax(emotion_prediction)]
    return emotion_label

# Define function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))  # Resize image to fit the GUI
        img = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img)
        uploaded_image_label.image = img
        # Save the file path for emotion detection
        upload_image.file_path = file_path

# Define function to open camera and capture image
def open_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        cap.release()
        img = Image.open("captured_image.jpg")
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img)
        uploaded_image_label.image = img
        # Save the file path for emotion detection
        upload_image.file_path = "captured_image.jpg"

# Define function to detect emotion
def detect_emotion():
    if hasattr(upload_image, 'file_path'):
        image_path = upload_image.file_path
        img = preprocess_image(image_path)
        emotion_label = predict_emotion(img)
        emotion_result_label.config(text=f"Detected Emotion: {emotion_label}", font=("Arial", 16, "bold"))
    else:
        emotion_result_label.config(text="Please upload an image first.")

# Create the main GUI window
root = Tk()
root.title("Emotion Detector")
root.geometry("750x650")

# Create frame for image display
image_frame = Frame(root)
image_frame.pack(pady=10)

# Create and position GUI elements inside image frame
uploaded_image_label = Label(image_frame)
uploaded_image_label.pack()

# Create frame for buttons
button_frame = Frame(root)
button_frame.pack(pady=10)

# Create and position GUI elements inside button frame
upload_button = Button(button_frame, text="Upload Image", bg = "Purple", command=upload_image, font=("Arial", 16))
upload_button.pack(side=LEFT, padx=10)

open_camera_button = Button(button_frame, text="Open Camera", bg = "Purple", command=open_camera, font=("Arial", 16))
open_camera_button.pack(side=LEFT, padx=10)

# capture_button = Button(button_frame, text="Capture Image", bg = "Purple", command=lambda: open_camera(), font=("Arial", 16))
# capture_button.pack(side=LEFT, padx=10)

detect_button = Button(button_frame, text="Detect Emotion", bg = "Purple", command=detect_emotion, font=("Arial", 16))
detect_button.pack(side=RIGHT, padx=10)

# Define label for displaying detected emotion
emotion_result_label = Label(root, text="", font=("Arial", 16, "bold"))
emotion_result_label.pack()

root.mainloop()
