import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import streamlit as st

# Load your pre-trained model (replace with your model path)
with open('Custom_ResNet50_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_emotions(frame):
    """
    Detects emotions in a given frame.

    Args:
        frame (np.ndarray): The frame to process.

    Returns:
        tuple: (frame, emotions)
            frame (np.ndarray): The processed frame with rectangles and emotions.
            emotions (list): A list of emotions detected in the frame.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    emotions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        emotions.append(emotion)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame, emotions


def app():
    """
    The main function of the Streamlit app.
    """

    st.title("Facial Emotion Recognition App")
    run_app = st.checkbox("Run App")

    frame_window = st.image([])

    if run_app:
        # Start video capture using OpenCV
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check for successful capture
            if not ret:
                print("Error: Unable to capture frame from video source")
                break

            # Detect emotions and get processed frame with emotions
            processed_frame, emotions = detect_emotions(frame.copy())

            # Display the processed frame with emotions
            frame_window.image(processed_frame, channels="BGR")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()