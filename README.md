# FaceRecognition

This project focuses on implementing face recognition using the RetinaFace algorithm for face detection. Face recognition is a biometric technology that identifies and verifies individuals based on their facial features. The RetinaFace algorithm is chosen for its accuracy and efficiency in detecting faces in images.


# Prerequisites
- Python 3.x installed
- Required python libraries: gradio, dlib, numpy, pandas, retina-face, face-recognition

# Steps
1. Clone the project repository:
   git clone https://github.com/NanoTcodes/FaceRecognition.git
  
2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   python app.py

# User Guide

Uploading Images

Multiple ways for taking input image have been provided using gr.Image()
Users can upload images from pc, or attendance can be taken from all the images from pre-specified folders with attendance images , or the user can upload images via their webcam as well.
These input images are processed as numpy arrays.
The user also needs to specify the date of the class for which attendance is being taken.
# Output
A csv file "attendance.csv" is maintained which will keep getting updated with every new attendance taken.
