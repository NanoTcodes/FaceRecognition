import gradio as gr
import pickle
import os
import numpy as np
from PIL import Image
import face_recognition
from retinaface import RetinaFace
import gradio as gr
import pandas as pd
import cv2




def func(imgg, option, selected_date):
    attendance = [0] * 86
    print("Loading Encode File ...")
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    print("Encode File Loaded")

    source_images = []
    source_faces = []

    if imgg is None:
    
        image_dir ='CS 203 Attendance'
        image_directory = os.path.join(image_dir,f"""{option}""")

        source_images = []
        source_faces = []

        for filename in os.listdir(image_directory):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_path = os.path.join(image_directory, filename)
                source_images.append(image_path)
                #print("folder")

        for image_path in source_images:
            img_arr = cv2.imread(image_path)
            resp = RetinaFace.detect_faces(img_arr)

            def crop_faces(arr):
                face_images = []
                for key in resp.keys():
                    x1,y1,x2,y2=resp[key]["facial_area"]
                    img=Image.fromarray(arr)
                    face = img.crop((x1, y1, x2, y2))
                    face_images.append(face)
                return face_images

            cropped_faces = crop_faces(img_arr)
            source_faces.extend(cropped_faces)

    else:
        
        img_arr = cv2.imread(imgg)
        resp = RetinaFace.detect_faces(img_arr)
        def crop_faces(arr):
            face_images = []
            for key in resp.keys():
                x1,y1,x2,y2=resp[key]["facial_area"]
                img=Image.fromarray(arr)
                face = img.crop((x1, y1, x2, y2))
                face_images.append(face)
            return face_images

        source_faces = crop_faces(img_arr)
    face_arrays = []
    for face in source_faces:
        face_array = np.array(face)
        face_arrays.append(face_array)

    for img in face_arrays:
        face_encodings = face_recognition.face_encodings(img)

        if face_encodings:
            for encode_face in face_encodings:
            # Compare the encoding of the current face with known face encodings
                matches = face_recognition.compare_faces(encodeListKnown, encode_face,tolerance=0.8)
                face_distances = face_recognition.face_distance(encodeListKnown, encode_face)

                match_index = np.argmin(face_distances)

                if matches[match_index]:
                    print("Known Face Detected")
                    print(studentIds[match_index])
                    last_two_digits = int(studentIds[match_index][-2:])
                    attendance[last_two_digits - 1] = 1
    studentIds.sort()
    studentIds.pop()
    studentIds.pop()
    studentIds.pop()
    studentIds.pop()
    studentIds.append('220002018')
    studentIds.append('220002029')
    studentIds.append('220002063')
    studentIds.append('220002081')

        
    df=pd.read_csv("attendance.csv")
    if selected_date not in df.columns:
        df[selected_date] = 0


    for i in range(len(attendance)):
        if attendance[i]==1:
            df[selected_date][i] = attendance[i]
    
    df.to_csv("attendance.csv" , index=False)
    
    

    for i in range(86):
        #print(studentIds[i])
        if(attendance[i]==1):
            attendance[i]="Present"
        else:
            attendance[i]="Absent"
    res = zip(studentIds, attendance)

    res= pd.DataFrame(list(res), columns=['Roll No.','Status'])
    return [res,"attendance.csv"]

demo = gr.Interface(fn=func, inputs=[gr.Image(type="filepath"), gr.Dropdown(
    ["21 August", "23 August", "28 August", "30 August", "4th September", "6th September", "13th September", "13th September(Extra class)", "12th October", "16th October", "18th October"], label="Past Attendance"
), "text"], outputs=["dataframe","file"])

demo.launch(debug=True)
