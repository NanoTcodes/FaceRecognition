import cv2
import face_recognition
import pickle
import os
import numpy as np
import torch
from PIL import Image

# Importing student images
folderPath ='trying_face\photos' 
pathList = os.listdir(folderPath)
#print(pathList)#contains name or path of each image
imgList = []
studentIds = []
for path in pathList:

    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

    # print(path)
    # print(os.path.splitext(path)[0])
print(studentIds)
print(len(imgList))
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # Check if the image is not empty
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            # Check if face_encodings found faces in the image
            if encode:
                encodeList.append(encode[0])
        else:
            print("Image is empty or invalid.")

    return encodeList

print("Encoding Started ...")

encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")