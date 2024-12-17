import cv2
import os
import face_recognition
import pickle
import numpy as np
import cvzone

cap= cv2.VideoCapture(0) #0 for default camera
cap.set (3, 400) #3 means width in open cv
cap.set (4, 370) # 4 means height in open cv

imgBackground = cv2.imread('D:/Minor Project/Project/attendance.png') #import background image

#Import Mode Images comple folder at once

folderModePath = 'D:/Minor Project/Project/Modes'     #Folder which contain all images
modePathList = os.listdir(folderModePath)
imgModeList = [] #initialize an empty list to store images
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))  #append use to add items in list 

#LOAD  THE ENCODING FILE
print("Loading Encode File Started.....")
file = open('Encodefile.p', 'rb')
EncodeListKnownWithIds = pickle.load(file)
file.close()
EncodeListKnown, StudentName = EncodeListKnownWithIds
# print(StudentName) #for testing part
# print("Encode file Loaded")


while True:
    success, img= cap.read() # REad frame from the camera
    
    imgS=cv2.resize(img,(0, 0), None, 0.25, 0.25) #resize image to 1/4 of size
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS) #encode faces
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame) #encode faces in camera frame
    
    imgBackground[155:155+480, 80:80+640]=img  #............... Fix the webcam in background image
    imgBackground[118:118+525, 910:910+350]=imgModeList[2] #Fixing image in background
    
    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurFrame): #zi method to extract and save encodings of codecurr frame in encoFace and same for faceCurFrame in FaceLoc..Location
        matches = face_recognition.compare_faces(EncodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(EncodeListKnown, encodeFace)
        # print("Matches", matches)
        # print("faceDis", faceDis)
        
        matchIndex = np.argmin(faceDis) #to extract the minimum value for exact face recoginition from all images index
        # print("MatchIndex", matchIndex)
        
        if matches[matchIndex]: #to check matched face
            print("Known Face Detected") 
            print(StudentName[matchIndex]) #To check which face matched and print the name of person whose face matched
           #PRINT A BOX ARROUND THE FACE 
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 80+x1, 155+y1, x2-x1, y2-y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
 
    cv2.imshow ("webcam", img) #Display frame
    cv2.imshow ("Face Attendance", imgBackground) #display bachground image
    cv2.waitKey(1) #wait time in mili seconds
