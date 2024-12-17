import cv2
import os
import face_recognition
import pickle
import numpy as np
import cvzone

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 400)  # Width
cap.set(4, 370)  # Height

# Load the background image
imgBackground = cv2.imread('D:/Minor Project/Project/attendance.png')

# Load mode images from the folder
folderModePath = 'D:/Minor Project/Project/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load the encoding file
print("Loading Encode File Started...")
with open('Encodefile.p', 'rb') as file:
    EncodeListKnownWithIds = pickle.load(file)
EncodeListKnown, StudentName = EncodeListKnownWithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()  # Capture frame from the webcam

    # Resize and convert frame to RGB
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Place the webcam feed and mode image onto the background
    imgBackground[155:155 + 480, 80:80 + 640] = img
    imgBackground[118:118 + 525, 910:910 + 350] = imgModeList[2]

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurFrame):
        # Compare the detected face with known encodings
        matches = face_recognition.compare_faces(EncodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(EncodeListKnown, encodeFace)

        # Determine the best match
        matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else -1
        name = "Unknown"

        if matchIndex != -1 and matches[matchIndex] and faceDis[matchIndex] < 0.6:  # 0.6 is the threshold
            name = StudentName[matchIndex]
            print("Known Face Detected:", name)
        else:
            print("Unknown Face Detected")

        # Extract and scale bounding box coordinates
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = 80 + x1, 155 + y1, x2 - x1, y2 - y1

        # Draw a bounding box and display the name
        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
        cv2.putText(imgBackground, name, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the webcam feed and the background image
    cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
