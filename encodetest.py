import cv2
import face_recognition  # For face recognition and face data
import pickle  # To save encodings for later use
import os
import glob

# Folder containing student images
folderPath = 'D:/Minor Project/Project/Student'
PathList = os.listdir(folderPath)
print("Images found:", PathList)

imgList = []  # Initialize an empty list to store images
StudentName = []

for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))  # Add images to list
    StudentName.append(os.path.splitext(path)[0])  # Extract names without extensions

# Function to encode all images
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("Face not detected in image, skipping...")
    return encodeList

# Encode images and save to file
print("Encoding Started...")
EncodeListKnown = findEncodings(imgList)
EncodeListKnownWithIds = [EncodeListKnown, StudentName]
print("Encoding Completed")

# Save encodings to pickle file
with open("Encodefile.p", 'wb') as file:
    pickle.dump(EncodeListKnownWithIds, file)
print("File Saved")
