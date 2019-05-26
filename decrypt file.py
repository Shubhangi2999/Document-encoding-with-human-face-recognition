import numpy as np
import os
import cv2
import subprocess as sp
from shutil import copyfile

def rename():
    f=open("file.txt","w")
    copyfile("input_file.txt","file.txt")
    f.close()
    os.remove("input_file.txt")
    os.rename("output2.txt","input_file.txt")
    os.rename("file.txt","output2.txt")
    os.remove("dictionary.txt")
    os.remove("output2.txt")

def decode(file):
    g = open(file,"r")
    s = g.readline()

    encoder=[]
    characters = []
    with open('dictionary.txt') as p:
        for line in p:
            line = line.strip('\n')
            w = line.split("=")
            characters.append(w[0])
            encoder.append(w[1])

    t = ""
    j =""
    for x in s:
        j = j + x
        for y in encoder:
            if j == y:
                q = encoder.index(y)
                t = t+ characters[q]
                j = ""
    print (t)
    f = open("output2.txt","w")
    f.write(t)

subjects = ["", "Person 1", "Person 2"]

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

haar_face_cascade = cv2.CascadeClassifier(r'''C:\Users\Administrator\Desktop\algo project\opencv-files\haarcascade_frontalface_alt.xml''')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_faces(f_cascade, colored_img, scaleFactor):
    img_copy = np.copy(colored_img)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    cv2.rectangle(colored_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return gray[y:y+w, x:x+h], faces[0]

def draw_text(img, text, x, y,confidence):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), int(1.5))    
    cv2.putText(img, str(confidence), (x+w,y+h+100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

def capture(n):
    camera = cv2.VideoCapture(0)
    i = 0
    while i < n:
        input('Press Enter to capture')
        return_value, image = camera.read()
        cv2.imwrite('opencv'+str(i)+'.png', image)
        img_c=np.copy(image)
        cv2.imshow('opencv.png', img_c)
        cv2.waitKey(1000)
        i += 1
    del(camera)

w=0
h=0

def predict(test_img):
    face, rect = detect_faces(haar_face_cascade, test_img, 1.1)
        
    label, confidence = face_recognizer.predict(face)
    if (confidence < 100):
        label = subjects[label]
        confidence = "  {0}%".format((round(confidence)))  
    else:
        label = subjects[label]
        confidence = "  {0}%".format(abs(round(100 - confidence)))
    draw_text(test_img, label, rect[0], rect[1]-5,confidence)
    cv2.imshow(subjects[1], test_img)
    cv2.waitKey(10000)
    
    if face is not None:
        file=input("Enter the name of the  file you want to decrypt: ")
        print("Decrypting...")
        decode(file)
        cv2.waitKey(50)
        os.system('cls')
        print("Your file has been decoded!")
        rename()
    return test_img

face_recognizer.read(r'''C:\Users\Administrator\Desktop\algo project\trainer\trainer.yml''')
capture(1)
test1 = cv2.imread(r'''C:\Users\Administrator\Desktop\algo project\opencv0.png''')
haar_detected_img = predict(test1)
