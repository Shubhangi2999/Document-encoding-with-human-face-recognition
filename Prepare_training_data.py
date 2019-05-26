import numpy as np
import os
import cv2

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

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)    
    faces = []
    labels = []  
   
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
           
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name  
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:           
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detect_faces(haar_face_cascade, image, 1.1)           
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

faces, labels = prepare_training_data(r'''C:\Users\Administrator\Desktop\algo project\training-data''')
face_recognizer.train(faces, np.array(labels))
face_recognizer.save(r'''C:\Users\Administrator\Desktop\algo project\trainer\trainer.yml''')
