import numpy as np
import os
import cv2

#ax1 = plt.subplot2grid((40,40), (0,0), rowspan=40, colspan=40)
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

def encode(file):
    f = open(file,"r")
    s = f.readline()
    s = s.lower()
    t = list(set(s))
    #print ("The String: ",s)
    q = len(t)
    #print ("The char list: ", t)
    b = [0]*q
    for x in s:
      count = s.count(x)
      y = t.index(x)
      if b[y] == 0:
        b[y] = count
    #print ("Frequency Table before sorting: ")
    #print (b)


    new = []
    newchar = []
    while b:
        max = b[0]
        for x in b:
            if x > max:
                max= x
        c=b.index(max)
        new.append(max)
        newchar.append(t[c])
        t.remove(t[c])
        b.remove(max)
    #print ("Frequency Table after sorting: ")
    #print (new)
    #print (newchar)
    k = 0
    for x in new:
        if x != 0:
            k = k + 1
    j = 2*k-1
    #print ("Number of nodes in the tree: ",j)

    #print ("Number of leafs in the tree: ",k)


    encoder = ["null"]*k
    #print ("encoder",encoder)


    bintree = list(new)
    strtree = list(newchar)

    print ("binarytree",bintree)
    print ("Stringtree",strtree)

    for q in range(len(bintree)-1,-1,-1):
        left = 999
        right = 999
        leftstr = ""
        rightstr = ""
        for i in range(len(bintree)-1,-1,-1):
            if bintree[i] < right:
                right = bintree[i]
                rightstr = strtree[i]
                j = i
        for y in rightstr:
            ci = newchar.index(y)
            if encoder[ci]=="null":
                encoder[ci]="0"
            else:
                encoder[ci]="0"+encoder[ci]
        v = strtree.index(rightstr)
        strtree.pop(v)
        bintree.pop(v)
        for i in range(len(bintree)-1,-1,-1):

            if bintree[i] < left or bintree[i] == right:
                    left = bintree[i]
                    leftstr = strtree[i]
                    z = i
        for x in leftstr:
            ci = newchar.index(x)
            if encoder[ci]=="null":
                encoder[ci]="1"
            else:
                encoder[ci]="1"+encoder[ci]
        w = strtree.index(leftstr)
        strtree.pop(w)
        bintree.pop(w)

        summation = right + left
        strsum = leftstr + rightstr
        strtree.insert(0,strsum)
        bintree.insert(0,summation)


        #print ("valuetable",bintree)
        #print ("strlist",strtree)
        #print ("encoder",encoder)
        if len(bintree)==1:
            break
    #print ("char",newchar)
    f.close()
    t = ""
    for i in s:
        for j in newchar:
            if j==i:
                p = newchar.index(j)
                t = t + encoder[p]

    #print (t)
    g = open("output.txt","w")
    g.write(t)
    g.close()
    os.remove(file)
    os.rename("output.txt",file)
    p = open("dictionary.txt","w")

    for i in range (0,len(encoder),1):
        q = newchar[i]
        t = encoder[i]
        w=q+"="+t
        print (w)
        p.write(w+"\n")
        w = ""
    p.close()
    

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
        file=input("Enter the name of the file you want to encode: ")
        encode(file)
        print("Your file has been encoded!")
    return test_img

face_recognizer.read(r'''C:\Users\Administrator\Desktop\algo project\trainer\trainer.yml''')
capture(1)
test1 = cv2.imread(r'''C:\Users\Administrator\Desktop\algo project\opencv0.png''')
haar_detected_img = predict(test1)
#cv2.destroyAllWindows()

