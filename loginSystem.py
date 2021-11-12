#!/usr/bin/python3
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

def faceDetection(img,size=0.5):
    face_classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(600,600))
    
    return img , roi


def login():
    data_path = "collections/"
    onlyFiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    training_data , labels = [],[]
    for i,files in enumerate(onlyFiles):
        image_path = data_path+onlyFiles[i]
        images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images,dtype=np.uint8))
        labels.append(i)

    labels = np.asarray(labels,dtype=np.int32)
    modal = cv2.face.LBPHFaceRecognizer_create()
    modal.train(np.asarray(training_data),np.asarray(labels))
    print("[+] Model Trained Successfully.")
    face_classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        img,face = faceDetection(frame)
        try:
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            result = modal.predict(face)
            if result[1] < 500:
                confidence = (100*(1-(result[1])/300))
                display_string = str(confidence)+ "% Confidence"
            
            cv2.putText(img,display_string,(100,30),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,200),2)
            
            if confidence >75:
                cv2.putText(img,"Unlocked",(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,200),2)
                cv2.imshow("Face Detection",img)
                print("Welcome to New World")
                break
                
            
            else:
                cv2.putText(img,"Locked",(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,0),2)
                cv2.imshow("Face Detection",img)


        except:
            cv2.putText(img,"Face not Found",(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(250,25,0),2)
            cv2.imshow("Face Detection",img)
            pass

        
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()