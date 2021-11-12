#!/usr/bin/python3
import cv2

def faceExtractor(img):
    face_classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey,1.3,5)
    if faces is():
        return None
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
    
    return cropped_face

def trainModel():
    device = cv2.VideoCapture(0)
    count = 0
    while True:
        ret,frame = device.read()
        if faceExtractor(frame) is not None:
            count+=1
            face = cv2.resize(faceExtractor(frame),(600,600))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            fileNamePath = f"collections/{count}.png"
            cv2.imwrite(fileNamePath,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow("Cropped face",face)

        else:
            print("face not Found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:
            break

    device.release()
    cv2.destroyAllWindows()
    print("Collection of Samples Completed")