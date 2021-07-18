import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #for face recognition
eye_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')   #for eye detection

cap=cv2.VideoCapture(0)

while True:
    _, img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    eyes=eye_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
    for(x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)    

    cv2.imshow('img',img)

    k=cv2.waitKey(30) & 0xff
    if k==27:
        break 
cap.release()



# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# img = cv2.imread('frd.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, , 3)


# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
#     # roi_gray = gray[y:y+h, x:x+w]
#     # roi_color = img[y:y+h, x:x+w]
#     # eyes = eye_cascade.detectMultiScale(roi_gray)
#     # for (ex,ey,ew,eh) in eyes:
#     #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()
