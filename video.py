'''
To capture a video, you need to create a VideoCapture object. 
Its argument can be either the device index or the name of a video file. 
Device index is just the number to specify which camera. 
Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). 
You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame. 
But at the end, don’t forget to release the capture.
'''
from cv2 import cv2
from pymongo import MongoClient
from pprint import pprint
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string, using pymongo!
client = MongoClient(port=27017)
# Creating the new db
db = client.drowsy_test


def driverprofile():
    driverprofile.names = input('Enter your name : ')
    driverprofile.regNo = input('Enter your reg no : ')
    driverprofile.email = input('Enter your email address : ')
    driverprofile.phoneNumber = input('Enter your phone number: ')
    driverprofile.nextOfKin = input('Enter your next of kin number: ')

    details = {
        'name': driverprofile.names,
        'regNo': driverprofile.regNo,
        'email': driverprofile.email,
        'phoneNumber': driverprofile.phoneNumber,
        'nextOfKin': driverprofile.nextOfKin
    }
    result = db.driverdetails.insert(details)
    print(f'finished inserting {result} details into the db')


# load XML classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load eye classifier
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
# check if classifier is loaded for error handling..
loaded = cv2.CascadeClassifier.empty(face_cascade)
print(loaded)
if loaded == True:
    print('You need to load the classifier')


capture = cv2.VideoCapture(0)
# names = input('Enter your name : ')
# regNo = input('Enter your reg no : ')
# email = input('Enter your email address : ')
# phoneNumber = input('Enter your phone number: ')
# nextOfKin = input('Enter your next of kin number: ')
driverprofile()

while True:

    # capture frame by frame (returns true or false)
    ret, frame = capture.read()

    # operation on the frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None, /)
        img = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 239), 2, 4)
        # ROI for the face so that eyes can be detected
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        # detecting the eye..
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # iterate over all eyes found on face
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (x, y+h+30)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, driverprofile.names, (x, y+h+30),
                    font, fontScale, fontColor, lineType)
        cv2.putText(img, driverprofile.regNo, (x, y+h+60),
                    font, fontScale, fontColor, lineType)
        cv2.putText(img, driverprofile.email, (x, y+h+90),
                    font, fontScale, fontColor, lineType)
        cv2.putText(img, driverprofile.phoneNumber, (x, y+h+120),
                    font, fontScale, fontColor, lineType)
        cv2.putText(img, driverprofile.nextOfKin, (x, y+h+150),
                    font, fontScale, fontColor, lineType)

    # display resulting frame
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is dones, release the capture
capture.release()
cv2.destroyAllWindows()
