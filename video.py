import cv2
from pymongo import MongoClient
from pprint import pprint
import dlib
import numpy as np
from imutils import face_utils


# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string, using pymongo!
client = MongoClient(port=27017)
# Creating the new db
db = client.drowsy_test


def driverprofile():
    driverprofile.names = input('Enter your name : ')
    driverprofile.regNo = input('Enter your reg no : ')
    # driverprofile.email = input('Enter your email address : ')
    driverprofile.phoneNumber = input('Enter your phone number: ')
    driverprofile.nextOfKin = input('Enter your next of kin number: ')

    # database schema
    details = {
        'name': driverprofile.names,
        'regNo': driverprofile.regNo,
        # 'email': driverprofile.email,
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


# Start the video stream
capture = cv2.VideoCapture(0)


eye_predictor_path = 'shape_predictor_68_face_landmarks.dat'
# Detector model
faceDetector = dlib.get_frontal_face_detector()
# predictor model
eye_predictor = dlib.shape_predictor(eye_predictor_path)


# Classes for exception handling...


class ManyFaces(Exception):
    pass


class NoFaceDetected(Exception):
    pass

# To get the eye landmarks from the video


def get_eyeLandmarks(vid):
    # Array that has detected faces
    rects = faceDetector(vid, 1)

    if len(rects) > 1:
        raise ManyFaces
    if len(rects) == 0:
        raise NoFaceDetected

    return numpy.matrix([[p.x, p.y] for p in eye_predictor(vid, rects[0]).parts()])

# Here we plot the numbers on the eyes


def plot_numberOnEyes(vid, landmarks):
    vid = vid.copy()
    for index, point in enumerate(landmarks):
        position = (point[0, 0], point[0, 1])
        cv2.putText(vid, str(index), position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(255, 255, 255))
        cv2.circle(vid, position, 3, color=(255, 255, 255))
# driverprofile()


while True:

    # capture frame by frame (returns true or false)
    ret, frame = capture.read()

    if ret == False:
        print('Camera Failed to start...')

    # operation on the frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for face in faces:
        # print(x, y, w, h)
        x1, y1, w, h = face
        x2 = x1 + w
        y2 = y1 + h
        # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None, /)
        img = cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 255, 239), 2, 4)
        # convert faces from np.array to dlib rectangle (For compatibility)
        faces_dlib = dlib.rectangle(x1, y1, x2, y2)
        # print(faces_dlib)
        landmarks = eye_predictor(gray, faces_dlib)
        # landmarks = face_utils.shape_to_np(landmarks)

        # for (lX, lY) in landmarks:
        #     cv2.circle(frame, (lX, lY), 1, (0, 0, 255), -1)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(gray, (x, y), 4, (36, 36, 246), -1)

        # ROI for the face so that eyes can be detected
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = gray[y1:y2, x1:x2]
        # detecting the eye..
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # iterate over all eyes found on face
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (x, y+h+30)
        # fontScale = 1
        # fontColor = (255, 255, 255)
        # lineType = 2

        # cv2.putText(img, driverprofile.names, (x, y+h+30),
        #             font, fontScale, fontColor, lineType)
        # cv2.putText(img, driverprofile.regNo, (x, y+h+60),
        #             font, fontScale, fontColor, lineType)
        # # cv2.putText(img, driverprofile.email, (x, y+h+90),
        # #             font, fontScale, fontColor, lineType)
        # cv2.putText(img, driverprofile.phoneNumber, (x, y+h+120),
        #             font, fontScale, fontColor, lineType)
        # cv2.putText(img, driverprofile.nextOfKin, (x, y+h+150),
        #             font, fontScale, fontColor, lineType)
    # landmarks = get_eyeLandmarks(capture)
    # video_with_landmarks = plot_numberOnEyes(capture, landmarks)
    # display resulting frame
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# when everything is dones, release the capture
capture.release()
cv2.destroyAllWindows()
