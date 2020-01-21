'''
To capture a video, you need to create a VideoCapture object. 
Its argument can be either the device index or the name of a video file. 
Device index is just the number to specify which camera. 
Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). 
You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame. 
But at the end, don’t forget to release the capture.
'''
from cv2 import cv2


class FaceEyeClassifier:
    """Drowsy driver classifier."""

    def main():
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

    # display resulting frame
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is dones, release the capture
capture.release()
cv2.destroyAllWindows()

if __name__ == 'main':
    main()
