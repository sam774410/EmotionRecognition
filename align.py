import cv2
import dlib
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./landmark/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=224)
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("/Users/mingshenglyu/Desktop/EmotionRecognition/happy.jpg")
image = imutils.resize(image, width=1200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# show the original input image and detect faces in the grayscale
# image
rects = detector(gray, 2)
i = 0
# loop over the face detections
for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=224)
        faceAligned = fa.align(image, gray, rect)
        # display the output images
        cv2.imwrite("org-"+str(i)+".jpg", faceOrig)
        cv2.imwrite("ali-"+str(i)+".jpg", faceAligned)
        i += 1
