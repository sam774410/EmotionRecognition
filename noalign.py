# encoding: utf-8
import numpy as np
import argparse
import cv2
from keras.models import Sequential
from keras import models
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils


# dismiss warning log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fear_threshold = 0.01

def main():
    # load model
    model = models.load_model('./model/emo0809_epo50lr6.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("./landmark/shape_predictor_68_face_landmarks.dat")
    # fa = FaceAligner(predictor, desiredFaceWidth=224)
    facecasc = cv2.CascadeClassifier('./harrcascade/haarcascade_frontalface_alt.xml')
    
    
    # emotion dict (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Worry"}

    # start webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 20)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = facecasc.detectMultiScale(rgb, scaleFactor=1.1 , minNeighbors=5)

        #rects = detector(rgb, 2)
        for (x, y, w, h) in faces:
            
            #for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                #(x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=224)
                #faceAligned = fa.align(frame, rgb, rect)
                # display the output images

            roi_rgb = rgb[y:y + h+5  , x:x + w]
            cropped_img = cv2.resize(roi_rgb, (224, 224))

            img_tensor = image.img_to_array(cropped_img)
                
                # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            img_tensor /= 255.

            prediction = model.predict(img_tensor)
            maxindex = int(np.argmax(prediction))

            pro = model.predict_proba(img_tensor)
            print('angry:{0:.15f}, disguested:{1:.15f}, fearful:{2:.15f}, \nhappy:{3:.15f}, neutral:{4:.15f}, sad:{5:.15f}, \nsurprised:{6:.15f}'.format(pro[0][0], 
                    pro[0][1], pro[0][2], pro[0][3], pro[0][4], pro[0][5], pro[0][6]))
            print('worry:{0:.15f}'.format(pro[0][7]))
            #pred proba
            #print(pro[0][7])
            print('Predict: {0}, Confidence: {1}'.format(str(emotion_dict[maxindex]), np.max(pro[0])))
            print('=====================')

            #fear threshold
            # if float(pro[0][2]) >= fear_threshold:
            #     cv2.putText(frame, emotion_dict[2], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # else:
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(frame, str(maxindex), (x+20, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(800,600), interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()