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
import time


# dismiss warning log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

fear_threshold = 0.00020
worry_threshold = 0.065
buffer_size = 2

def main():
    # load model
    model = models.load_model('./model/emo0809_epo50lr6.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    #detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./landmark/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=224)
    facecasc = cv2.CascadeClassifier('./harrcascade/haarcascade_frontalface_alt.xml')
    
    
    # emotion dict (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Worry"}
    # buffer init
    emotion_list = [0] * len(emotion_dict)
    prev_emotion = ''
    # start webcam 0: 內建camera, 1: 外接webcam
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = facecasc.detectMultiScale(rgb, scaleFactor=1.1 , minNeighbors=5)

        for (x, y, w, h) in faces:

            start = time.time()
            rect = dlib.rectangle(x,y,x+w+10,y+h+15)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)    
            # extract the ROI of the *original* face, then align the face
            # alignment
            faceAligned = fa.align(frame, rgb, rect)

            #no alignment use roi_rgb
            #roi_rgb = rgb[y:y + h+5  , x:x + w]
            cropped_img = cv2.resize(faceAligned, (224, 224))

            img_tensor = image.img_to_array(cropped_img)
            # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            prediction = model.predict(img_tensor)
            maxindex = int(np.argmax(prediction))

            pre = model.predict_proba(img_tensor)
            end = time.time()
            seconds = end - start
            print( "Time taken : {0} seconds".format(seconds))
            print('angry:{0:.15f}, disguested:{1:.15f}, fearful:{2:.15f}, \nhappy:{3:.15f}, neutral:{4:.15f}, sad:{5:.15f}, \nsurprised:{6:.15f}'.format(pre[0][0], 
                        pre[0][1], pre[0][2], pre[0][3], pre[0][4], pre[0][5], pre[0][6]))
            print('worry:{0:.15f}'.format(pre[0][7]))
            #pred proba
            print('Predict: {0}, Confidence: {1}'.format(str(emotion_dict[maxindex]), np.max(pre[0])))
            fps = 1 / seconds
            print( "Estimated frames per second : {0}".format(fps))
            print('=====================')

        
            if float(pre[0][7]) >= worry_threshold:
                cv2.putText(frame, emotion_dict[7], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #emotion_list[7] += 1
            # show fear but is worry
            elif maxindex==2 and float(pre[0][7]) >= 0.0050:
                cv2.putText(frame, emotion_dict[7], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #emotion_list[7] += 1
            else:
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #emotion_list[maxindex] += 1
            
            # index = np.argmax(emotion_list)
            # if emotion_list[index] >= buffer_size:
            #     emotion_list=[0] * len(emotion_dict)
            #     prev_emotion = emotion_dict[index]
            #     cv2.putText(frame, emotion_dict[index], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(frame, prev_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(800,600), interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()


