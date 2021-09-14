import os
import random
from datetime import datetime
import threading
import cv2
import numpy as np
from person_and_phone import YoloV3, load_darknet_weights, draw_outputs
from mouth_tracking_helper import MouthOpenDetector
from eye_tracking_helper import EyeTracker
from save_image_to_log import save_image_log
import time
from head_pose_tracker import HeadPoseDetector
from face_verification import verifyFace

# WAIT = random.randint(3, 10)
WAIT = 5

while(True):
    if verifyFace():
        print("Face Verified. Proceeding")
        break
    else:
        print("Trying again. ")

yolo = YoloV3()
load_darknet_weights(yolo, 'models/yolov3.weights') 

cap = cv2.VideoCapture(0)

mouth = MouthOpenDetector()
eyes = EyeTracker()
headpose = HeadPoseDetector(cap)



def detect_person_and_phone():
    while(True):
        try:
            ret, image = cap.read()
            if ret == False:
                break
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 320))
            img = img.astype(np.float32)
            img = np.expand_dims(img, 0)
            img = img / 255
            class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
            boxes, scores, classes, nums = yolo(img)
            count=0
            for i in range(nums[0]):
                if int(classes[0][i] == 0):
                    count +=1
                if int(classes[0][i] == 67):
                    print('Mobile Phone detected')
                    cheatingtype = "MOBILE_PHONE"
                    save_image_log(image,datetime.now(), cheatingtype)
            if count == 0:
                print('No person detected')
                cheatingtype = "NO_PERSON"
                save_image_log(image,datetime.now(), cheatingtype)
            elif count > 1: 
                print('More than one person detected')
                cheatingtype = "MULTIPLE_FACES"
                save_image_log(image,datetime.now(), cheatingtype)


            image = draw_outputs(image, (boxes, scores, classes, nums), class_names)
            cv2.imshow('Prediction', image)
            # time.sleep(WAIT)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass
    


# os.system("python audio_part.py")




person_phone = threading.Thread(target = detect_person_and_phone)
# detect_person_and_phone()

mouth_track = threading.Thread(target=mouth.detect_mouth, args=(cap,))
# mouth.detect_mouth(cap)


track_eyes = threading.Thread(target = eyes.track_eye, args=(cap,))
# eyes.track_eye(cap)

track_heads = threading.Thread(target = headpose.track_head, args=(cap,))


mouth.get_mask(cap)

person_phone.start()
mouth_track.start()
track_eyes.start()
track_heads.start()


person_phone.join()
mouth_track.join()
track_eyes.join()
track_heads.join()

cap.release()
cv2.destroyAllWindows()
