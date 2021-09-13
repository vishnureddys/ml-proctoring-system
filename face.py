from deepface import DeepFace
import cv2
import threading
import queue
import multiprocessing
# from multiprocessing import Queue
import os

class VideoCapture:
    def __init__(self,name):
        self.capture=cv2.VideoCapture(name)
        self.q=queue.Queue()
        t=threading.Thread(target=self._reader)
        t.daemon=True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("No frames captured, exiting")
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release (self):
        self.capture.release()




# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

capture = VideoCapture(0)

i=1
while(True):
    # ret,image = capture.read()
    image = capture.read()
    try:
        found = 0
        print("Comparing for frame "+str(i))
        df = DeepFace.find(image, db_path = "database/vishnu")
        print(df)
        i+=1
        #cv2.imshow('frame',image)
        if i == 6:
            print("Face verified successfully")
            break

    except ValueError:
        print("Face not detected. Retrying...")


    # df = DeepFace.find(image,db_path="/database")
    # print(df)


    cv2.imshow('frame',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()
