from deepface import DeepFace
import cv2
import threading
import queue
from multiprocessing import Queue
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
def verifyFace():
    capture = VideoCapture(0)
    i = 0
    while(True):
        # ret,image = capture.read()
        image = capture.read()
        try:
            found = 0
            print("Comparing for frame " + str(i))
            with os.scandir("database/vishnu") as it:
                for entry in it:
                    if entry.name.endswith(".jpg") and entry.is_file():
                        result = DeepFace.verify(entry.path, image, detector_backend = 'opencv', model_name = "ArcFace")
                        if result['verified']:
                            found = 1
            if found:
                print("Succesful")
                return True
            else:
                print("Retrying.")

        except ValueError:
            print("Face not detected. Retrying...")
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
