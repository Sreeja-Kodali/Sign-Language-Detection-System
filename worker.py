import os, cv2, pickle, numpy as np
from rq import Connection, Worker
import mediapipe as mp

listen = ['default']

if __name__ == '__main__':
    with Connection():
        worker = Worker(list(map(str, listen)))
        worker.work()
