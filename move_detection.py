import math
from multiprocessing.connection import wait
import cv2
import numpy as np
from PIL import ImageGrab
import sys, getopt

from utils import Utils

def prepare_frame(frame):
    new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    new_frame = cv2.GaussianBlur(src=new_frame, ksize=(5,5), sigmaX=0)
    return new_frame

def detect_motion(frame_pack, min_area):
    prev_f = frame_pack[0]
    f = frame_pack[1]

    prep_f = prepare_frame(f)

    diff_frame = cv2.absdiff(src1=prev_f, src2=prep_f)

    kernel = np.ones((11,11))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    thresh_frame = cv2.threshold(src=diff_frame, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]

    
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            # too small: skip!
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img=f, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    return prep_f, f


def capture_cam(cap, basic_fps, fps, min_area):
    frame_count = 0
    previous_frame = None

    fps2ms = int(1000/fps) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frames left")
            break 

        frame_count += 1

        if(frame_count % math.ceil(basic_fps/fps) != 0):
            continue

        if(previous_frame is None):
            previous_frame = prepare_frame(frame)
            continue
        
        previous_frame, detect_frame = detect_motion([previous_frame, frame], min_area)

        cv2.imshow('frame', detect_frame)

        if cv2.waitKey(fps2ms) == ord('q'):
            break

    
def main(argv):

    Utils.wait_info()

    fileName, isCam, fps, min_area = Utils.check_args(argv)

    basic_fps = 0

    if fileName != '':
        cap = cv2.VideoCapture(fileName, cv2.CAP_ANY)
        basic_fps = cap.get(cv2.CAP_PROP_FPS)
        if(not fps):
            fps = basic_fps
    elif isCam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        basic_fps = 25
        if(not fps or (fps and fps > 25)):
            fps = basic_fps
    else:
        print("no video source")
        sys.exit()

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 1280, 720)

    capture_cam(cap, basic_fps, fps, min_area)

    cap.release()
        
    Utils.exit_info()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])