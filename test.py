from multiprocessing.connection import wait
import cv2
import numpy as np
from PIL import ImageGrab
import sys, getopt

def checkArgs(argv):
    fileName = ''
    isCam = True
    fs = 5

    try:
        opts, args = getopt.getopt(argv, "h", ["fn=", "cam", "fs="])
    except getopt.GetoptError:
        print("mpiexec -n PROCESSES python move detect.py {--fn FILENAME | --cam} [--fs FRAMES PER SECOND]")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("mpiexec -n PROCESSES python move detect.py {--fn FILENAME | --cam} [--fs FRAMES PER SECOND]")
            sys.exit()
        elif opt == "--fn":
            fileName = arg
        elif opt == "--cam":
            isCam = True
        elif opt == "--fs":
            fs = arg

    return fileName, isCam, fs

def captureCam(cap):
    # while True:
    #     ret, frame = cap.read()
    #     #frame = cv2.cvtColor(src=frame, code=cv2.COLOR_XYZ2RGB)
    #     cv2.imshow('webcam', frame)

    #     if (cv2.waitKey(1) == 27):
    #         break
        
    # # press escape to exit

    # cap.release()
    # cv2.destroyAllWindows()

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('frame', frame)

        while True:
            if cv2.waitKey(0) == ord('n'):
                break
        
        if cv2.waitKey(25) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main(argv):
    fileName, isCam, fs = checkArgs(argv)

    if fileName != '':
        cap = cv2.VideoCapture(fileName)
    elif isCam:
        cap = cv2.VideoCapture(0)
    else:
        print("no video source")
        sys.exit()

    captureCam(cap)


if __name__ == "__main__":
   main(sys.argv[1:])