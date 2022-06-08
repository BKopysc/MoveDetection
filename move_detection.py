from fileinput import close
import math
from multiprocessing.connection import wait
from time import sleep
import cv2
import numpy as np
from mpi4py import MPI
from PIL import ImageGrab
import sys, getopt

from utils import Utils

def prepare_frame(frame):
    new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    new_frame = cv2.GaussianBlur(src=new_frame, ksize=(5,5), sigmaX=0)
    return new_frame

def detect_motion(pack, min_area):
    prev_f = pack[0]
    f = pack[1]
    id = pack[2]

    prep_prev_f = prepare_frame(prev_f)
    prep_f = prepare_frame(f)

    diff_frame = cv2.absdiff(src1=prep_prev_f, src2=prep_f)

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

    return [id, f]

def render_frame(comm, fps):
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 800, 600)
    frame_buffer = []
    fps2ms = int(1000/int(fps))
    frame_counter = 0
    close_flag = False

    # task_count = [0] * comm.Get_size()

    # for i in range(0, 2):
    #     task_count[i] = 2000000

    while(True):
        # recv_task_count = comm.irecv(source = 0, tag = 12)
        # recv_task_count.wait()
        # if recv_task_count != None:
        #     task_count = recv_task_count

        status = MPI.Status() 
        #print(status.Get_tag())
        recv_data = comm.recv(source = MPI.ANY_SOURCE, tag=11)
        # recv_data = recv_data_req.wait()        
        if (recv_data != None):
            frame_buffer.append(recv_data)
            # task_count[i] -= 1
        
        # req = comm.isend(task_count, dest=0, tag=12)
        # req.wait()
        
        check_frames = [frame_buffer[i] for i in range(len(frame_buffer)) if frame_buffer[i][0] == frame_counter]

                

        if len(check_frames) > 0:    

            frame_to_view = check_frames[0]
            cv2.imshow('frame', frame_to_view[1])
            frame_buffer.remove(frame_to_view)
            frame_counter += 1

            if cv2.waitKey(fps2ms) == ord('q'):
                break

    cv2.destroyAllWindows()


def capture_cam(comm, cap, basic_fps, fps):
    frame_count = 0
    previous_frame = None
    # fps2ms = int(1000/int(fps))

    # frames_packs = []
    # packs_count= 0
    pack_id = 0

    dest_counter = 2

    # task_count = [0] * comm.Get_size()


    # for i in range(0, 2):
    #     task_count[i] = 2000000

    first_loop = True

    while True:

        ret, frame = cap.read()
        if not ret:
            print("No frames left")
            break 

        frame_count += 1

        if(frame_count % math.ceil(basic_fps/fps) != 0):
            continue

        if(previous_frame is None):
            previous_frame = frame
            continue
        # recv_task_count = comm.irecv(source = 1, tag = 12)
        # recv_task_count.wait()
        # if recv_task_count != None:
        #     task_count = recv_task_count  

        # destination = 2
        # min_tasks = task_count[2]

        # for i in range(3, comm.Get_size()):
        #     if task_count[i] < min_tasks:
        #         min_tasks = task_count[i]
        #         destination = i


        comm.send([previous_frame, frame, pack_id], dest=dest_counter, tag=10)
        pack_id += 1
        # task_count[destination] += 1
        dest_counter += 1
        if dest_counter == comm.Get_size():
            dest_counter = 2

        previous_frame = frame

        if(first_loop):
            first_loop = False
            comm.send(True, dest=1, tag=21)

        
        status = MPI.Status
        if(status.tag == 404):
            flag_req = comm.recv(source=1, tag=404)
            if(flag_req):
                break


        # req = comm.isend(task_count, dest=1, tag=12)
        # req.wait()

        # frames_packs.append([previous_frame, frame, pack_id])
        # pack_id += 1
        
        # for i in range(0, comm_size):
        #     packs_to_send = frames_packs[0]
        #     frames_packs.pop[0]
        #     comm.isend(packs_to_send)
            
        
        # previous_frame, detect_frame = detect_motion([previous_frame, frame], min_area)

        # cv2.imshow('frame', detect_frame)

        # if cv2.waitKey(fps2ms) == ord('q'):
        #     break

    
def main(argv):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if(not Utils.check_mpi_size(size)):
        sys.exit()

    fileName, isCam, fps, min_area = Utils.check_args(argv)

    print("Program started")

    #print(fileName, isCam, fps, min_area)

    if(rank == 0):
        print("start rank: ", rank)
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
            if(rank == 0):
                print("no video source")
                sys.exit()
        
        Utils.wait_info()

        comm.send(fps, dest=1, tag=20)

        capture_cam(comm, cap, basic_fps, fps)
        cap.release()

        Utils.exit_info()

    elif(rank == 1):
        print("start rank: ", rank)
        fps = comm.recv(source=0, tag=20)

        # task_count = [0] * comm.Get_size()
        # for i in range(0, 2):
        #     task_count[i] = 2000000
        # comm.send(task_count, dest=0, tag=12)   
        
        should_start = comm.recv(source=0, tag=21)
        render_frame(comm, fps)

        comm.send(True, dest = 0, tag = 404)

        for i in range(2,comm.Get_size()):
            comm.send(True, dest = i, tag = 404)


    else:
        close_flag = False

        while(True):
            pack = comm.recv(source = 0, tag = 10)
            processed_pack = detect_motion(pack, min_area)
            req = comm.send(processed_pack, dest=1, tag=11)

            status = MPI.Status
            if(status.tag == 404):
                flag_req = comm.recv(source=1, tag=404)
                if(flag_req):
                    break
    
    MPI.Finalize()
    exit(0)



if __name__ == "__main__":
   main(sys.argv[1:])