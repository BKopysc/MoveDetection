import sys, getopt

class Utils:

    def check_args(argv):
        fileName = ''
        isCam = True
        fps = 50
        area = 300

        try:
            opts, args = getopt.getopt(argv, "h", ["fn=", "cam", "fps=", "area="])
        except getopt.GetoptError:
            print("mpiexec -n PROCESSES python move detect.py {--fn FILENAME | --cam} [--fps FRAMES PER SECOND] [--area MIN AREA]")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print("mpiexec -n PROCESSES python move detect.py {--fn FILENAME | --cam} [--fps FRAMES PER SECOND] [--area MIN AREA]")
                sys.exit()
            elif opt == "--fn":
                fileName = arg
            elif opt == "--cam":
                isCam = True
            elif opt == "--fps":
                fps = arg
            elif opt == "--area":
                area = arg
                
        return fileName, isCam, fps, area

    def wait_info():
        print("Initializing. Wait...")

    def exit_info():
        print("Program exited")

    def info():
        print("Usage: move_detection.py ")

