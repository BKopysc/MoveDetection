import sys, getopt

class Utils:

    def check_mpi_size(size):
        if(size < 3):
            print("Too few processes. At least 3! (-n 3)")
            return False
        return True
        

    def check_args(argv):
        fileName = ''
        isCam = False
        fps = 50
        area = 300

        try:
            opts, args = getopt.getopt(argv, "h", ["fn=", "cam", "fps=", "area="])
        except getopt.GetoptError:
            print("mpiexec -n PROCESSES python move detect.py [--fn FILENAME] [--fps FRAMES PER SECOND] [--area MIN AREA]")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print("mpiexec -n PROCESSES python move detect.py [--fn FILENAME] [--fps FRAMES PER SECOND] [--area MIN AREA]")
                sys.exit()
            elif opt == "--fn":
                fileName = arg
            elif opt == "--fps":
                fps = int(arg)
            elif opt == "--area":
                area = int(arg)

        if (len(fileName) == 0):
            isCam = True
                
        return fileName, isCam, fps, area

    def wait_info():
        print("Initializing. Wait...")

    def exit_info():
        print("Program exited")

    def info():
        print("Usage: move_detection.py ")

