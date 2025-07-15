import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class Unbuffered:
    def __init__(self, stream, filename):
        self.stream = stream
        self.te = open(filename, 'w')  # File where you need to keep the logs

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.te.write(data)    # Write the data of stdout here to a text file as well
        self.te.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass





