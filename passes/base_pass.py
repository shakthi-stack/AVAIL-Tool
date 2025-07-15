import time


class BasePass:
    def __init__(self):
        self.start_time = time.time()
        print("Initialize {0}".format(self.__class__.__name__))

    def execute(self):
        print("{0} executed at {1:10.4f} s".format(self.__class__.__name__, time.time() - self.start_time))
