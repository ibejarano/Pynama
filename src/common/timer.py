from datetime import datetime

class Timer:
    def __init__(self):
        self.time = datetime.now()
    
    def getTime(self):
        oldTime = self.time
        self.time = datetime.now()
        return (self.time - oldTime).total_seconds()

    def tic(self):
        self.t = datetime.now()

    def toc(self):
        toc = datetime.now()
        elapsedTime = toc - self.t
        self.t = None
        return elapsedTime
