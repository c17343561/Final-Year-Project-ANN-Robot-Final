from threading import Thread
import time
finalDistances = []

def MotorControl():

    global finalDistances
    finalDistances = [45,234,234,2342,34,234,234,3,24,23,4,23,42,35,34,645,656,7,67,867,8]
    for i in range(len(finalDistances)):
        print("From thread 1")
        print(finalDistances)
        
def DataControl():
    global finalDistances
    for i in range(len(finalDistances)):
        print("From thread 2")
        print(finalDistances)


motorControlThread = Thread(target = MotorControl())

dataThread = Thread(target = DataControl())


motorControlThread.start()
dataThread.start()


