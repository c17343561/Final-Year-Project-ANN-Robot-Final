import RPi.GPIO as GPIO
import time
#variables

delay = 0.005
steps = 500

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#Enable GPIO pins for ENA and ENB for stepper

enable_a = 18
enable_b = 22

#Enable pins for IN1-3 to control step sequence
coil_A_1_pin = 27 #black
coil_A_2_pin = 22 #green
coil_B_1_pin = 6 #red
coil_B_2_pin = 5 #blue

#Set pin states

GPIO.setup(enable_a, GPIO.OUT)
GPIO.setup(enable_b, GPIO.OUT)
GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)

#set ENA and ENB to high to enable stepper

GPIO.output(enable_a, True)
GPIO.output(enable_b, True)

#function for step sequence

def setStep(w1, w2, w3, w4):
     GPIO.output(coil_A_1_pin, w1)
     GPIO.output(coil_A_2_pin, w2)
     GPIO.output(coil_B_1_pin, w3)   
     GPIO.output(coil_B_2_pin, w4) 

#loop through step sequence based on number of steps

for i in range(0, steps):
     setStep(1,1,0,0)
     time.sleep(delay)
     setStep(0,1,1,0)
     time.sleep(delay)
     setStep(0,0,1,1)
     time.sleep(delay)
     setStep(1,0,0,1)
     time.sleep(delay)

#reverse previous step sequence to reverse motor direction

for i in range(0, steps):
     setStep(1,0,0,1)
     time.sleep(delay)
     setStep(0,0,1,1)
     time.sleep(delay)
     setStep(0,1,1,0)
     time.sleep(delay)
     setStep(1,1,0,0)
     time.sleep(delay)