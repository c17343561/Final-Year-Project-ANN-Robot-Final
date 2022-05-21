import RPi.GPIO as GPIO
import time
#*****************************************************************************
        # Setting up pins 
#*****************************************************************************

GPIO.setmode(GPIO.BCM)
left_control_pins = [24,25,7,8]
right_control_pins = [27,22,6,5]
number_of_phases = 4
time_between_steps = 0.001 #seconds
step_angle = 5.625 #degrees
steps_revolution = int((360/step_angle)*number_of_phases)
RPM = steps_revolution/time_between_steps # revolutions per second
for left_pin in left_control_pins:
        print("Setting up Left Motor GPIO pins")
        GPIO.setup(left_pin,GPIO.OUT)
        GPIO.output(left_pin, False)
for right_pin in right_control_pins:
        print("Setting up Right Motor GPIO pins")
        GPIO.setup(right_pin,GPIO.OUT)
        GPIO.output(right_pin, False)
#*****************************************************************************
        # Half Step Sequence
#*****************************************************************************
def halfstep_antiClockwise(revolutions,control_pins):
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution

    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstepAnti_seq[halfstep][pin])
            time.sleep(0.001)
#-----------------------------------------------------------------------------          
def halfstep_clockwise(revolutions,control_pins):
    halfstep_seq = [
      [1,0,0,1],
      [0,0,0,1],
      [0,0,1,1],
      [0,0,1,0],
      [0,1,1,0],
      [0,1,0,0],
      [1,1,0,0],
      [1,0,0,0],
    ]

    #Half Step Motor configuration 1 Full revolution

    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
            time.sleep(0.001)
#*****************************************************************************
        # Full Step Sequence
#*****************************************************************************
def fullstep_antiClockwise(revolutions):
    fullstep_seq = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]


    for i in range(steps_revolution*revolutions):
        for fullstep in range(4):
            for pin in range(4):
                GPIO.output(right_control_pins[pin], fullstep_seq[fullstep][pin])
            time.sleep(0.005)
#-----------------------------------------------------------------------------
def fullstep_clockwise(revolutions):
    fullstepAnti_seq = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]
    for i in range(steps_revolution*revolutions):
        for fullstep in range(4):
            for pin in range(4):
                GPIO.output(right_control_pins[pin], fullstepAnti_seq[fullstep][pin])
            time.sleep(0.005)
#-----------------------------------------------------------------------------
def foward_motors(revolutions,left_control_pins,right_control_pins):
    halfstep_seq = [
    [1,0,0,1],
    [0,0,0,1],
    [0,0,1,1],
    [0,0,1,0],
    [0,1,1,0],
    [0,1,0,0],
    [1,1,0,0],
    [1,0,0,0],
    ]

    #Half Step Motor configuration 1 Full revolution

    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstep_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstep_seq[halfstep][pin])
            time.sleep(0.001)
#-----------------------------------------------------------------------------
def reverse_motors(revolutions,left_control_pins,right_control_pins):
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution

    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
            time.sleep(0.001)
#-----------------------------------------------------------------------------
def left_motors(revolutions,left_control_pins,right_control_pins):
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution
    reverse = [3,2,1,0]
    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
            time.sleep(0.001)
#-----------------------------------------------------------------------------
def right_motors(revolutions,left_control_pins,right_control_pins):
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution
    reverse = [3,2,1,0]
    for i in range(steps_revolution*revolutions):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
            time.sleep(0.001)
#-----------------------------------------------------------------------------
foward_motors(1,left_control_pins,right_control_pins)
#reverse_motors(1,left_control_pins,right_control_pins)
#left_motors(1,left_control_pins,right_control_pins)
#right_motors(1,left_control_pins,right_control_pins)

#fullstep_clockwise(1)
#fullstep_antiClockwise(1)

GPIO.cleanup()
