'''
# Code written by: Rafal Wolk
# Date of last update: 21/05/2022
# Description: The code allows the user to control the robot remotely using a VNC
# connection and keyboard input through the laptop as long as both devices are in 
# the same Wi-Fi connection
# Purpose: Part of DT021A Year 4 Final Year Project
'''
import pygame # Pygame module
import os # Operating system module
import RPi.GPIO as GPIO # GPIO module
import time # Time module

def getKey(keyName): # This function reads the key pressed on the keyboard
    ans = False # Set the ans Boolean to False
    for eve in pygame.event.get():pass # Read the event
    keyInput = pygame.key.get_pressed() # Get the key input 
    myKey = getattr(pygame,'K_{}'.format(keyName)) # figure out which key has been pressed
    if keyInput [myKey]: # If the key pressed is one of the registered keys
        ans = True	# Ans is True
    pygame.display.update() # Update the display
    
    return ans # Return the answer
GPIO.setmode(GPIO.BCM)
left_control_pins = [24,25,7,8] # Pins that the left stepper motor is connected to
right_control_pins = [27,22,6,5] # Pins that the right stepper motor is connected to
number_of_phases = 4 # motors have 4 phases
time_between_steps = 0.0006 # Timeout between steps
step_angle = 5.625 # The angle of each individual step
steps_revolution = int((360/step_angle)*number_of_phases) # Number of steps needed for 1 revolution
RPM = steps_revolution/time_between_steps # revolutions per second
for left_pin in left_control_pins: # Setting up left pins
        print("Setting up Left Motor GPIO pins")
        GPIO.setup(left_pin,GPIO.OUT) # Setting them up as outputs
        GPIO.output(left_pin, False)
for right_pin in right_control_pins: # Setting up right pins
        print("Setting up Right Motor GPIO pins")
        GPIO.setup(right_pin,GPIO.OUT) # Setting them up as outputs
        GPIO.output(right_pin, False)

def left_motors(left_control_pins,right_control_pins): # Turning left
    # Half step sequence
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
    while getKey('a'): # If letter ‘a’ is pressed on the keyboard turn robot left
        for halfstep in range(8): # Going through the sequence
            for pin in range(4): # Writing to pins
                GPIO.output(left_control_pins[pin], halfstep_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstep_seq[halfstep][pin])
                time.sleep(time_between_steps) # Timeout between steps
#-----------------------------------------------------------------------------
def right_motors(left_control_pins,right_control_pins): # Turning right
    # Half step sequence
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
    while getKey('d'): # If letter ‘d’ is pressed on the keyboard turn robot right
        for halfstep in range(8): # Going through the sequence
            for pin in range(4): # Writing to pins
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps) # Timeout between steps
#-----------------------------------------------------------------------------
def forward_motors(left_control_pins,right_control_pins): # Going forward
    # Half step sequence
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
    reverse = [3,2,1,0] # Reverse the sequence
    while getKey('w'): # If letter ‘w’ is pressed on the keyboard make robot go forward
        for halfstep in range(8): # Going through the sequence
            for pin in range(4): # Writing to pins
                GPIO.output(left_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps) # Timeout between steps
#-----------------------------------------------------------------------------
def reverse_motors(left_control_pins,right_control_pins): # Going backward
    # Half step sequence
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
    reverse = [3,2,1,0] # Reverse the sequence
    while getKey('s'): # If letter ‘s’ is pressed on the keyboard make robot go backward
        for halfstep in range(8): # Going through the sequence
            for pin in range(4): # Writing to pins
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps) # Timeout between steps
#-----------------------------------------------------------------------------
def init(): # Initialising the pygame instance
    pygame.init() # Initialise an instance
    os.environ["DISPLAY"] = ":0" # Create a display
    pygame.display.init() # Initialise the display
    win = pygame.display.set_mode((100,100)) # Set dimensions of the display
def main(): # Main 
    if getKey('w'): # If ‘w’ key is pressed
        forward_motors(left_control_pins,right_control_pins) # Call function motors forward
        print('Forward function is selected') # Print to terminal
    if getKey('s'): # If ‘s’ key is pressed
        reverse_motors(left_control_pins,right_control_pins) # Call function motors backward
        print('Backward function is selected') # Print to terminal
    if getKey('a'): # If ‘a’ key is pressed
        left_motors(left_control_pins,right_control_pins) # Call function motors left
        print('Left function is selected') # Print to terminal
    if getKey('d'): # If ‘d’ key is pressed
        right_motors(left_control_pins,right_control_pins) # Call function motors right
        print('Right function is selected') # Print to terminal
    if getKey('q'): # If ‘1’ key is pressed
        print('Quitting....') # Print to terminal
        pygame.quit() # Terminate the instance
        exit() # Exit the terminal

if __name__ == '__main__': # If main
    init() # Initialise function
    while True: # Continuous while loop
            main() # Main
