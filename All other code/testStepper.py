import RPistepper as stp
M1_pins = [24,25,7,8]
M2_pins = [27,22,6,5]
with stp.Motor(M1_pins) as M1:
    for i in range(1000):               # moves 20 steps,release and wait
        print(M1)
        M1.move(1)
        M1.release()
        