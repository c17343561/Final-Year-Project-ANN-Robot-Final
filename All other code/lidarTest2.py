from pyrplidar import PyRPlidar
from pyrplidar_protocol import PyRPlidarResponse
from pyrplidar_serial import PyRPlidarSerial

#lidar = PyRPlidar()
#lidar.connect(port="/dev/ttyUSB0", baudrate=115200, timeout=3)
#response = PyRPlidarResponse()

serial = PyRPlidarSerial()
serial.open("/dev/ttyUSB0", 115200, 3)
serial.receive_data(8)

#serial.__init__()
#response.__init__()
#response.__str__()

'''
info = lidar.get_info()
print("info :", info)

health = lidar.get_health()
print("health :", health)

samplerate = lidar.get_samplerate()
print("samplerate :", samplerate)


scan_modes = lidar.get_scan_modes()
print("scan modes :")
for scan_mode in scan_modes:
    print(scan_mode)
'''
#start_scan = lidar.start_scan()
#print("start_scan :", start_scan)

#receive_discriptor = lidar.receive_discriptor()
#print("receive_discriptor :", receive_discriptor)


#lidar.disconnect()
