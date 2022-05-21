MAP_SIZE_PIXELS = 500
MAP_SIZE_METERS = 10
LIDAR_DEVICE = '/dev/ttyUSB0'
# Using 200 samples from RPLiDAR
MIN_SAMPLES = 200
from slam.algorithms import RMHC_SLAM
from slam.sensors import RPLidarA1 as LaserModel
from rplidar import RPLidar as Lidar
from roboviz import MapVisualizer

if __name__ == '__main__':
    lidar = Lidar(LIDAR_DEVICE) # Connect to Lidar
    slam = RMHC_SLAM(LaserModel(), MAP_SIZE_PIXELS, MAP_SIZE_METERS)
    viz = MapVisualizer(MAP_SIZE_PIXELS, MAP_SIZE_METERS, 'SLAM')
    
trajectory = []# Initialize an empty trajectory
mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
iterator = lidar.iter_scans()#Create an iterator to collect scan data from the RPLidar
previous_distances = None# store previous scan in case current scan is inadequate`
previous_angles = None
next(iterator)
while True:
    items = [item for item in next(iterator)]# Extract (quality, angle, distance) triples from current scan
    distances = [item[2] for item in items]# Extract distances and angles from triples
    angles = [item[1] for item in items]
     
    if len(distances) > MIN_SAMPLES:# Update SLAM with current Lidar scan
        slam.update(distances, scan_angles_degrees=angles)
        previous_distances = distances.copy()
        previous_angles = angles.copy()
    elif previous_distancesis not None:
        slam.update(previous_distances, scan_angles_degrees=previous_angles)
        x, y, theta = slam.getpos()# current robot position
        slam.getmap(mapbytes)
    if not viz.display(x/1000., y/1000., theta, mapbytes):
        exit(0)
    
lidar.stop()# Shutting down the lidar connection
lidar.disconnect()