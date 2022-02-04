import cv2
import numpy as np
from PIL import Image

#using video
CAMERA_1 = cv2.VideoCapture(0)
CAMERA_2 = cv2.VideoCapture(1)

for i in range(10):
    frame_1, image_1 = CAMERA_1.read()
    frame_2, image_2 = CAMERA_2.read()

    cv2.imwrite('opencv'+str(i)+'.png', image)
del(camera)


#using captured image
CAMERA_1 = Image.open("left_handcream.jpg")
CAMERA_2 = Image.open("right_handcream.jpg")
CAMERA_1_array = np.array(CAMERA_1)
CAMERA_2_array = np.array(CAMERA_2)




























from matplotlib.pyplot import thetagrids
from zmq import THREAD_AFFINITY_CPU_ADD
import numpy as np    
import matplotlib.pyplot as plt      
from rplidar import RPLidar


def get_data():    
    lidar = RPLidar('COM5', baudrate=115200)
    for scan in lidar.iter_scans(max_buf_meas=500):    
        break    
        lidar.stop()    
    return scan

for i in range(1000000):    
    if(i%7==0):    
        x = np.radians([])   
        y = []    
    print(i)    
    current_data=get_data()    
    for point in current_data:    
        if point[0]==15:    
            x.append(point[2]*np.sin(point[1]))    
            y.append(point[2]*np.cos(point[1]))    
    plt.clf()    
    plt.scatter(x, y)    
    plt.pause(.1)    
plt.show()
 

LiDAR_DATA = 

# get RGB data from camera using cv2 and np
CAMERA_RGB_DATA = 

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

# Read an image, a window and bind the function to window
image = cv2.imread("image.jpg")
cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

#Do until esc pressed
while(1):
    cv2.imshow('mouseRGB',image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
#if esc pressed, finish.
cv2.destroyAllWindows()



translation = np.zeros((3,1)) 
translation[:,0] = homography[:,2]

rotation = np.zeros((3,3))
rotation[:,0] = homography[:,0]
rotation[:,1] = homography[:,1]
rotation[:,2] = np.cross(homography[0:3,0],homography[0:3,1])

cameraMatrix = np.zeros((3,4))
cameraMatrix[:,0:3] = rotation
cameraMatrix[:,3] = homography[:,2]

cameraMatrix = cameraMatrix/cameraMatrix[2][3] #normalize the matrix



CAMERA_1 = 
LEFT_EYE = CAMERA_1 
RIGHT_EYE = CAMERA_2
THETA = ALPHA/2
distance = 57.3 * (b/2)/THETA  

# RESULT = GET Height of person and GET information of person 


