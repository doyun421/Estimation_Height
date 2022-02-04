import cv2
import numpy as np
from PIL import Image

#using video
CAMERA_1 = cv2.VideoCapture(0)
CAMERA_2 = cv2.VideoCapture(1)

while(CAMERA_1.isOpened()):
    ret, frame = CAMERA_1.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break

    # Save Frame by Frame into disk using imwrite method
    cv2.imwrite('Frame' + str(i)+ '.jpg', frame)
    i += 1

CAMERA_1.release()
cv2.destroyAllWindows()




for i in range(10):
    frame_1, image_1 = CAMERA_1.read()
    frame_2, image_2 = CAMERA_2.read()

    cv2.imwrite('opencv'+str(i)+'.png', image)
del(camera)

#using captured image
CAMERA_1 = Image.open("left_handcream.jpg")
CAMERA_2 = Image.open("right_handcream.jpg")
CAMERA_1_array = np.array(CAMERA_1)
def degreeToRadians(float degree):

  return degree * (PI / 180.0);


def sphericalToCartesian(float sphericalPoints[], float cartesianPoints[]:

  float pan = sphericalPoints[0];
  float tilt = sphericalPoints[1];
  float distance = sphericalPoints[2];

  float x = distance * cos(degreeToRadians(pan)) * cos(degreeToRadians(tilt)); //x value
  float y = distance * cos(degreeToRadians(tilt)) * sin(degreeToRadians(pan)); //y value
  float z = distance * sin(degreeToRadians(tilt));                             //z value

  cartesianPoints[0] = x;
  cartesianPoints[1] = y;
  cartesianPoints[2] = z;

#camera 2개, Lidar sensor 1개, person 1명, Kiosk 1대, Height measure, 
"""
We took the body's measurements of the participants manually using a measuring tape to measure their
shoulder, bust, waist, hip, and length with centimeter as the measurement unit.





그리고 라이다 센서 장착은  

"""

camera_1 = getvideo()
camera_1.array()
features_camera_1 = camera_1.calculated_features

lidar_dataset = get.lidar_yml()


if distance_between_features_body_toe == 0:
    img_camera_1_part_as_toe = "toe"
    img_camera_1_part_as_head = "head"

    distance_to_toe = lidar.read()
    distance_to_head = lidar.read()
    distance_vertical_to_nearby_belly = lidar.read()

    length_upper_body = sqrt((distance_to_head).^2 - (distance_vertical_to_nearby_belly).^2)
    length_lower_body = sqrt((distance_to_toe).^2 - (distance_vertical_to_nearby_belly).^2)

    height_between_toe_and_head = length_upper_body + length_lower_body

if distance_between_features_face_bottom == 0:
    img_camera_1_part_as_face_bottom = "chin"
    img_camera_1_part_as_face_head = "head"

    distance_btw_face_bottom = lidar.read()
    distance_btw_face_head = lidar.read()
    distance_vertical_btw_nearby_nose = lidar.read()

    length_upper_face = sqrt((distance_btw_face_head).^2 - (distance_vertical_btw_nearby_belly).^2)
    length_under_face = sqrt((distance_btw_face_bottom).^2 - (distance_vertical_btw_nearby_belly).^2)

    size_of_face_between_bottom_head = length_upper_body - length_lower_body

height = height_between_toe_and_head
face_length = size_of_face_between_bottom_head

# to show the face information using lidar sensor




float calculatePointDistance(float coordinatesOne[], float coordinatesTwo[])
{

  float x1 = coordinatesOne[0];
  float y1 = coordinatesOne[1];
  float z1 = coordinatesOne[2];
  float x2 = coordinatesTwo[0];
  float y2 = coordinatesTwo[1];
  float z2 = coordinatesTwo[2];

  float distance = sqrt(sq(x2 - x1) + sq(y2 - y1) + sq(z2 - z1));

  return distance;
}





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


