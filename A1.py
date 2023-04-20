import cv2
import numpy as np

main_window_width = 640
main_window_height = 480
fps = 30

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, main_window_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, main_window_height)
cam.set(cv2.CAP_PROP_FPS, fps)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def trackbar1(val):
    global Hue1Low 
    Hue1Low = val

def trackbar2(val):
    global Hue1High 
    Hue1High = val

def trackbar3(val):
    global Hue2Low 
    Hue2Low = val

def trackbar4(val):
    global Hue2High 
    Hue2High = val

def trackbar5(val):
    global SatLow
    SatLow = val

def trackbar6(val):
    global SatHigh
    SatHigh = val

def trackbar7(val):
    global ValLow 
    ValLow = val

def trackbar8(val):
    global ValHigh
    ValHigh = val

Hue1High = 0
Hue1Low = 0
Hue2High = 0
Hue2Low = 0
SatLow = 0
SatHigh = 0
ValHigh = 0
ValLow = 0

trackbars_win_name = 'Track Bars'

cv2.namedWindow(trackbars_win_name)
cv2.resizeWindow(trackbars_win_name,int(main_window_width/2),main_window_height)
cv2.moveWindow(trackbars_win_name,main_window_width,0)

cv2.createTrackbar('Hue1 Low', trackbars_win_name,0,180, trackbar1)
cv2.createTrackbar('Hue1 High', trackbars_win_name,0,180, trackbar2)
cv2.createTrackbar('Hue2 Low', trackbars_win_name,0,180, trackbar3)
cv2.createTrackbar('Hue2 High', trackbars_win_name,0,180, trackbar4)
cv2.createTrackbar('Sat Low', trackbars_win_name,0,255, trackbar5)
cv2.createTrackbar('Sat High', trackbars_win_name,0,255, trackbar6)
cv2.createTrackbar('Val Low', trackbars_win_name,0,255, trackbar7)
cv2.createTrackbar('Val High', trackbars_win_name,0,255, trackbar8)

while True:
    ret, frame = cam.read()

    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsvLow1 = np.array([Hue1Low,SatLow,ValLow])
    hsvHigh1 = np.array([Hue1High,SatHigh,ValHigh])
    hsvLow2 = np.array([Hue2Low,SatLow,ValLow])
    hsvHigh2 = np.array([Hue2High,SatHigh,ValHigh])
    
    Mask1 = cv2.inRange(frameHsv,hsvLow1,hsvHigh1)
    Mask2 = cv2.inRange(frameHsv,hsvLow2,hsvHigh2)
    Mask = cv2.add(Mask1,Mask2)
    inverted_frame = cv2.bitwise_and(frame, frame, mask= Mask)

    contours, junk = cv2.findContours(Mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 700:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)

    Mask_small = cv2.resize(Mask,(int(main_window_width/2), int(main_window_height/2)))
    inverted_frame_small = cv2.resize(inverted_frame,(int(main_window_width/2), int(main_window_height/2)))

    cv2.imshow('@Asquare', frame)
    cv2.moveWindow('@Asquare', 0,0)

    cv2.imshow('Mask', Mask_small)
    cv2.moveWindow('Mask', 0,main_window_height)

    cv2.imshow('Inverted Frame', inverted_frame_small)
    cv2.moveWindow('Inverted Frame', int(main_window_width/2),main_window_height)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()