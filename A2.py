import cv2
import numpy as np
from face_landmarks import facelm

main_window_width = 640
main_window_height = 480
fps = 30

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, main_window_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, main_window_height)
cam.set(cv2.CAP_PROP_FPS, fps)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# colors = {
#     'Orange':[4,11,0,255,0,255],  'Green': [49,70,0,255,0,255], 'White': [74,91,0,34,255,255], 'Blue': [95,108,95,255,0,255], 'Red': [162,180,84,255,105,255]
# }
colors = {
    'Orange':[4,20,86,255,0,255],  'Green': [68,77,38,255,0,255], 'White': [82,121,0,142,208,255], 'Blue': [89,106,103,255,0,255], 'Red': [165,180,116,255,0,255], 'Yellow ': [21,45,78,255,0,255]
}

facelandm = facelm(main_window_width,main_window_height)

while True:
    ret, frame = cam.read()
    face_lm_color = (127,127,127) #gold_colour = (0,215,255)
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for color in colors:
        # color = 'Orange'
        hsv = np.zeros((1,1,3), dtype= np.uint8)
        hsvLow1 = np.array([colors[color][0],colors[color][2],colors[color][4]])
        hsvHigh1 = np.array([colors[color][1],colors[color][3],colors[color][5]]) 
        hsv[0,0] = (int((colors[color][0]+colors[color][1])/2),180,255)
        hsv = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        if color == 'White':
            colour = (255,255,255)
        else:
            colour = (int(hsv[0,0][0]),int(hsv[0,0][1]),int(hsv[0,0][2]))

        # hsvLow2 = np.array([Hue2Low,SatLow,ValLow])
        # hsvHigh2 = np.array([Hue2High,SatHigh,ValHigh])

        Mask1 = cv2.inRange(frameHsv,hsvLow1,hsvHigh1)

        # Mask2 = cv2.inRange(frameHsv,hsvLow2,hsvHigh2)
        # Mask = cv2.add(Mask1,Mask2)
        # inverted_frame = cv2.bitwise_and(frame, frame, mask= Mask1)

        contours, junk = cv2.findContours(Mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 700:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.putText(frame,color,(x,y-3),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),colour,2)
                face_lm_color = colour

    # Mask_small = cv2.resize(Mask1,(int(main_window_width/2), int(main_window_height/2)))
    # inverted_frame_small = cv2.resize(inverted_frame,(int(main_window_width/2), int(main_window_height/2)))

    facelandm.landmarks(frame,face_lm_color)

    cv2.imshow('@Asquare', frame)
    cv2.moveWindow('@Asquare', 0,0)

    # cv2.imshow('Mask', Mask_small)
    # cv2.moveWindow('Mask', 0,main_window_height)

    # cv2.imshow('Inverted Frame', inverted_frame_small)
    # cv2.moveWindow('Inverted Frame', int(main_window_width/2),main_window_height)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()