import numpy as np
import os, cv2
import time
from grabscreen import grab_screen
from LaneFinderThread import LaneFinder
import time, math

def drawROI(image):
    for pos in lf.ROI:
        cv2.circle(image, (pos[0], pos[1]), 10, (0,0,255), -1)
        
def checkClickedControl(x,y,controlPos):
    lStart = controlPos[0]
    lEnd = controlPos[1]

    rStart = controlPos[2]
    rEnd = controlPos[3]
    
    lStartDistance = math.sqrt(((lStart[0] - x) ** 2) + (((lStart[1] - y) ** 2)))
    lEndDistance = math.sqrt(((lEnd[0] - x) ** 2) + (((lEnd[1] - y) ** 2)))

    rStartDistance = math.sqrt(((rStart[0] - x) ** 2) + (((rStart[1] - y) ** 2)))
    rEndDistance = math.sqrt(((rEnd[0] - x) ** 2) + (((rEnd[1] - y) ** 2)))
    
    if(lStartDistance < 10):
        return 1
    elif(lEndDistance < 10):
        return 2
    elif(rStartDistance < 10):
        return 3
    elif(rEndDistance < 10):
        return 4
    else:
        return None

def mouseCallback(event, x, y, flags, params):
    
    global leftDown, rightDown, controlHit

    if(x > 800):
        x = 800
    elif (x < 0):
        x = 0
        
    if(y > 450):
        y = 450
    elif (y < 0):
        y = 0

    if(event == cv2.EVENT_LBUTTONDOWN):

        controlHit = checkClickedControl(x, y, lf.ROI)
        print(controlHit)
        leftDown = True
    elif(event == cv2.EVENT_LBUTTONUP):
        leftDown = False
        
    if event == cv2.EVENT_MOUSEMOVE:
        if(leftDown):
            if(controlHit == 1):
                lf.ROI[0] = [x, y]
            elif(controlHit == 2):
                lf.ROI[1] = [x, y]
            elif(controlHit == 3):
                lf.ROI[2] = [x, y]
            elif(controlHit == 4):
                lf.ROI[3] = [x, y]

            lf.MWarp = cv2.getPerspectiveTransform(lf.ROI, lf.dist)
            lf.MNormal = cv2.getPerspectiveTransform(lf.dist, lf.ROI)
            
def onchange(x):
    pass

SCREEN_REGION = ((1920//2) - (1280//2), (1080//2) - (720//2), (1920//2) + (1280//2), (1080//2) + (720//2))

width = SCREEN_REGION[2] - SCREEN_REGION[0] + 1
height = SCREEN_REGION[3] - SCREEN_REGION[1] + 1
newSize = (800, 450)

#Lane finder thread
lf = LaneFinder((newSize[1], newSize[0]))
lf.start()

#Control Parameters
leftDown = False
rightDown = False
controlHit = -1

#Lane finding parameters
cv2.namedWindow('Lane')
cv2.namedWindow('Settings')
cv2.setMouseCallback('Lane', mouseCallback)
cv2.createTrackbar('HY','Settings',0,255,onchange)
cv2.createTrackbar('SY','Settings',0,255,onchange)
cv2.createTrackbar('VY','Settings',0,255,onchange)
cv2.createTrackbar('HW','Settings',0,255,onchange)
cv2.createTrackbar('SW','Settings',0,255,onchange)
cv2.createTrackbar('VW','Settings',0,255,onchange)
cv2.createTrackbar('Threshold','Settings',0,255,onchange)

cv2.setTrackbarPos('HY','Settings',lf.hValueY)
cv2.setTrackbarPos('SY','Settings',lf.sValueY)
cv2.setTrackbarPos('VY','Settings',lf.vValueY)
cv2.setTrackbarPos('HW','Settings',lf.hValueW)
cv2.setTrackbarPos('SW','Settings',lf.sValueW)
cv2.setTrackbarPos('VW','Settings',lf.vValueW)
cv2.setTrackbarPos('Threshold', 'Settings', lf.threshMin)

cv2.resizeWindow('Settings', 1000,200)

while(True):

    #Check for keyboard interrupt
    try:
        screen = grab_screen(((1920//2) - (1280//2), (1080//2) - (720//2), (1920//2) + (1280//2), (1080//2) + (720//2)))
        screen = cv2.resize(screen, (800, 450))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        #Set lane finding parameters
        lf.screen = screen
        lf.hValueY = cv2.getTrackbarPos('HY','Settings')
        lf.sValueY = cv2.getTrackbarPos('SY','Settings')
        lf.vValueY = cv2.getTrackbarPos('VY','Settings')
        lf.hValueW = cv2.getTrackbarPos('HW','Settings')
        lf.sValueW = cv2.getTrackbarPos('SW','Settings')
        lf.vValueW = cv2.getTrackbarPos('VW','Settings')
        lf.threshMin = cv2.getTrackbarPos('Threshold','Settings') 

        #If lanes have been found show them
        if lf.screen is not None and lf.overlay is not None:
            screen = cv2.add(lf.screen, lf.overlay) # Ad found lane overlay to original Image
            drawROI(screen) #Draw Reigion of interest to screen
            laneVisuals = lf.generateLaneVisualisations(screen)
            cv2.imshow("Lane Finding Process", laneVisuals) #Uncomment this to see visuals
            cv2.imshow("Lane", screen)
            cv2.waitKey(2) 
            
    except KeyboardInterrupt:
        print("Closing")
        break
    
cv2.destroyAllWindows()
