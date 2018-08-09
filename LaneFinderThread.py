import numpy as np
import os, cv2
import time
from threading import Thread

class LaneFinder(Thread):
  
  def __init__(self, dimensions):

    self.width = dimensions[1]
    self.height = dimensions[0]
    self.imageCenter = self.width / 2
    self.overlay = np.zeros((self.height,self.width,3), np.uint8)
    self.numBoxes = 10
    self.boxWidth = 75
    self.boxHeight = int(self.height/self.numBoxes)
    
    self.ROI = np.zeros((4, 2), dtype = "float32")
    self.dist = np.zeros((4, 2), dtype = "float32")

    #Storage for individual images from lane finding process
    self.roiImage = None
    self.warp = None
    self.edges = None
    self.red = None
    self.green = None
    self.blue = None
    self.redThresh = None
    self.greenThresh = None
    self.blueThresh = None
    self.redEdges = None
    self.greenEdges = None
    self.blueEdges = None
    self.hLines = None
    self.hsv = None
    self.h = None
    self.s = None
    self.v = None
    self.hsvThresh = None
    self.hThresh = None
    self.sThresh = None
    self.vThresh = None
    self.currentImage = None
    self.colorMask = None
    self.sobelEdges = None
    self.rgbEdges = None
    self.canny = None
    self.laneImage = None

    self.previousLCurves = []
    self.previousRCurves = []
    self.previousLCurve = None
    self.previousRCurve = None

    #Approximate lane width values to start with
    self.laneWidth = 230
    self.laneWidthThresh = 400
    self.laneConfidenceThresh = 100
    self.laneAccuracyThresh = 25

    self.frames = []
    self.maxFrames = 2

    #Variables that can be controlled
    self.hValueY = 70
    self.sValueY = 80
    self.vValueY = 50
    self.hValueW = 20
    self.sValueW = 0
    self.vValueW = 195
    self.threshMin = 100

    #Stores current box positions and final curve of lane
    self.leftLine = None
    self.rightLine = None
    self.leftCurve = None
    self.rightCurve = None

    
    #Region Of Interest that is cut out of input image
    self.ROI = np.array([[int(self.width * 0.45), int(self.height * 0.55)],
                         [int(self.width * 0.55), int(self.height * 0.55)],
                         [int(self.width * 0.9), int(self.height * 0.75)],
                         [int(self.width * 0.1), int(self.height * 0.75)]], dtype = np.float32)

    #Distortion Coordinates
    self.dist[0] = [int(self.width * 0.3), 0 ]
    self.dist[1] = [int(self.width * 0.7), 0 ]
    self.dist[2] = [int(self.width * 0.7), self.height - 1]
    self.dist[3] = [int(self.width * 0.3), self.height - 1]

    #Matrix Transforms for warping image
    self.MWarp = cv2.getPerspectiveTransform(self.ROI, self.dist)
    self.MNormal = cv2.getPerspectiveTransform(self.dist, self.ROI)

    #Contrast
    self.contrastValue = 64
    self.brightness = 0
    
    #Edges
    self.threshold = (190, 255)
    self.yPositions = [(self.height - self.boxHeight) - (x * self.boxHeight) for x in range(self.numBoxes)]

    #Hough Lines
    self.minLineLength = 150
    self.maxLineGap = 30
    
    #Lane Positions
    self.leftEdge = 0
    self.rightEdge = 0
    self.laneCenter = 0

    #Images
    self.screen = None
    self.overlay = None
    
    self.frameTime = 0
    
    Thread.__init__(self)
    #self.daemon = True
    super(LaneFinder, self).__init__()
    
  def run(self):

    prevTime = time.time()
    
    while(True):
      
      foundLine = False
      lCurve = None
      rCurve = None
      
      if(self.screen is not None):
        foundLine, lCurve, rCurve = self.process(self.screen)

      if(foundLine):
          self.overlay = self.drawLanes(lCurve, rCurve)
                          
      self.frameTime = time.time() - prevTime
      prevTime = time.time()
      
  def process(self, image):
    laneImage = self.processImage(image)
    foundLine, lCurve, rCurve = self.findLanes(laneImage)
    return foundLine, lCurve, rCurve

  def processImage(self, image):
    self.currentImage = image
    roiImage = self.processROI(image, self.ROI)
    self.roiImage = roiImage
    
    contrast = self.contrast(roiImage, self.contrastValue)
    warp = cv2.warpPerspective(contrast, self.MWarp, (self.width, self.height))
    self.warp = warp
    blur = cv2.GaussianBlur(warp, (7, 7), 0)
    self.blur = blur
    self.hsv = cv2.cvtColor(self.blur, cv2.COLOR_RGB2HSV)

    #Color mask lower and upper limits
    yellowLower = np.array([self.hValueY, self.sValueY, self.vValueY])
    yellowUpper = np.array([120, 255, 255])
    whiteLower = np.array([self.hValueW, self.sValueW, self.vValueW])
    whiteUpper = np.array([255, 80, 255])

    #Create the mask
    yellowMask = cv2.inRange(self.hsv, yellowLower, yellowUpper)
    whiteMask = cv2.inRange(self.hsv, whiteLower, whiteUpper)

    #Apply the mask
    yellowColorMask = cv2.bitwise_and(self.blur, self.blur, mask = yellowMask)
    whiteColorMask = cv2.bitwise_and(self.blur, self.blur, mask = whiteMask)

    #Concatenate each mask to create a full image
    halfMask = np.concatenate((yellowColorMask[:, :400], whiteColorMask[:, 400:]), axis=1)
    halfMask = cv2.cvtColor(halfMask, cv2.COLOR_RGB2GRAY)
    self.colorMask = cv2.add(yellowColorMask, whiteColorMask)

    #Threshold mask
    maskGray = cv2.cvtColor(self.colorMask, cv2.COLOR_RGB2GRAY)
    _, maskThreshold = cv2.threshold(maskGray,self.threshMin,255,cv2.THRESH_BINARY)

    #RGB Edges
    rgbGray = cv2.cvtColor(self.blur, cv2.COLOR_RGB2GRAY)
    _, rgbGrayThresh = cv2.threshold(rgbGray,self.threshMin,255,cv2.THRESH_BINARY)
    rgbEdges = cv2.Canny(rgbGrayThresh,self.threshMin,200) 
    self.rgbEdges = cv2.cvtColor(rgbEdges, cv2.COLOR_GRAY2RGB)

    #Color Mask Edges
    colorMaskEdges = cv2.Sobel(maskThreshold,cv2.CV_64F,1,0,ksize=7).astype(np.uint8)
    
    self.edges = cv2.add(colorMaskEdges, maskThreshold)

    if(len(self.frames) >= self.maxFrames):
        self.frames.pop(0)

    self.frames.append(self.edges)

    #Add previous frames to try and fill in gaps with dashed lanes
    compoundImage = np.zeros_like(self.edges)
    for frame in self.frames:
      compoundImage = cv2.add(compoundImage, frame)

    laneImage = compoundImage
    self.laneImage = cv2.cvtColor(laneImage, cv2.COLOR_GRAY2RGB)
    
    return laneImage
    
  def findLanes(self, laneImage):
    #Find left lane base and right lane base
    leftBase, rightBase = self.findLaneStart(laneImage)
    self.leftBase = leftBase
    self.rightBase = rightBase
    self.laneCenter = abs(((leftBase - rightBase)) // 2) + leftBase

    #Find rest of lane from base
    lCurveFound, lCurve, lConfidence, lAccuracy, self.leftLine = self.calculateCurveLine(laneImage, leftBase)
    rCurveFound, rCurve, rConfidence, rAccuracy, self.rightLine = self.calculateCurveLine(laneImage, rightBase)

    self.leftCurve = lCurve
    self.rightCurve = rCurve
    
    lLaneUsable = False
    rLaneUsable = False

    #Trying to minimize errors...
    if(lCurveFound and lConfidence <= -self.laneConfidenceThresh and lAccuracy <= self.laneAccuracyThresh):
      lLaneUsable = True
    
    if(rCurveFound and rConfidence >= self.laneConfidenceThresh and rAccuracy <= self.laneAccuracyThresh):
      rLaneUsable = True

    if lLaneUsable:
      self.previousLCurve = lCurve
    elif lLaneUsable:
      lCurve = np.array([rCurve[:, 0] - self.laneWidth, lCurve[:, 1]])
      lCurve = np.transpose(lCurve)
      np.fliplr(lCurve)
                        
    if rLaneUsable:
      self.previousLCurve = lCurve
    elif lLaneUsable:
      rCurve = np.array([lCurve[:, 0] + self.laneWidth, lCurve[:, 1]])
      rCurve = np.transpose(rCurve)
      np.fliplr(rCurve)
    
    return lCurveFound and rCurveFound, lCurve, rCurve
    
  def findLaneStart(self, edges):
    histogram = np.sum(edges[edges.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftBase = np.argmax(histogram[:midpoint])
    rightBase = np.argmax(histogram[midpoint:]) + midpoint
          
    return (leftBase, rightBase)
  
  def calculateCurveLine(self, edges, base):
    line = []# Used to draw boxes
    pixels = []# Used to store all positions of white pixels
    XS = []
    YS = []

    boxWidth = self.boxWidth
    boxHeight = self.boxHeight
    
    for i in range(self.numBoxes):
      yPos = self.yPositions[i]

      #calculate average X position of white pixels
      if(i > 1):
        #Creats a crop and only keeps white pixel values
        crop = np.argwhere(edges[yPos:yPos + boxHeight, base-boxWidth:base+boxWidth] == 255)

        #If no white pixels
        if(len(crop) == 0):
          averagePos = line[i-1][0]
        else:
          averagePos = int(np.average(crop[:, 1])) + (base - boxWidth)# Get average X position of white pixels
          
        pos = [averagePos, yPos]
        line.append(pos)
        
        xs, ys = self.cropToList(crop, base - boxWidth, yPos)# Create a list of xs and ys of white pixels

        #Concatenate them with master white pixel positions
        XS += xs
        YS += ys
        
      else:
        #Runs on the first iteration
        pos = [base, yPos]
        line.append(pos)

        crop = np.argwhere(edges[yPos:yPos + boxHeight, base-boxWidth:base+boxWidth] == 255)
        
        xs, ys = self.cropToList(crop, base - boxWidth, yPos)

        XS += xs
        YS += ys
    
    try:
      pFit = np.polyfit(YS, XS, 2)# Creates polynomial from white pixels
      curve = self.getPolynomialCurve(pFit, self.width).astype(np.int32) # Creates coordinates of curve from polynomial

      #print("getPolynomialCurve() {}".format(time.time() - getPolynomialCurveStart))
    except:
        return False, None, None, None, None # If curve is found

    line = np.array(line) # Used for boxes
    confidence = base - self.imageCenter # Distance from center, sound be about 200
    accuracy = abs(np.average(line[:, 0]) - base) # How much curve differs from base
    
    return True, curve, confidence, accuracy, line
  
  def clipImage(self, image):
    1-(image == [255]).all(-1).ravel() #This makes clip image take 0.001 seconds as apose to 0.7
    
    return image

  def drawLanes(self, leftCurve, rightCurve):
    overlay = np.zeros((self.height,self.width,3), np.uint8)
    rightCurve = np.flipud(rightCurve)
    points = np.concatenate((leftCurve, rightCurve))
    cv2.fillPoly(overlay, [points], color=[0,255,0])
  
    overlay = cv2.warpPerspective(overlay, self.MNormal, (self.width, self.height))

    return overlay
  
  def drawLines(self, image, leftCurve, rightCurve):
    cv2.polylines(image,[leftCurve],False,(0,0,255))
    cv2.polylines(image,[rightCurve],False,(0,0,255))
    
  def getPolynomialCurve(self, poly, width, offset = 0):
    
    curve = []
    xCoords = np.arange(width)
    
    for i in range(width):
      x = xCoords[i]
      y = 0
      power = len(poly) - 1
      
      for j, b in enumerate(poly):
        y += b * (x ** power)
        power -= 1

      curve.append([y + offset, x])
      
    return np.array(curve)
  
  def cropToList(self, imgCrop, xOffset, yOffset):
    xs = []
    ys = []
    
    for i in range(len(imgCrop)):
      xs.append(imgCrop[i][1] + xOffset)
      ys.append(imgCrop[i][0] + yOffset)

    return (xs, ys)
  
  def contrast(self, image, contrastValue, brightness = 0):
    newImage = np.int16(image)  

    newImage = newImage*(contrastValue/127 + 1) - contrastValue + brightness
    newImage = np.clip(newImage, 0, 255)
    newImage = np.uint8(newImage)
    
    return newImage
  
  def processROI(self, img, roi):
    mask = np.zeros_like(img)
    roi = np.array(roi,dtype=np.int32)
    cv2.fillPoly(mask, [roi], (255,255, 255)).astype(np.int8)
    masked = cv2.bitwise_and(img, mask)
  
    return masked
  
  def drawBox(self, image, pos, boxWidth = 75, boxHeight = 50, lineWidth = 1):
    topRight = (pos[0] - boxWidth, pos[1] + boxHeight)
    bottomLeft = (pos[0] + boxWidth, pos[1])

    cv2.rectangle(image,topRight,bottomLeft,(0,255,0),lineWidth)
    
  def drawLaneBoxes(self):
    newImage = np.copy(self.laneImage)
    
    if(self.leftLine is None or self.rightLine is None):
      return newImage
    
    for i in range(len(self.leftLine)):
      lPos = self.leftLine[i]
      rPos = self.rightLine[i]

      self.drawBox(newImage, lPos, boxHeight = int(self.height / self.numBoxes))
      self.drawBox(newImage, rPos, boxHeight = int(self.height / self.numBoxes))

    return newImage

  def generateLaneVisualisations(self, firstImage):
    boxImage = self.drawLaneBoxes()
    self.drawLines(boxImage, self.leftCurve, self.rightCurve)
    edgeColor = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2RGB)
    
    topLayer = np.concatenate((firstImage, self.roiImage, self.warp),axis=1)
    bottomLayer = np.concatenate((edgeColor, self.laneImage, boxImage), axis=1)

    finalImage = np.concatenate((topLayer, bottomLayer), axis=0)
    finalImage = cv2.resize(finalImage, (0,0), fx=0.5, fy=0.5)

    return finalImage
