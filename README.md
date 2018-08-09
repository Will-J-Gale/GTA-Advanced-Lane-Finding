# GTA5-Lane-Finding
Interactive lane finding algorithm that runs in real time at about 10-15fps

## Found lane
![alt text](https://github.com/Will-J-Gale/GTA-Lane-Finding/blob/master/Images/Lane%20Overlay%20Half%20Size.gif)  

## Visualisation of lane finding process
![alt text](https://github.com/Will-J-Gale/GTA-Lane-Finding/blob/master/Images/Visualisation%20HALF%20SIZE.gif)  

### Algorithm Steps:
1. Cut out reigion of interest
2. Warp image so lanes are parallel
3. Threshold and find edges of image
4. Add previous images to try and fill in gaps of dashes lane lines
5. Create histogram of images to find base of left and right lane
6. Use boxes to find white pixels from base and average the X position
7. Use all white pixels found in boxes to create a polynomial fit
8. Use polynomial fit to create curved lanes
9. Warp image back to original shape and overlay on original image

## Comments on algorithm
While this algorithm can find lanes, it is not very robust.  
The example above shows that the algorithm works well on straight, high contrast roads.  
However, as soon as curves appear the algorithm breaks down.  
Moreover, when the colour of the road changes the algorithm struggles.  
Tweaking of parameters can combat this, however the algorithm cannot figure this out on its own

## Prerequisites 
1. GTA 5
2. Python 3.6
3. OpenCV
4. Numpy

## Usage
Recommended to use on dual monitors
1. Run GTA5 in windowed mode 1280x720
2. Run GTA_LaneFinding.py
3. Three Windows will appear
   * __Lane:__ 
      * This window shows Region of interest which can be moved by clicking and moving the red spheres
   * __Lane Finding Process:__ 
      * This windows shows some of the process of finding the lanes
   * __Settings:__
      * This window allows control of some settings used to find the lanes such as image threshold and HSV thresholds
   

