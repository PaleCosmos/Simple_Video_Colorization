# Program To Read video 
# and Extract Frames 
import cv2
import numpy as np
import glob

directory = ""
count = 0

# Function to extract frames 
def FrameCapture(path): 
    global directory, count
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(directory + "frame/frame%d.jpg" % count, image) 
  
        count += 1

# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function
    FrameCapture('output.mp4')
