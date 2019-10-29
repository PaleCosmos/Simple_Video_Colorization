# Program To Read video 
# and Extract Frames 
import cv2
import numpy as np
import glob


# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        n += 1    
     
    out = cv2.VideoWriter('resources/frame/link_frame.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
        
    out.release()


# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function
    directory = 'resources/sample/'
    FrameCapture(directory + 'output.mp4')
    MakeVideo()
