# Program To Read video 
# and Extract Frames 
import cv2
import numpy as np
import glob

directory = "frame/"
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
        cv2.imwrite(directory + "frame%d.jpg" % count, image) 
  
        count += 1
    
# Function to make video
def MakeVideo():
    global directory, count;
    img_array = []
    n = 0
    for n in range(count):
        for filename in glob.glob(directory + 'frame%d.jpg' % n):
            img = cv2.imread(filename)
            
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        n += 1    

    out = cv2.VideoWriter(directory + 'link_frame.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
        
    out.release()


# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function
    FrameCapture('output.mp4')
    MakeVideo()
