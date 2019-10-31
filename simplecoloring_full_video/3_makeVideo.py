# Program To Read video 
# and Extract Frames 
import cv2
import numpy as np
import glob

directory = ""
count = 45
    
# Function to make video
def MakeVideo():
    global directory, count;
    img_array = []
    n = 0; size = (0, 0); width = 0; height = 0
    
    for n in range(count):
        if n == 0:
            continue
        #for filename in glob.glob(directory + 'add_output/add_output%d.jpg' % n):
        img = cv2.imread('add_output/add_output%d.png' % n)
        #cv2.imshow("img", img)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

        n += 1    

    out = cv2.VideoWriter('link_frame.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()


# Driver Code 
if __name__ == '__main__': 
    MakeVideo()
