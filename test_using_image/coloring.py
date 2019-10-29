import cv2
import numpy as np

coloring_input = cv2.imread('output.jpg')
raw = coloring_input.copy()

coloring_input[np.where((coloring_input == [255,255,255]).all(axis = 2))] = [0,255,255] 

cv2.imshow('coloring_output', coloring_input)
cv2.imshow('coloring_input', raw)

cv2.imwrite("coloring_output.jpg", coloring_input)

'''
coloring_input = cv2.imread('coring_output.jpg')
rm_backgroud_input = cv2.cvtColor(coloring_input, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(rm_backgroud_input,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(coloring_nput)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite("test.png", dst)
'''
