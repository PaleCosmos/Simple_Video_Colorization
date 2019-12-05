from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys

main = Tk()
main.title('SVC')

cvFrame = Frame(main)
cvFrame.grid(row =0, column = 0, padx = 10, pady = 10)

lbl1 = Label(cvFrame)
lbl1.grid(row = 0, column = 0)

def ExitButton():
    sys.exit()
# 색 가져옴
def getColor():
    cvt2Colors = np.array(askcolor()[0])
    print(cvt2Colors)

btn = Button(cvFrame, text = "Exit", command=ExitButton)
btn.grid(row =1, column = 0, columnspan = 2, sticky= N + S + W+ E)

cap = cv2.VideoCapture('video/sampleVideo.mp4')
# 크기설정
cap.set(3, 480)
cap.set(4, 320)

frame_size = (int(cap.get(3)), int(cap.get(4)))

def show_frame():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)

    lbl1.imgtk = imgtk
    lbl1.configure(image =imgtk)

    main.after(10, show_frame)

show_frame()
main.mainloop()