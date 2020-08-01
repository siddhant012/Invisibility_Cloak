#scroll down to the main function to make changes
import numpy as np
import time
import cv2
from tkinter import Scale,Tk,mainloop,HORIZONTAL,Button,Label


#utility functions
def load_image(save_name):
    image=cv2.imread(save_name)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.uint8)
    return image

def load_numpy(save_name):
    with open(save_name,'rb') as file : array=np.load(file)
    return array

def save_image(image,save_name):
    image=cv2.cvtColor(cv2.UMat(image),cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(save_name,image) : raise Exception("Could not write image")

def save_numpy(array,save_name):
    with open(save_name,'wb') as file : np.save(file,array)


#helper functions
def get_background(shape,timer=5):

    cam=cv2.VideoCapture(0)
    if(not cam.isOpened()) : raise IOError("Cannot open webcam")

    for i in range(1,timer+1) : print("recording background in ",timer+1-i,end="\r\r");time.sleep(1.0)
    _,background=cam.read()
    background=cv2.resize(background,shape)
    cv2.imshow('Background Image',background)
    background=cv2.cvtColor(background,cv2.COLOR_BGR2RGB).astype(np.uint8)
    print("\nbackground_image recorded")
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    cam.release()

    return background

def get_new_frame(frame1,frame2,boundaries,kernel_size=(3,3)):
    frame1=cv2.cvtColor(frame1,cv2.COLOR_RGB2HSV)
    frame2=cv2.cvtColor(frame2,cv2.COLOR_RGB2HSV)

    mask=cv2.inRange(frame1,lowerb=boundaries[0],upperb=boundaries[1])
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones(kernel_size,np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones(kernel_size,np.uint8))
    mask=np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)

    np.putmask(frame1,mask,frame2)
    return cv2.cvtColor(frame1,cv2.COLOR_HSV2RGB)

def get_boundaries(shape):

    print("Tweak the HSV sliders to decide the minimum and maximum values of the color to be identified.")
    print("Press esc to save and exit.")

    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    master=Tk()
    lmin=Label(master,text="minimum")
    hmin=Scale(master,from_=0,to=179,orient=HORIZONTAL,label="hue")
    smin=Scale(master,from_=0,to=255,orient=HORIZONTAL,label="saturation")
    vmin=Scale(master,from_=0,to=255,orient=HORIZONTAL,label="value")
    lmax=Label(master,text="maximum")
    hmax=Scale(master,from_=0,to=179,orient=HORIZONTAL,label="hue")
    smax=Scale(master,from_=0,to=255,orient=HORIZONTAL,label="saturation")
    vmax=Scale(master,from_=0,to=255,orient=HORIZONTAL,label="value")

    lmin.pack()
    hmin.pack()
    smin.pack()
    vmin.pack()
    lmax.pack()
    hmax.pack()
    smax.pack()
    vmax.pack()

    while(1):
        _,frame=cam.read()
        frame=cv2.resize(frame,shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)

        curr_boundaries=np.array([[hmin.get(),smin.get(),vmin.get()],[hmax.get(),smax.get(),vmax.get()]])
        frame=get_new_frame(frame,np.zeros(frame.shape,dtype=frame.dtype),curr_boundaries)

        if(cv2.waitKey(1)==27) : break
        frame=cv2.cvtColor(cv2.UMat(frame),cv2.COLOR_RGB2BGR)
        cv2.imshow('WebCam',frame)
        master.update()
    
    print("Values stored.")

    cam.release()
    cv2.destroyAllWindows()
    master.destroy()
    return curr_boundaries




def main():

    #You only need to specify these 4 values
    shape=(500,500)     #shape of the webcam window
    timer=3             #timer before the webcam opens
    kernel_size=(3,3)   #kernel size in the morphological operation.Can be set to (3,3) or (5,5).
    path=""             #specify the path of the directory where this file is stored



    #background image and boundaries array name . Do not change this.
    background_path=path+"background.jpg" 
    boundaries_path=path+"boundaries.npy"


    #comment this code block after recording and saveing the color boundaries and background from webcam successfully.  
    boundaries=get_boundaries(shape)
    background=get_background(shape,timer)
    save_image(background,background_path)
    save_numpy(boundaries,boundaries_path)


    #load the recorded background and boundaries array
    background=load_image(background_path)
    boundaries=load_numpy(boundaries_path)


    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("Press esc to exit.")
    for i in range(1,timer+1) : print("starting in ",timer+1-i,end="\r\r");time.sleep(1.0)

    while True:
        _,frame=cam.read()
        frame=cv2.resize(frame,shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)

        frame=get_new_frame(frame,background,boundaries,kernel_size)

        frame=cv2.cvtColor(cv2.UMat(frame),cv2.COLOR_RGB2BGR)
        cv2.imshow('WebCam',frame)
        if(cv2.waitKey(1)==27) : break

    cam.release()
    cv2.destroyAllWindows()

if(__name__=="__main__") : main()