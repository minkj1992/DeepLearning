import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint
import sys
import os

new_path='/home/minkj1992/anaconda3/envs/opencv/share/OpenCV/haarcascades/'
FACE_CASCADE = cv2.CascadeClassifier(new_path+'haarcascade_frontalface_default.xml')
 
def detect_faces(image_path,cnt,path):
    image_grey = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    
    for x,y,w,h in faces:
        sub_img=image_grey[y-10:y+h+10,x-10:x+w+10]
        # 저장할때 gray
        cv2.imwrite(path+'Extracted/'+str(cnt)+".jpg",sub_img)

targets = ['crime/', 'athlete/', 'ceo/', 'professor/' ,'celebrity/']
path = '/home/minkj1992/code/facial_project/cnn-keras/origin_data/'   

for target in targets:
    cnt = 0 
    for i in [i for i in os.listdir(path+target)]:
        detect_faces(path+target+i,cnt,path+target)
        cnt+=1
    print(target+"done")

