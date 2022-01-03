# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:31:08 2021

@author: rabid
"""

import face_recognition
from PIL import Image, ImageDraw

#load image as numpy array
face_image = face_recognition.load_image_file("images/test/test2.jpg")

#finad all facial landmarks of all faces in image
face_landmarks_list = face_recognition.face_landmarks(face_image)
print(len(face_landmarks_list))

index  = 0


#convert numpy array to PIL image and create a Draw object
pil_image = Image.fromarray(face_image)
d = ImageDraw.Draw(pil_image)
        
while index <len(face_landmarks_list):
        
    #iterate through all landmarks
    for face_landmark in face_landmarks_list:
        

        
        #draw white line connecting landmarks
        d.line(face_landmark['chin'], fill = (255,255,255) , width = 2)
        d.line(face_landmark['left_eyebrow'], fill = (255,255,255), width = 2)
        d.line(face_landmark['right_eyebrow'], fill = (255,255,255), width = 2)
        d.line(face_landmark['nose_bridge'], fill = (255,255,255), width = 2)
        d.line(face_landmark['nose_tip'], fill = (255,255,255), width = 2)
        d.line(face_landmark['left_eye'], fill = (255,255,255), width = 2)
        d.line(face_landmark['right_eye'], fill = (255,255,255), width = 2)
        d.line(face_landmark['top_lip'], fill = (255,255,255), width = 2)
        d.line(face_landmark['bottom_lip'], fill = (255,255,255), width = 2)
    
    index += 1
    
pil_image.show()

#save the image
pil_image.save("images/sample/prashi_landmarks.jpg")

