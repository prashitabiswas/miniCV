# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:31:08 2021

@author: rabid
"""

import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

#capturing video from default camera
webcam_video_stream = cv2.VideoCapture(0)

#initialize empty array variable to hold all face locations
all_face_locations = []

#loop through every frame in video
while True:
    #get current frame
    ret, current_frame = webcam_video_stream.read()
 
        
    #finad all facial landmarks of all faces in image
    face_landmarks_list = face_recognition.face_landmarks(current_frame)
#    print(len(face_landmarks_list))
    
    index  = 0
    
    
    #convert numpy array to PIL image and create a Draw object
    pil_image = Image.fromarray(current_frame)
    d = ImageDraw.Draw(pil_image,'RGBA')
            
    while index < len(face_landmarks_list):
            
        #iterate through all landmarks
        for face_landmark in face_landmarks_list:
            
    
            
            #draw white line connecting landmarks
            #    d.line(face_landmark['chin'], fill = (25,0,25,50) , width = 5)
            #d.polygon(face_landmark['left_eyebrow'], fill = (39,54,68, 128))
            #d.polygon(face_landmark['right_eyebrow'], fill = (39,54,68, 128))
            d.line(face_landmark['left_eyebrow'], fill = (0,0,0, 150), width = 5)
            d.line(face_landmark['right_eyebrow'], fill = (0,0,0, 150), width = 5)
            d.line(face_landmark['nose_bridge'], fill = (255,255,255,20), width = 6)
        #    d.line(face_landmark['nose_tip'], fill = (255,255,255), width = 2)
            d.polygon(face_landmark['left_eye'], fill = (255,0,0, 100))
            d.polygon(face_landmark['right_eye'], fill = (255,0,0, 100))
            d.line(face_landmark['left_eye']+[face_landmark['left_eye'][0]], fill = (0,0,0,70), width = 5)
            d.line(face_landmark['right_eye']+[face_landmark['right_eye'][0]], fill = (0,0,0,70), width = 5)
            d.polygon(face_landmark['top_lip'], fill = (0,0,150, 128))
            d.polygon(face_landmark['bottom_lip'], fill = (0,0,150, 128))
            d.line(face_landmark['top_lip'], fill = (0,0,150,64), width = 8)
            d.line(face_landmark['bottom_lip'], fill = (0,0,150,64), width = 8)

        index += 1
        
        
    #convert PIL image to RGB to show in opencv window
    rgb_image = pil_image.convert('RGB')    
    rgb_open_cv_image = np.array(pil_image)
    
    #convert RGB to BGR
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr_open_cv_image = bgr_open_cv_image[:,:,::-1].copy()
    
    
    #showing current face with rectangle drawn
    cv2.imshow("Webcam video", bgr_open_cv_image)

    #wait for key press to break while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
