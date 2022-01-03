# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:53:06 2021

@author: rabid
"""

import cv2
import face_recognition

image_to_detect = cv2.imread('images/test2.jpg')

all_face_locations = face_recognition.face_locations(image_to_detect, model = "hog")

print("there are {} faces in this image".format(len(all_face_locations)))

#looping through the face locations
for index, current_face_location in enumerate(all_face_locations):
    #splitting tuple to get four position values of face
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("found face {} at top:{}, right:{}, bottom:{}, left:{}".format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("face no: "+str(index), current_face_image)
