# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:03:39 2021

@author: rabid
"""

import cv2
import face_recognition

#capturing video from default camera
webcam_video_stream = cv2.VideoCapture(0)

#initialize empty array variable to hold all face locations
all_face_locations = []

#loop through every frame in video
while True:
    #get current frame
    ret, current_frame = webcam_video_stream.read()
    #resize current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx = 0.25, fy = 0.25)
    
    #find all face locations
    #model can be hog or cnn
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2, model = "hog")
    
    #looping through the face locations
    for index, current_face_location in enumerate(all_face_locations): 
        #splitting tuple to get four position values of face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4

        print("found face {} at top:{}, right:{}, bottom:{}, left:{}".format(index+1, top_pos,right_pos,bottom_pos,left_pos))
        #draw rectangle B,G,R and thickness
        cv2.rectangle(current_frame, (left_pos, top_pos),(right_pos, bottom_pos), (0,0,255),2)
    #showing current face with rectangle drawn
    cv2.imshow("Webcam video", current_frame)

    #wait for key press to break while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    



            