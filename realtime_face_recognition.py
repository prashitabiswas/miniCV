# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:03:39 2021

@author: rabid
"""

import cv2
import face_recognition

#capturing video from default camera
webcam_video_stream = cv2.VideoCapture(0)

#load sample inages and get 128 embedding 
musk_image = face_recognition.load_image_file('images/sample/elon_musk.jpg')
musk_face_enodings = face_recognition.face_encodings(musk_image)[0]

bezos_image = face_recognition.load_image_file('images/sample/jeff_bezos.jpg')
bezos_face_enodings = face_recognition.face_encodings(bezos_image)[0]

prashita_image = face_recognition.load_image_file('images/sample/prashita.jpg')
prashita_face_enodings = face_recognition.face_encodings(prashita_image)[0]

known_face_encodings = [musk_face_enodings, bezos_face_enodings, prashita_face_enodings]
known_face_names = ['Elon Musk', 'Jeff Bezos','Prashita Biswas']

#initialize empty array variable to hold all face locations
all_face_locations = []
all_face_encodings = []
all_face_names = []


#loop through every frame in video
while True:
    #get current frame
    ret, current_frame = webcam_video_stream.read()
    #resize current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx = 0.25, fy = 0.25)
    
    #find all face locations
    #model can be hog or cnn
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2, model = "hog")
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)
    all_face_names = []
    
    #looping through the face locations
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings): 
        #splitting tuple to get four position values of face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4

    
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        name_of_person = 'Unknown face'
        
        #check if all matches have atleast one item
        #if yes, get the index number of face that is located in the first index of all matches
        #get the name corresponding to the index numbebr and save it in name_of_person
        
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
            
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person,(left_pos,bottom_pos), font, 0.5, (0,255,0),1)
        
        
    #showing current face with rectangle drawn
    cv2.imshow("Webcam video", current_frame)

    #wait for key press to break while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    



            