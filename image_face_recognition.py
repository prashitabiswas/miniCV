# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:53:06 2021

@author: rabid
"""

import cv2
import face_recognition

original_image = cv2.imread('images/test/test2.jpg')


#load sample inages and get 128 embedding 
musk_image = face_recognition.load_image_file('images/sample/elon_musk.jpg')
musk_face_enodings = face_recognition.face_encodings(musk_image)[0]

bezos_image = face_recognition.load_image_file('images/sample/jeff_bezos.jpg')
bezos_face_enodings = face_recognition.face_encodings(bezos_image)[0]

known_face_encodings = [musk_face_enodings, bezos_face_enodings]
known_face_names = ['Elon Musk', 'Jeff Bezos']


image_to_recognize = face_recognition.load_image_file('images/test/test2.jpg')

all_face_locations = face_recognition.face_locations(image_to_recognize, model = "hog")
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

#looping through the face locations and embeddding
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    #splitting tuple to get four position values of face
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    
    
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    name_of_person = 'Unknown face'
    
    #check if all matches have atleast one item
    #if yes, get the index number of face that is located in the first index of all matches
    #get the name corresponding to the index numbebr and save it in name_of_person
    
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
        
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person,(left_pos,bottom_pos), font, 0.5, (0,255,0),1)
    
    cv2.imshow("faces indentified", original_image)
 