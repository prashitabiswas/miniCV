# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:53:06 2021

@author: rabid
"""

import cv2
import face_recognition

image_to_recognize_path = 'images/test/musk_test.jpg'

original_image = cv2.imread(image_to_recognize_path)


#load sample inages and get 128 embedding 
musk_image = face_recognition.load_image_file('images/sample/elon_musk.jpg')
musk_face_encodings = face_recognition.face_encodings(musk_image)[0]

bezos_image = face_recognition.load_image_file('images/sample/jeff_bezos.jpg')
bezos_face_encodings = face_recognition.face_encodings(bezos_image)[0]

prashita_image = face_recognition.load_image_file('images/sample/prashita.jpg')
prashita_face_encodings = face_recognition.face_encodings(prashita_image)[0]

known_face_encodings = [musk_face_encodings, bezos_face_encodings, prashita_face_encodings]
known_face_names = ['Elon Musk', 'Jeff Bezos', 'Prashita']

#load unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]

#find the distance of current encoding with all the known encodings
face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)

for i, face_distance in enumerate(face_distances):
    print("The face distance is {:.2} against the sample {}". format(face_distance, known_face_names[i]))
    print("The percentage match is {} against the sample {}". format(round((1-float(face_distance))*100,2), known_face_names[i]))
