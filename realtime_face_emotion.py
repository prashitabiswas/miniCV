

import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

#capturing video from default camera
webcam_video_stream = cv2.VideoCapture(0)

#face expression model initialize 
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
#load weights into model
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#list of emotions label
emotions_label = ('angry', 'disgust','fear','happy','sad','surprise','neutral')
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
        
        #slicing
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
                
        #draw rectangle B,G,R and thickness
        cv2.rectangle(current_frame, (left_pos, top_pos),(right_pos, bottom_pos), (0,0,255),2)
         
        #preprocess input, convert image to data in dataset
        #convert to grescale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        #resize to 48x48
        current_face_image = cv2.resize(current_face_image,(48,48))
        #convert PIL image into 3d numpy array
        img_pixels = image.img_to_array(current_face_image)
        #expand the shape of an array into single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #pixels are in range of [0,255], normalize all pixels to [0,1]
        img_pixels/=255
              
        #do predictin using model, get the predicted values for 7 expressions
        exp_predictions = face_exp_model.predict(img_pixels)
        #find max indexed prediction value (0 till 7)
        max_index = np.argmax(exp_predictions[0])
        #get corresponding label from emotions_label
        emotion_label = emotions_label[max_index]
        
        #display label
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label,(left_pos,bottom_pos),font, 0.5, (255,255,255), (1))
        
        
    #showing current face with rectangle drawn
    cv2.imshow("Webcam video", current_frame)

    #wait for key press to break while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #release the webcam resource
webcam_video_stream.release()
cv2.destroyAllWindows()
    
    
    



