import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

image_to_detect = cv2.imread('images/test2.jpg')
#face expression model initialize 
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
#load weights into model
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#list of emotions label
emotions_label = ('angry', 'disgust','fear','happy','sad','surprise','neutral')

all_face_locations = face_recognition.face_locations(image_to_detect, model = "hog")

print("there are {} faces in this image".format(len(all_face_locations)))

#looping through the face locations
for index, current_face_location in enumerate(all_face_locations):
    #splitting tuple to get four position values of face
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("found face {} at top:{}, right:{}, bottom:{}, left:{}".format(index+1, top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    
   #draw rectangle B,G,R and thickness
    cv2.rectangle(image_to_detect, (left_pos, top_pos),(right_pos, bottom_pos), (0,0,255),2)
     
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
    cv2.putText(image_to_detect, emotion_label,(left_pos,bottom_pos),font, 0.5, (255,255,255), (1))
    
    
#showing current face with rectangle drawn
cv2.imshow("image emotion", image_to_detect)
    
