#Test_2.py trying to get a video stream to work with the model.

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import sys


#*********************************
#code from the Learnable machine exported code (sets up model)
#*********************************

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
path = "converted_keras/keras_model.h5"
model = tensorflow.keras.models.load_model(path)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#*********************************
# Set up video stream object
#*********************************
video = cv2.VideoCapture(0)
a = 0

#Create prediction array
predic = [["HP_Correct", "HP_turn_left", "HP_turn_right", "HP_too_far"],[0,0,0,0]]
pos = 0     #pos is the index of the highest rated prediction.

while True:
    a = a + 1
    # *********************************
    # Set up video stream object
    # *********************************
    #create a video frame
    check, frame = video.read()
    #output the video reading
    #print(check)
    #print(frame)
    #conver to greyScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #show frame
    cv2.imshow("Capturing", gray)

    # *********************************
    # Press 'q' to stop video transmission and end program
    # *********************************
    #cv2.waitKey(0)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # *********************************
    # save video as image (also image for the model)
    # *********************************
    cv2.imwrite("photos/useImg.jpg", frame)
    image_path = "photos/useImg.jpg"
    image = Image.open(image_path)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # *********************************
    # run the model
    # *********************************
    prediction = model.predict(data)

    # place predictions into predic array
    for i in range(4):
        storVal = round(prediction[0][i] * 100, 2)
        predic[1][i] = (prediction[0][i])
        #check if new best prediction.
        if predic[1][i] > predic[1][pos]:
            pos = i
    # output the position instructions
    if pos == 0:
        print("Please stay still")
    elif pos == 1:
        print("Please turn your head to the LEFT")
    elif pos == 2:
        print("Please turn your head to the RIGHT")
    elif pos == 3:
        print("Please get closer!")
    print("{}: {}".format(predic[0][pos], predic[1][pos]))

print("video lasted {} ms".format(a))

video.release()


