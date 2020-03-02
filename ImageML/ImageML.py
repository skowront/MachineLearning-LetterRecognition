#imports
import tensorflow as tf
import keras 
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import load_model
import numpy as np
import string
import json
import pandas as pd
import cv2
import glob
import imutils
from imutils import paths
import os
import os.path
import PIL
import PIL.Image

from numpy import asarray
from numpy import save
from numpy import load


fileList=os.listdir("Images")
#print(fileList)

fileNumber=1

filesCount=len(fileList)

#analyze images
try:
    #load X and Y sets from files
    CaptchaXtrain=load('NpArrays\\data0.npy',allow_pickle=True)
    CaptchaYtrain=load('NpArrays\\sign0.npy',allow_pickle=True)
except:
    #if no data and sign files exist, create them from captcha files
    for partNb in range(0,1):
        print(fileNumber)
        print(filesCount)
        CaptchaX_train=list([])
        CaptchaY_train=list([])
        for FILE in range(0,filesCount):
            captcha_image = fileList[FILE]
            # Load the image and convert it to grayscale
            image = cv2.imread('Images\\'+fileList[FILE])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # grab the base filename as the text
            filename = os.path.basename(captcha_image)
            captcha_text = os.path.splitext(filename)[0]

            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

            # apply threshold
            thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

            # create empty list for holding the coordinates of the letters
            letter_image_regions = []
 
            # find the contours
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
              # Get the rectangle that contains the contour
              (x, y, w, h) = cv2.boundingRect(contour)
        
              # check if any counter is too wide
              # if countour is too wide then there could be two letters joined together or are very close to each other
              if w / h > 1.25:
                # Split it in half into two letter regions
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
              else:  
                letter_image_regions.append((x, y, w, h))

            letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

            # Save each letter as a single image
            for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_text):
              # Grab the coordinates of the letter in the image
              x, y, w, h = letter_bounding_box

              # Extract the letter from the original image with a 2-pixel margin around the edge
              letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
      
              #convert to 28x28 and to float grey scaled from 0-1
              letter_image = cv2.resize(letter_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
              data=np.array(letter_image).astype(float)
              for i in range(data.shape[0]):
                  for j in range(data.shape[1]):
                      data[i][j] = 255-data[i][j]
                      data[i][j] = data[i][j]/255.0
              CaptchaX_train.append(data.tolist())
              CaptchaY_train.append(letter_text)
              print('File number: ' + str(fileNumber))
            fileNumber=fileNumber+1
        save('NpArrays\\data'+str(partNb)+'.npy', CaptchaX_train)
        del CaptchaX_train
        save('NpArrays\\sign'+str(partNb)+'.npy', CaptchaY_train)
        del CaptchaY_train

print(CaptchaYtrain)
temp=list()
#convert nbs and letters to ints to do this backwards use chr(int)
for iterator in range(0,len(CaptchaYtrain)):
    temp.append(ord(CaptchaYtrain[iterator]))
    
print(temp)
CaptchaYtrain=temp
print(CaptchaYtrain)
print(CaptchaXtrain)

#mnist = keras.datasets.mnist
#(x_train,y_train), (x_test,y_test) = mnist.load_data()



#x_train = keras.utils.normalize(x_train, axis=1)
#x_test = keras.utils.normalize(x_test, axis=1)
#print(x_train)
#print(y_train)
#y_train
#np.asarray(CaptchaYtrain)
##Read JSON data into the datastore variable
CaptchaY_train=keras.utils.normalize(CaptchaY_train,1)
try:
    #try to load trained model
    model = load_model('model.h5')
    print('ok')
except:
    #no trained model found, train a new one
    model = keras.models.Sequential() 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(120,activation='softmax'))
    model.compile('adam','sparse_categorical_crossentropy',['accuracy'])
    model.fit(CaptchaXtrain, np.asarray(CaptchaYtrain), epochs=50)

#evaluate
val_loss,val_acc=model.evaluate(CaptchaXtrain,CaptchaYtrain)
print(val_loss,val_acc)

# If the file name exists, write a JSON string into the file.
# Writing JSON data
#with open('architecture.json', 'w') as f:
    #datastore=model.to_json()
    #json.dump(datastore, f)

#save model
model.save('model.h5')

#np.set_printoptions(suppress=True,linewidth=np.nan)
#print(x_train[0])
#prediction=model.predict([[x_train[0]]])
#print(chr(np.argmax(prediction[0])))

#load new image
#for each number to be rated as a sign try to predict what it is
for count in range(1,9):
    #load image
    filename="ImageML\\number"+str(count)+".bmp"
    with PIL.Image.open(filename) as image:
        width, height=image.size

    try:  
        image= PIL.Image.open(filename)
    except IOError: 
        pass

    #some resizing
    out=PIL.Image.Image.convert(image,"P")
    image = image.resize((28,28),PIL.Image.BOX)


    data=np.array(out).astype(np.float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j]=data[i][j]/255.0
    #print(data);

    prediction=model.predict([[data]])
    print(str(count)+': '+chr(np.argmax(prediction[0])))

