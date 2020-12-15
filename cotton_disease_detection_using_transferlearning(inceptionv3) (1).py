# -*- coding: utf-8 -*-
"""Cotton_Disease_Detection_Using TransferLearning(InceptionV3)
"""

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

##Now we resize image 
IMAGE_SIZE = [224, 224]
train_path = "/content/drive/My Drive/Cotton_disease/Cotton_Disease/train"
test_path = "/content/drive/My Drive/Cotton_disease/Cotton_Disease/val"

inception = InceptionV3(input_shape= IMAGE_SIZE + [3], weights= 'imagenet', include_top= False)

##now we don't train existing weights
for layer in inception.layers:
  layer.trainable = False

##Let's see the number of ouput folder using glob function
folders = glob("/content/drive/My Drive/Cotton_disease/Cotton_Disease/train/*")

###Now flatten the layers
x = Flatten()(inception.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

##create a model object
model = Model(inputs = inception.input, outputs = prediction)

##let's view the structure of the model
model.summary()

##tell the model what cost & optimization method to use
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Using imagedatagenerator function we import images from dataset
train_datagen = ImageDataGenerator(rescale= 1./255, shear_range= 0.2, zoom_range= 0.2, horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)

##Remembered that we should provide same target size as initialized for the image size.
training_set = train_datagen.flow_from_directory("/content/drive/My Drive/Cotton_disease/Cotton_Disease/train", target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("/content/drive/My Drive/Cotton_disease/Cotton_Disease/val", target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

#now we train the model using fit function

result = model.fit_generator(training_set, validation_data = test_set, epochs = 20, steps_per_epoch = len(training_set), validation_steps = len(test_set))

import matplotlib.pyplot as plt

# plot the loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

##now we save the model in h5.

from tensorflow.keras.models import load_model
model.save("cotton_disease_model_inception.h5")

y_pred = model.predict(test_set)

import numpy as np
y_pred = np.argmax(y_pred, axis = 1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model_inception.h5')

img = image.load_img("/content/drive/My Drive/Cotton_disease/Cotton_Disease/train/diseased cotton leaf/dis_leaf (1)_iaip.jpg", target_size= (224, 224))

x = image.img_to_array(img)
print(x)

print(x.shape)

x = x/255

import numpy as np
x = np.expand_dims(x, axis= 0)
img_data = preprocess_input(x)
print(img_data.shape)

model.predict(img_data)

a = np.argmax(model.predict(img_data), axis = 1)
print(a)

if a == 0:
  print("It is diseased cotton leaf")
elif a == 1 :
  print("It is diseased cotton plant")
elif a == 2:
  print("It is freshly cotton leaf")
else:
  print("It is freshly cotton plant")

























































