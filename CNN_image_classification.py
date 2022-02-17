#import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D


emotion_l = [0,1,2,3,4,5,6]
emotion_n = ['Angry', 'Disgust','Fear','Happy','Sad','Surprise','Neutral']

train  = pd.read_csv('Q1_Train_Data.csv')
image_p = train["pixels"]
emotion = train["emotion"]

train_images = np.stack(train["pixels"].str.split().values , axis = 0).astype(np.float32)
train_images  = (train_images/255) - 0.5

##Validation test model
validation  = pd.read_csv('Q1_Validation_Data.csv')
image_p_val = validation["pixels"]
emotion_val = validation["emotion"]
validation_images = np.stack(validation["pixels"].str.split().values , axis = 0).astype(np.float32)
validation_images  = (validation_images/255) - 0.5

test  = pd.read_csv('Q1_Test_Data.csv')
image_p_test = validation["pixels"]
emotion_test = validation["emotion"]
test_images = np.stack(test["pixels"].str.split().values , axis = 0).astype(np.float32)
test_images  = (test_images/255) - 0.5

train_images_3d = train_images.reshape(len(train),48,48,1)
test_images_3d = test_images.reshape(len(test),48,48,1)
validate_images_3d = validation_images.reshape(len(validation),48,48,1)
print(train_images_3d.shape)

# Define a fetaure extraction model that is shared for both mnist and fashion-mnist tasks
Base_feature_model = Sequential([Conv2D(12, kernel_size=2, activation='relu',input_shape=(48,48,1)),
            #MaxPooling2D(pool_size=(2,2)),
            Conv2D(12, kernel_size=2, activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            #Conv2D(14, kernel_size=3, activation='relu'),
            #Conv2D(32, kernel_size=3, activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(), Dense(512, activation='relu'),])


#Classifier_mnist = Sequential([Dense(10, activation='softmax')])
Classifier_fer = Sequential([Dense(7, activation='softmax')])

# Instantiate a Tensor to feed Input (Input Layer)
mnist_input = Input(shape=(28,28,1))
#mnist_features = Base_feature_model(mnist_input)
#mnist_prediction = Classifier_mnist(mnist_features)

#train_images_3d = train_images.reshape(-1,48,48)
#train_images_3d = train_images_3d.astype('float32') / 255
fer_features = Base_feature_model(train_images_3d)
fer_prediction = Classifier_fer(emotion.values)

joint_model = Model(inputs=[mnist_input, fer_input],
                    outputs=[mnist_prediction, fer_prediction])

print(joint_model.summary())
