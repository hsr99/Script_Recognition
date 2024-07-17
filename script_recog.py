from __future__ import division
import numpy as np
import os
import glob
from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ZeroPadding2D, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import cv2

def parse_line(line):
    parts = line.strip().split()
    image_filename = parts[0] + '.png'
    x, y, w, h = map(int, parts[3:7]) 
    transcription = ' '.join(parts[8:])
    return image_filename, (x, y, w, h), transcription

def load_image(image_filename):
    image = cv2.imread(f'C:\\Users\\hridy\\OneDrive\\Documents\\Sem4\\ML Project\\IAM_HW\\{image_filename}')
    if image is None:
        print(f"Error: Failed to load image '{image_filename}'")
        return None
    return image

dataset = []
img_files=[]
img_targets=[]
with open(r'C:\Users\hridy\OneDrive\Documents\Sem4\ML Project\labels.txt', 'r') as file:
    for line in file:
        if not line.startswith('#'):
            image_filename, bounding_box, transcription = parse_line(line)
            image = load_image(image_filename)
            dataset.append((image))
            img_files.append(f'C:\\Users\\hridy\\OneDrive\\Documents\\Sem4\\ML Project\\IAM_HW\\{image_filename}')
            img_targets.append(transcription)


row, col, ch = 128, 128, 1  
num_classes = len(np.unique(img_targets))

def load_image(image_filename, target_size=(128, 128)):
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image '{image_filename}'")
        return None
 
    image = cv2.resize(image, target_size)
    return image

batch_size = 16
num_classes = 50
def generate_data(samples, target_files,  batch_size=batch_size, factor = 0.1 ):
    num_samples = len(samples)
    from sklearn.utils import shuffle
    while 1: 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]
            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]
                cur_height = im.size[1]
                height_fac = 113 / cur_height
                new_width = int(cur_width * height_fac)
                size = new_width, 113
                imresize = im.resize((size), Image.ANTIALIAS)  
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                avail_x_points = list(range(0, now_width - 113 ))
                pick_num = int(len(avail_x_points)*factor)
                random_startx = sample(avail_x_points,  pick_num)

                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113, 113))
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)
                
                if images:
                    X_train = np.array(images)
                    y_train = np.array(targets)
                    X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
                    X_train = X_train.astype('float32')
                    X_train /= 255
                    y_train = to_categorical(y_train, num_classes)
                    yield shuffle(X_train, y_train)
                    
# print(num_classes)
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image,[56,56])

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))
model.add(Lambda(resize_image))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(row, col, ch)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(512, name='dense1'))  #1024
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())

encoder = LabelEncoder()
encoded_targets = encoder.fit_transform(img_targets)

# %%
train_files, rem_files, train_targets, rem_targets = train_test_split(
    img_files, encoded_targets, train_size=0.66, random_state=52, shuffle=True)
validation_files, test_files, validation_targets, test_targets = train_test_split(
    rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

train_generator = generate_data(train_files, train_targets)
validation_generator = generate_data(validation_files, validation_targets)
test_generator = generate_data(test_files, test_targets)

# history = model.fit_generator(train_generator, steps_per_epoch=len(train_files)//16,
#                               epochs=10, validation_data=validation_generator,
#                               validation_steps=len(validation_files)//16)

loss, accuracy = model.evaluate_generator(test_generator, steps=len(test_files)//16)
print("Test Accuracy:", accuracy)
np.save("label_encoder_classes.npy", encoder.classes_)
model.save("handwriting_recognition_model.h5")



