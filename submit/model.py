import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.core import Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt
from keras import optimizers

y_horizon = 60 # Horizon line y = 60
y_hood = 25 # Hood line y = bottom 25
st_correction = 0.2 # steer correction for left/right image
keep_prob = 0.5
arch = 3 # 1-Lenet, 3-NVIDIA
use_left_right = True
if use_left_right:
	factor = 6
else:
	factor = 2
#all_data_dirs = ['data1', 'data2', 'data_reverse', 'data_weave', 'data_weave_reverse', 'curves', 'curve_reverse']
#all_data_dirs = ['data', 'recovery']
all_data_dirs = ['data']
img_shape = (160,320,3)

samples = []
for data_dir in all_data_dirs:
	with open(os.path.join(data_dir,'driving_log.csv')) as csvfile:
		reader = csv.reader(csvfile)
		next(reader) # skip first row
		for line in reader:
			samples.append(line)
			
import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                if use_left_right:
                    name = './data/IMG/'+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = center_angle+st_correction
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(np.fliplr(left_image))
                    angles.append(-left_angle)
                    name = './data/IMG/'+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)
                    right_angle = center_angle-st_correction
                    images.append(right_image)
                    angles.append(right_angle)
                    images.append(np.fliplr(right_image))
                    angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


if arch==1: # LeNet
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_shape))
	# Crop out above horizon line and below hood line
	model.add(Cropping2D(cropping=((y_horizon, y_hood), (0, 0))))
	#model.add(AveragePooling2D()) # downsample by 2
	model.add(Convolution2D(6, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Convolution2D(16, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dropout(keep_prob))
	model.add(Activation('relu'))
	model.add(Dense(84))
	model.add(Dropout(keep_prob))
	model.add(Activation('relu'))
	model.add(Dense(1))
elif arch==3: # NVIDIA
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_shape))
	# Crop out above horizon line and below hood line
	model.add(Cropping2D(cropping=((y_horizon, y_hood), (0, 0))))
	#model.add(AveragePooling2D()) # downsample by 2
	model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))	
	model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))		
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(keep_prob))
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Dropout(keep_prob))
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Dropout(keep_prob))
	model.add(Activation('relu'))	
	model.add(Dense(1))
	
			
### train the model
model.compile(optimizer='adam', loss='mse')
#history_object = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True, verbose=1)
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*factor, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*factor, nb_epoch=8, verbose=1)


### save the model
model.save('model_nvidia.h5')
