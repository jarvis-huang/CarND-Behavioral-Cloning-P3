import os
import csv
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.core import Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import matplotlib.pyplot as plt


y_horizon = 60 # Horizon line y = 60
y_hood = 25 # Hood line y = bottom 25
st_correction = 0.2 # steer correction for left/right image
keep_prob = 0.5
arch = 1
use_left_right = False
#all_data_dirs = ['data1', 'data2', 'data_reverse', 'data_weave', 'data_weave_reverse', 'curves', 'curve_reverse']
#all_data_dirs = ['data', 'recovery']
#all_data_dirs = ['data']
all_data_dirs = ['curves', 'curve_reverse', 'recovery', 'data1']
img_shape = (160,320,3)
if use_left_right:
	factor = 6
else:
	factor = 2

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
                name = batch_sample[0].strip()
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                if use_left_right:
                    name = batch_sample[1].strip()
                    left_image = cv2.imread(name)
                    left_angle = center_angle+st_correction
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(np.fliplr(left_image))
                    angles.append(-left_angle)
                    name = batch_sample[2].strip()
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

model = load_model('good/model_nvidia3_ok.h5')
			
### train the model
adam = Adam(lr=0.0002)
model.compile(optimizer=adam, loss='mse')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*factor, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*factor, nb_epoch=5, verbose=1)

### save the model
model.save('model_nvidia.h5')
