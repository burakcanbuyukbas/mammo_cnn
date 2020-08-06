import numpy as np
import cv2
import skimage.color
import skimage.filters
import skimage.io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import adadelta, adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from preprocess_data import load_data

image_size = 256

X_Train_Mass = np.load('X_mass_train_' + str(image_size) + '.npy')
X_Train_Calc = np.load('X_calc_train_' + str(image_size) + '.npy')
X_Test_Calc = np.load('X_calc_test_' + str(image_size) + '.npy')
X_Test_Mass = np.load('X_mass_test_' + str(image_size) + '.npy')
Y_Train_Mass = np.load('Y_mass_train.npy')
Y_Train_Calc = np.load('Y_calc_train.npy')
Y_Test_Calc = np.load('Y_calc_test.npy')
Y_Test_Mass = np.load('Y_mass_test.npy')
print("Mass Train Benign: " + str(np.count_nonzero(Y_Train_Mass == 0)))
print("Mass Train Malignant: " + str(np.count_nonzero(Y_Train_Mass == 1)))
print("Calc Train Benign: " + str(np.count_nonzero(Y_Train_Calc == 0)))
print("Calc Train Malignant: " + str(np.count_nonzero(Y_Train_Calc == 1)))
print("Mass Test Benign: " + str(np.count_nonzero(Y_Test_Mass == 0)))
print("Mass Test Malignant: " + str(np.count_nonzero(Y_Test_Mass == 1)))
print("Calc Test Benign: " + str(np.count_nonzero(Y_Test_Calc == 0)))
print("Calc Test Malignant: " + str(np.count_nonzero(Y_Test_Calc == 1)))



print("X_Train_Mass.shape = " + str(X_Train_Mass.shape))
X_Train_Mass_count = X_Train_Mass.shape[2]

print("X_Test_Mass.shape = " + str(X_Test_Mass.shape))


X_Val_Mass = X_Test_Mass[:189, :, :, :]
Y_Val_Mass = Y_Test_Mass[0: 189]
X_Test_Mass = X_Test_Mass[189:, :, :, :]
Y_Test_Mass = Y_Test_Mass[189:]

print(X_Train_Mass.shape)
print(Y_Train_Mass.shape)
print(X_Test_Mass.shape)
print(Y_Test_Mass.shape)
print(X_Val_Mass.shape)
print(Y_Val_Mass.shape)

batches_per_epoch = int(1318 / 32)
print("Steps per epoch: ", batches_per_epoch)
lr_decay = (1./0.80 - 1) / batches_per_epoch

epoch_count = 30
model_cnn = Sequential()

#conv layer 1
model_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#conv layer 2
model_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#conv layer 3
model_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#max pool 1
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_cnn.add(Dropout(0.5))

#conv layer 4
model_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#conv layer 5
model_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))

#max pool 2
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model_cnn.add(Dropout(0.5))


#conv layer 6
model_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#conv layer 7
model_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))

#max pool 3
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



#conv layer 8
model_cnn.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
#conv layer 9
#model_cnn.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

#max pool 4
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))


#conv layer 10
model_cnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))

##conv layer 11
#model_cnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

#max pool 5
model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

##conv layer 12
#model_cnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
#conv layer 13
#model_cnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))


#max pool 6
#model_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool6'))


# transpose to 256
#unpool1 = Conv2DTranspose(256, kernel_size=(8, 8), strides=(4, 4), activation='elu')
#model_cnn.add(unpool1)
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

#unpool1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
#model_cnn.add(unpool1)
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
#
#unpool2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
#model_cnn.add(unpool2)
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

#unpool2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
#model_cnn.add(unpool2)
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
##
#merged = Concatenate([unpool1, unpool2])

#model_cnn.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), activation='elu'))

#model_cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.#add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))


#model_cnn.add(Conv2D(56, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))


#model_cnn.add(Concatenate(axis=-1))


#model_cnn.add(Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), activation='elu'))


#model_cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
#
#model_cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
#
#model_cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

model_cnn.add(Flatten())
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(2048, activation=None))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(2048, activation=None))
model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))
model_cnn.add(Activation('relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(2, activation=None))

#model_cnn.add(Conv2D(2, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
#model_cnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-8, center=True, scale=True))

#model_cnn.add(Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), activation='elu'))
#
#model_cnn.add(Conv2DTranspose(2, kernel_size=(3, 3), strides=(1, 1), activation='softmax'))

opt = adam(lr=0.001, decay=lr_decay)

model_cnn.compile(optimizer=opt, loss=sparse_categorical_crossentropy, metrics=['accuracy'])


model_cnn.fit(X_Train_Mass, Y_Train_Mass,
              batch_size=32,
              epochs=epoch_count,
              verbose=1,
              validation_data=(X_Val_Mass, Y_Val_Mass))


model_cnn.save(str(image_size) + "px_" + str(epoch_count) + "epoch" + '_mass' + ".h5")
score = model_cnn.evaluate(X_Test_Mass, Y_Test_Mass, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])