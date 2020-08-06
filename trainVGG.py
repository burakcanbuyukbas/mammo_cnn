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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Activation,\
    Concatenate
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import load_model, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from preprocess_data import get_data
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.metrics
from sklearn.utils import class_weight
from utils import scheduler

image_size = 256
method = 0
batch_size = 8

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]


#get_data(save_data=True, method=method)

X_Train = np.load('data/X_train_256_GRY.npy')
X_Val = np.load('data/X_val_256_GRY.npy')
X_Test = np.load('data/X_test_256_GRY.npy')
Y_Train = np.load('data/Y_train.npy')
Y_Val = np.load('data/Y_val.npy')
Y_Test = np.load('data/Y_test.npy')
print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))

print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))



print("X_Train shape: " + str(X_Train.shape))
print("Y_Train shape: " + str(Y_Train.shape))
print("X_Test shape: " + str(X_Test.shape))
print("Y_Test shape: " + str(Y_Test.shape))
print("X_Val shape: " + str(X_Val.shape))
print("Y_Val shape: " + str(Y_Val.shape))

batches_per_epoch = int(X_Train.shape[0] / batch_size)
print("batches_per_epoch= " + str(batches_per_epoch))
val_batches_per_epoch = int(X_Val.shape[0] / batch_size)




print("validation batches_per_epoch= " + str(val_batches_per_epoch))
print("Steps per epoch: ", batches_per_epoch)

epoch_count = 25

class_weights = {0: 0.5, 1: 1.0}


#data Augmentation
train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=180,
    shear_range=15,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')
val_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=180,
    shear_range=15,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')
train_generator.fit(X_Train)
val_generator.fit(X_Val)
# Create callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               verbose=1, mode='min')



#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
reduce_lr = LearningRateScheduler(scheduler)

filepath="checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

callbacks = [reduce_lr, early_stopping, checkpointer]

vgg = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(image_size, image_size, 3))

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Freeze the convolutional base
vgg.trainable = False


opt = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

# Train
history = model.fit(
    train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
    steps_per_epoch=len(X_Train) / batch_size,
    epochs=14,
    class_weight=class_weights,
    shuffle=True,
    validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
    callbacks=callbacks,
    verbose=2
)


model.save("models/vgg.h5")
