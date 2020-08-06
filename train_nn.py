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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from preprocess_data import get_data
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.metrics
from sklearn.utils import class_weight


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

if(method==0):
    X_Train = np.load('X_train_' + str(image_size) + '_2.npy')
    X_Test = np.load('X_test_' + str(image_size) + '_2.npy')
    Y_Train = np.load('Y_train.npy')
    Y_Test = np.load('Y_test.npy')
    print("Train Benign: " + str(np.count_nonzero(Y_Train == 0)))
    print("Train Malignant: " + str(np.count_nonzero(Y_Train == 1)))

    print("Test Benign: " + str(np.count_nonzero(Y_Test == 0)))
    print("Test Malignant: " + str(np.count_nonzero(Y_Test == 1)))



    X_Val = X_Test[:320, :, :, :]
    Y_Val = Y_Test[0: 320]
    X_Test = X_Test[320:, :, :, :]
    Y_Test = Y_Test[320:]

    Y_Train = Y_Train.astype(bool)
    Y_Val = Y_Val.astype(bool)
    Y_Test = Y_Test.astype(bool)

    # X_Train = X_Train.reshape([X_Train.shape[0], image_size, image_size, 1])
    # X_Test = X_Test.reshape([X_Test.shape[0], image_size, image_size, 1])
    # X_Val = X_Val.reshape([X_Val.shape[0], image_size, image_size, 1])

    Y_Train = to_categorical(Y_Train)
    Y_Val = to_categorical(Y_Val)
    Y_Test = to_categorical(Y_Test)


    print("X_Train shape: " + str(X_Train.shape))
    print("Y_Train shape: " + str(Y_Train.shape))
    print("X_Test shape: " + str(X_Test.shape))
    print("Y_Test shape: " + str(Y_Test.shape))
    print("X_Val shape: " + str(X_Val.shape))
    print("Y_Val shape: " + str(Y_Val.shape))

    batches_per_epoch = int(X_Train.shape[0] / batch_size)
    print("batches_per_epoch= " + str(batches_per_epoch))
    val_batches_per_epoch = int(X_Val.shape[0] / batch_size)



else:
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

    Y_Train_Mass = Y_Train_Mass.astype(bool)
    Y_Val_Mass = Y_Val_Mass.astype(bool)
    Y_Test_Mass = Y_Test_Mass.astype(bool)

    print("X_Train_Mass shape: " + str(X_Train_Mass.shape))
    print("Y_Train_Mass shape: " + str(Y_Train_Mass.shape))
    print("X_Test_Mass shape: " + str(X_Test_Mass.shape))
    print("Y_Test_Mass shape: " + str(Y_Test_Mass.shape))
    print("X_Val_Mass shape: " + str(X_Val_Mass.shape))
    print("Y_Val_Mass shape: " + str(Y_Val_Mass.shape))

    batches_per_epoch = int(X_Train_Mass.shape[0] / batch_size)
    print("batches_per_epoch= " + str(batches_per_epoch))
    val_batches_per_epoch = int(X_Val_Mass.shape[0] / batch_size)

    X_Train = X_Train_Mass
    X_Test = X_Test_Mass
    X_Val = X_Val_Mass
    Y_Train = Y_Train_Mass
    Y_Test = Y_Test_Mass
    Y_Val = Y_Val_Mass



print("validation batches_per_epoch= " + str(val_batches_per_epoch))
print("Steps per epoch: ", batches_per_epoch)
lr_decay = (1./0.80 - 1) / batches_per_epoch

epoch_count = 25

class_weights = {0: 0.5, 1: 1.0}


def batch_generator(X, Y, batch_size):
    indices = np.arange(len(X))
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]

#data Augmentation
train_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    rotation_range=25,
    shear_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    vertical_flip=True)
val_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    rotation_range=25,
    shear_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    vertical_flip=True)
train_generator.fit(X_Train)
val_generator.fit(X_Val)
# Create callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               verbose=1, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, verbose=1)

filepath="checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

callbacks = [reduce_lr, early_stopping, checkpointer]

#top resnet50 layer
top_layer_nb = 162

model = ResNet(weights='imagenet', include_top=False,
                  input_shape=None, pooling='avg')
x = model.output
x = Dropout(0.5)(x)
preds = Dense(2, activation='softmax',
              kernel_regularizer=l2(0.001))(x)
model = Model(inputs=model.input, outputs=preds)





# Stage 1:
# Train on the last dense layer
print("Stage 1:")
for layer in model.layers[:-1]:
    layer.trainable = False
model.compile(optimizer=Adam(0.001),
              loss='categorical_crossentropy', metrics=METRICS)

hist = model.fit(
    train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
    steps_per_epoch=len(X_Train) / batch_size,
    epochs=5,
    class_weight=class_weights,
    shuffle=True,
    validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
    callbacks=callbacks,
    verbose=2)
print("First stage done.")

try:
    loss_history = hist.history['val_loss']
    acc_history = hist.history['val_acc']
except KeyError:
    loss_history = []
    acc_history = []

# Stage 2:
# Train on the top layers
print("Stage 2:")
for layer in model.layers[top_layer_nb:]:
    layer.trainable = True


dense_layer = model.layers[-1]
dropout_layer = model.layers[-2]
dense_layer.kernel_regularizer.l2 = 0.01
dropout_layer.rate = .5
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=METRICS)
#old: 0.001

hist = model.fit(
    train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
    steps_per_epoch=len(X_Train) / batch_size,
    epochs=10,
    class_weight=class_weights,
    shuffle=True,
    validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
    callbacks=callbacks,
    verbose=2)

print("Second stage done.")
try:
    loss_history = np.append(loss_history, hist.history['val_loss'])
    acc_history = np.append(acc_history, hist.history['val_acc'])
except KeyError:
    pass

# Stage 3:
print("Stage 3:")
for layer in model.layers:
    layer.trainable = True
dropout_layer.rate = .5
model.compile(optimizer=Adam(0.0000001),
              loss='categorical_crossentropy', metrics=METRICS)

hist = model.fit(
    train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
    steps_per_epoch=len(X_Train) / batch_size,
    epochs=10,
    class_weight=class_weights,
    shuffle=True,
    validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
    callbacks=callbacks,
    verbose=2)

print("Third stage done. Please be good.")
try:
    loss_history = np.append(loss_history, hist.history['val_loss'])
    acc_history = np.append(acc_history, hist.history['val_acc'])
except KeyError:
    pass



#score = model.evaluate(X_Test, Y_Test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

model.save(str(image_size) + "px_" + str(epoch_count) + "epoch_" + "6" + ".h5")
