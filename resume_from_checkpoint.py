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

metrics = {
    'tp': keras.metrics.TruePositives(name='tp'),
    'fp': keras.metrics.FalsePositives(name='fp'),
    'tn': keras.metrics.TrueNegatives(name='tn'),
    'fn': keras.metrics.FalseNegatives(name='fn'),
    'accuracy': keras.metrics.BinaryAccuracy(name='accuracy'),
    'precision': keras.metrics.Precision(name='precision'),
    'recall': keras.metrics.Recall(name='recall'),
    'auc': keras.metrics.AUC(name='auc'),
}
class_weights = {0: 0.5, 1: 1.0}

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

batches_per_epoch = int(X_Train.shape[0] / 32)
print("batches_per_epoch= " + str(batches_per_epoch))
val_batches_per_epoch = int(X_Val.shape[0] / 32)

print("validation batches_per_epoch= " + str(val_batches_per_epoch))
print("Steps per epoch: ", batches_per_epoch)
lr_decay = (1./0.80 - 1) / batches_per_epoch

epoch_count = 25


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
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=3, verbose=1)

filepath="checkpoints/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

callbacks = [reduce_lr, early_stopping, checkpointer]

#top resnet50 layer
top_layer_nb = 162

print("Loading model...")
model = ResNet(weights='imagenet', include_top=False,
                  input_shape=None, pooling='avg')
x = model.output
x = Dropout(0.5)(x)
preds = Dense(2, activation='softmax',
              kernel_regularizer=l2(0.001))(x)
model = Model(inputs=model.input, outputs=preds)
model.load_weights("checkpoints/checkpoint-08-0.65.hdf5")
print("Model loaded.")



loss_history = []
acc_history = []


# print("Stage 2:")
# for layer in model.layers[top_layer_nb:]:
#     layer.trainable = True
#
#
# dense_layer = model.layers[-1]
# dropout_layer = model.layers[-2]
# dense_layer.kernel_regularizer.l2 = 0.01
# dropout_layer.rate = .5
# model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=METRICS)
# #old: 0.001
#
# hist = model.fit(
#     train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
#     steps_per_epoch=len(X_Train) / batch_size,
#     epochs=10,
#     class_weight=class_weights,
#     shuffle=True,
#     validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
#     callbacks=callbacks,
#     verbose=2)
#
# print("Second stage done.")
# try:
#     loss_history = np.append(loss_history, hist.history['val_loss'])
#     acc_history = np.append(acc_history, hist.history['val_acc'])
# except KeyError:
#     pass

# Stage 3:
print("Stage 3:")


for layer in model.layers:
    layer.trainable = True
dropout_layer = model.layers[-2]
dropout_layer.rate = .5
model.compile(optimizer=Adam(0.0000001),
              loss='categorical_crossentropy', metrics=METRICS)
print("Model compiled. Initiating training...")
# epoch = 10


hist = model.fit(
    train_generator.flow(X_Train, Y_Train, batch_size=batch_size),
    steps_per_epoch=len(X_Train) / batch_size,
    epochs=10,
    class_weight=class_weights,
    shuffle=True,
    validation_data=val_generator.flow(X_Val, Y_Val, batch_size=batch_size),
    callbacks=callbacks,
    verbose=2)

model.save(str(image_size) + "px_" + str(epoch_count) + "epoch_" + "5" + ".h5")
