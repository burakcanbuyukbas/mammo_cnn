from keras.models import load_model, Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

X_Train = np.load('data/0/X_calc_train_256GRY.npy')
X_Test = np.load('data/0/X_calc_test_256GRY.npy')
Y_Train = np.load('data/0/Y_calc_train.npy')
Y_Test = np.load('data/0/Y_calc_test.npy')

image_size = 256

X_Train = X_Train.reshape((X_Train.shape[0], image_size, image_size, 1))
X_Test = X_Test.reshape((X_Test.shape[0], image_size, image_size, 1))

X_Train = X_Train.astype('uint16') / 256
X_Test = X_Test.astype('uint16') / 256

X_Train = np.repeat(X_Train, 3, axis=3)
X_Test = np.repeat(X_Test, 3, axis=3)

X_Val = X_Test[:206, :, :, :]
Y_Val = Y_Test[0: 206]
X_Test = X_Test[206:, :, :, :]
Y_Test = Y_Test[206:]

print("X_Val: " + str(X_Val.shape))
print("Y_Val: " + str(Y_Val.shape))
print("X_Test: " + str(X_Test.shape))
print("Y_Test: " + str(Y_Test.shape))

np.save('data/calc/X_train_256_GRY.npy', X_Train)
np.save('data/calc/X_val_256_GRY.npy', X_Val)
np.save('data/calc/X_test_256_GRY.npy', X_Test)
np.save('data/calc/Y_train.npy', Y_Train)
np.save('data/calc/Y_val.npy', Y_Val)
np.save('data/calc/Y_test.npy', Y_Test)