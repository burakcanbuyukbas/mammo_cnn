from keras.models import load_model, Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50 as ResNet, preprocess_input
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import keras.metrics
from keras.optimizers import Adam

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


X_Test = np.load('X_test_' + str(256) + '_2.npy')
Y_Test = np.load('Y_test.npy')

X_Test = X_Test[320:, :, :, :]
Y_Test = Y_Test[320:]

Y_Test = Y_Test.astype(bool)
Y_Test = to_categorical(Y_Test)

#model = load_model(r"256px_25epoch_4.h5")
model = ResNet(include_top=False,
                  input_shape=None, pooling='avg')
x = model.output
x = Dropout(0.5)(x)
preds = Dense(2, activation='softmax',
              kernel_regularizer=l2(0.001))(x)
model = Model(inputs=model.input, outputs=preds)
model.load_weights("checkpoints/checkpoint-01-0.69.hdf5")
test_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=25,
    shear_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    vertical_flip=True)
test_generator.fit(X_Test)

model.compile(optimizer=Adam(0.0000001),
              loss='categorical_crossentropy', metrics=METRICS)

score = model.evaluate(test_generator.flow(
        X_Test,
        Y_Test,
        batch_size=8),
    steps=len(X_Test) / 8)
print('Test Accuracy:', score[0])
print('Binary Accuracy:', score[5])
print('Precision:', score[6])
print('Recall:', score[7])
print('AUC:', score[8])

# y_pred = model.predict(
#     test_generator.flow(
#         X_Test,
#         Y_Test,
#         batch_size=8),
#     steps=len(X_Test) / 8)
#
# y_pred = np.argmax(y_pred, axis=1)
# Y_Test = np.argmax(Y_Test, axis=1)
# print(confusion_matrix(Y_Test, y_pred, labels=[0, 1]))
# print(classification_report(Y_Test, y_pred))
