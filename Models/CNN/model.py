import os
import librosa  # for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile  # for audio processing
import warnings
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix
from pylab import rcParams

warnings.filterwarnings("ignore")
train_audio_path = 'D:/Phd/MAHSA_PROJECT/Dataset_speechRecog/train/audio'
labels = os.listdir(train_audio_path)

all_wave = np.load('train_data.npy')
all_label = np.load('train_label.npy')

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

y = np_utils.to_categorical(y, num_classes=len(labels))
all_wave = np.array(all_wave).reshape(-1, 8000, 1)
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2,
                                            random_state=777, shuffle=True)
np.save('test_data', x_val)
np.save('test_label', y_val)
K.clear_session()

inputs = Input(shape=(8000, 1))

# First Conv1D layer
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

# Flatten layer
conv = Flatten()(conv)

# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_split=0.2)


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title("Model accuracy for color audio classification")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for color audio classification')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

model.save("model.h5")

def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


# import random
# index=random.randint(0,len(x_val)-1)
# samples=x_val[index].ravel()
# print("Audio:",classes[np.argmax(y_val[index])])
# ipd.Audio(samples, rate=8000)
# print("Text:",predict(samples))

y_pred = []
y_real = []
y_pred_label = []
y_real_label = []
acc = 0
for index in range(len(x_val) - 1):
    samples = x_val[index].ravel()
    _real = labels.index(classes[np.argmax(y_val[index])])
    _pred = labels.index(predict(samples))
    if _pred == _real:
        acc += 1
    y_pred.append(_pred)
    y_real.append(_real)
    y_pred_label.append(predict(samples))
    y_real_label.append(classes[np.argmax(y_val[index])])

test_accuracy = acc / len(y_pred)
print("Test accuracy = {}".format(test_accuracy))

confusion_mtx = confusion_matrix(y_real, y_pred)
print(confusion_mtx)

cm = confusion_matrix(y_real_label, y_pred_label,labels)
n_classes = 6
plt.figure(figsize=(30, 30))
rcParams['figure.figsize'] = 12, 12
plt.matshow(cm)
plt.colorbar()
plt.title('Confusion matrix of the classifier')
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, range(n_classes))
plt.yticks(tick_marks, range(n_classes))
plt.set_xticklabels([''] + labels)
plt.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
