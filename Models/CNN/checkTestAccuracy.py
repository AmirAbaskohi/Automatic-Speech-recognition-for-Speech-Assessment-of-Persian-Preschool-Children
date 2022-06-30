
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
import IPython.display as ipd
from sklearn.metrics import confusion_matrix
from pylab import rcParams
import matplotlib.pyplot as plt
from keras.utils import np_utils
# load model
model = load_model('model.h5')
# summarize model.
model.summary()

all_wave = np.load('train_data.npy')
all_label = np.load('train_label.npy')

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)



def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

fsnew = 8000
offset = 1
def paddOrStretch(offset, samples, sample_rate, fsnew):
    if len(samples) / fsnew < offset:
        pad_len = int(np.ceil((offset - (len(samples) / sample_rate)) * fsnew))
        padding = np.zeros(pad_len)
        samples = np.concatenate((samples, padding))
    if len(samples) / fsnew > offset:
        samples = librosa.effects.time_stretch(samples, len(samples) / fsnew)
    return samples

#reading the voice commands
filepath ='D:/Phd/MAHSA_PROJECT/convertedToWav/oneWord.wav'
samples, sample_rate = librosa.load(filepath , sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate=8000)
correct_sample = paddOrStretch(offset, samples, sample_rate, fsnew)
print(predict(correct_sample))

x_val = np.load('test_data.npy')
y_val = np.load('test_label.npy')
labels = ['abi', 'ghermez', 'meshki', 'sabz', 'siyah', 'zard']

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
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# n_classes = 6
# plt.figure(figsize=(30, 30))
# rcParams['figure.figsize'] = 12, 12
# plt.matshow(cm)
# plt.colorbar()
# plt.title('Confusion matrix of the classifier')
# tick_marks = np.arange(n_classes)
# plt.xticks(tick_marks, range(n_classes))
# plt.yticks(tick_marks, range(n_classes))
# # plt.set_xticklabels([''] + labels)
# # plt.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
