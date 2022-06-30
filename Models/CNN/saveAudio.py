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

warnings.filterwarnings("ignore")
train_audio_path = 'D:/Phd/MAHSA_PROJECT/Dataset_speechRecog/train/audio'
labels = os.listdir(train_audio_path)


# no_of_recordings = []
# for label in labels:
#     waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
#     no_of_recordings.append(len(waves))

# # plot
# plt.figure(figsize=(10, 5))
# index = np.arange(len(labels))
# plt.bar(index, no_of_recordings)
# plt.xlabel('Commands', fontsize=12)
# plt.ylabel('No of recordings', fontsize=12)
# plt.xticks(index, labels, fontsize=10, rotation=0)
# plt.title('No. of recordings for each command')
# plt.show()
#
# duration_of_recordings = []
# for label in labels:
#     waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
#     print(label)
#     for wav in waves:
#         samples,sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav)
#         duration_of_recordings.append(float(len(samples) / sample_rate))
#         print(wav)
#
# plt.hist(np.array(duration_of_recordings))
# plt.show()

def padfunc(offset, samples, sample_rate, fsnew):
    pad_len = int(np.ceil((offset - (len(samples) / sample_rate)) * fsnew))
    padding = np.zeros(pad_len)
    samples_ = np.concatenate((samples, padding))
    if len(samples_) / fsnew > offset:
        samples_ = librosa.effects.time_stretch(samples, len(samples) / fsnew)
    return samples_



all_wave = []
all_label = []
fsnew = 8000
offset = 1
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, fsnew)
        # print('original len :', len(samples))
        if len(samples) / fsnew < offset:
            samples = padfunc(offset, samples, sample_rate, fsnew)
            print('after padding :', len(samples))
        elif len(samples) / fsnew > offset:
            samples = librosa.effects.time_stretch(samples, len(samples) / fsnew)
            print('after stretch :', len(samples))
        all_wave.append(samples)
        all_label.append(label)
X = np.zeros([len(all_wave), fsnew])
for i in range(len(all_wave)):
    X[i, :] = all_wave[i]

Y = np.array(all_label)
np.save('train_data', X)
np.save('train_label', Y)
